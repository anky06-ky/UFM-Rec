"""ID-only causal SASRec baseline on the fixed temporal validation protocol.

Each train row is a prefix and its next item. Negatives exclude all of that
user's train positives. This is an adapted baseline, not the paper's code.
Test labels are never loaded by this command. Unknown item IDs have no trained
embedding; their candidate score falls back to train popularity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_content_baseline import REGIMES, summarize, tie_noise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/toys_games_full_temporal'
LEGACY = ROOT / 'data/processed/gpu_cache_h20_t64_v1'
GRAPH = ROOT / 'data/processed/ufm_positive_graph_h20_v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class SASRec(nn.Module):
    """Causal self-attention over right-padded, one-based item histories."""

    def __init__(self, items, history, dim=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        if min(items, history, dim, heads, layers) < 1 or dim % heads:
            raise ValueError('Invalid SASRec dimensions.')
        self.history = history
        self.dim = dim
        self.items = nn.Embedding(items + 1, dim, padding_idx=0)
        self.position = nn.Embedding(history, dim)
        nn.init.normal_(self.items.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[0].zero_()
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def encode_tokens(self, history):
        if history.ndim != 2 or history.shape[1] != self.history:
            raise ValueError('Expected [batch,history] item IDs.')
        occupied = history.ne(0)
        length = occupied.sum(-1)
        padding = ~occupied
        padding = padding.clone()
        padding[:, 0] = False  # Avoid all-masked attention for empty histories.
        position = torch.arange(self.history, device=history.device)
        vectors = self.items(history) * self.dim**0.5
        vectors = vectors + self.position(position) * occupied.unsqueeze(-1)
        causal = torch.ones(self.history, self.history, dtype=torch.bool,
                            device=history.device).triu(1)
        hidden = self.encoder(vectors, mask=causal, src_key_padding_mask=padding)
        return hidden, length

    def encode(self, history):
        hidden, length = self.encode_tokens(history)
        last = hidden[torch.arange(len(history), device=history.device),
                      length.clamp_min(1) - 1]
        return self.norm(last) * length.gt(0).unsqueeze(-1)

    def logits(self, history, candidates):
        user = self.encode(history)
        item = self.items(candidates)
        return (user.unsqueeze(1) * item).sum(-1)


def sample_negatives(eligible, keys, indptr, positives, users, targets, rng, number):
    """Sample train-observed IDs only, excluding every train positive per user."""
    if number < 1:
        raise ValueError('Need at least one negative.')
    rows = np.searchsorted(keys, users)
    if np.any(rows >= len(keys)) or np.any(keys[rows] != users):
        raise ValueError('Training user absent from positive graph.')
    result = np.empty((len(users), number), dtype=np.int64)
    for j, row in enumerate(rows):
        forbidden = set(map(int, positives[indptr[row]:indptr[row + 1]]))
        forbidden.add(int(targets[j]))
        # Graph positives and targets are all train-observed IDs in eligible.
        if len(eligible) - len(forbidden) < number:
            raise ValueError('Insufficient train-unobserved items for negatives.')
        selected = set()
        for k in range(number):
            for _ in range(128):
                item = int(rng.choice(eligible))
                if item not in forbidden and item not in selected:
                    break
            else:
                available = np.asarray([x for x in eligible if x not in forbidden
                                        and x not in selected], dtype=np.int64)
                item = int(rng.choice(available))
            selected.add(item)
            result[j, k] = item
    return result


def load_sources(data, legacy, graph, history):
    lm, gm = read(legacy / 'complete.json'), read(graph / 'complete.json')
    if (lm['history_size'] != history or gm['history_size'] != history
            or lm['rows'] != gm['rows'] or lm['items'] != gm['items']
            or lm['fingerprint'] != gm['sources']['legacy_source']):
        raise ValueError('Legacy histories and positive graph disagree.')
    paths = ['histories.npy', 'targets.npy', 'users.npy', 'train_counts.npy']
    for name in paths:
        if sha256(legacy / name) != gm['sources'][name]:
            raise ValueError('Legacy source hash mismatch: ' + name)
    for name, expected in gm['graph_sha256'].items():
        if sha256(graph / name) != expected:
            raise ValueError('Positive graph hash mismatch: ' + name)
    arrays = {name: np.load(legacy / (name + '.npy'), mmap_mode='r')
              for name in ('histories', 'targets', 'users', 'train_counts')}
    arrays.update({name: np.load(graph / (name + '.npy'), mmap_mode='r')
                   for name in ('user_keys', 'positive_indptr', 'positive_items')})
    if (arrays['histories'].shape != (lm['rows'], history)
            or arrays['targets'].shape != (lm['rows'],)
            or arrays['users'].shape != (lm['rows'],)
            or arrays['train_counts'].shape != (lm['items'] + 1,)
            or not np.all(arrays['user_keys'][1:] > arrays['user_keys'][:-1])
            or np.any(arrays['targets'] < 1)
            or np.any(arrays['targets'] > lm['items'])):
        raise ValueError('Invalid source array shape, keys, or IDs.')
    fingerprint = dict(legacy=sha256(legacy / 'complete.json'),
                       graph=sha256(graph / 'complete.json'),
                       validation=sha256(data / 'evaluation/samples_validation.npz'))
    return arrays, fingerprint, lm


@torch.no_grad()
def evaluate(model, samples, counts, history, batch_size, cap=0):
    model.eval()
    device = next(model.parameters()).device
    selected = np.arange(len(samples['candidates']))
    if cap:
        # Stratified diagnostic subset; production selection uses every row.
        groups = [selected[samples['regime_codes'] == i][:max(1, cap // 4)]
                  for i in range(4)]
        selected = np.sort(np.concatenate(groups))
    ranks, lengths = [], []
    for start in range(0, len(selected), batch_size):
        indices = selected[start:start + batch_size]
        candidates = samples['candidates'][indices]
        ids = torch.as_tensor(candidates.astype(np.int64) + 1, device=device)
        h = np.zeros((len(indices), history), dtype=np.int64)
        batch_lengths = []
        for j, index in enumerate(indices):
            a, b = samples['history_indptr'][index:index + 2]
            prior = samples['history_indices'][max(a, b - history):b] + 1
            h[j, :len(prior)] = prior
            batch_lengths.append(int(b - a))
        histories = torch.as_tensor(h, device=device)
        scores = model.logits(histories, ids).float()
        count = torch.as_tensor(np.asarray(counts[candidates + 1]), device=device)
        scores = torch.where(count > 0, scores, 0)
        empty = histories.ne(0).sum(-1).eq(0)
        scores[empty] = count[empty].float().log1p()
        values = scores.cpu().numpy().astype(np.float64)
        values += tie_noise(candidates) * 1e-10
        ranks.extend((1 + (values[:, 1:] > values[:, :1]).sum(-1)).tolist())
        lengths.extend(batch_lengths)
    report = summarize(np.asarray(ranks), samples['regime_codes'][selected],
                       np.asarray(lengths))
    report['cold_macro_ndcg@10'] = float(np.mean(
        [report['by_regime'][name]['ndcg@10'] for name in REGIMES[:3]]))
    return report


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    arrays, sources, legacy = load_sources(args.data, args.legacy_cache,
                                           args.graph, args.history)
    config = dict(version=1, sources=sources, code_sha256=sha256(Path(__file__)),
                  epochs=args.epochs, batch_size=args.batch_size, dim=args.dim,
                  heads=args.heads, layers=args.layers, history=args.history,
                  negatives=args.negatives, lr=args.lr, patience=args.patience,
                  checkpoint_every=args.checkpoint_every, validation_cap=args.validation_cap,
                  eval_batch=args.eval_batch, max_steps=args.max_steps,
                  seed=args.seed, device=args.device, data=str(args.data),
                  legacy_cache=str(args.legacy_cache), graph=str(args.graph),
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  selection='Validation cold macro NDCG@10; fixed candidate set; no test')
    output = args.output
    if args.resume:
        if read(output / 'config.json') != config or (output / 'completed.json').exists():
            raise ValueError('Resume configuration changed or run already complete.')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output or use --resume.')
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(output / 'config.json', config)
    model = SASRec(legacy['items'], args.history, args.dim, args.heads,
                   args.layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    epoch0 = batch0 = steps = stale = 0
    best = -1.0
    if args.resume:
        state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch0, batch0, steps = state['epoch'], state['next_batch'], state['steps']
        best, stale = state['best'], state['stale']
        torch.set_rng_state(state['torch_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if (epoch0 >= args.epochs or stale >= args.patience
                or (args.max_steps and steps >= args.max_steps)):
            if not (output / 'best.pt').is_file():
                raise FileNotFoundError('Best checkpoint missing at completed resume state.')
            done = dict(epochs=epoch0, steps=steps,
                        best_cold_macro_ndcg_at_10=best,
                        smoke_run=bool(args.max_steps), test_evaluated=False,
                        time_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / 'completed.json', done)
            return done
    validation = np.load(args.data / 'evaluation/samples_validation.npz')
    counts = arrays['train_counts']
    eligible = np.flatnonzero(counts > 0).astype(np.int64)
    n = len(arrays['targets'])
    batches = (n + args.batch_size - 1) // args.batch_size
    started = time.monotonic()

    def checkpoint(epoch, next_batch):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch, next_batch=next_batch, steps=steps, best=best,
                    stale=stale, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [])

    if not args.resume:
        save_torch(output / 'latest.pt', checkpoint(0, 0))

    for epoch in range(epoch0, args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(n)
        losses = []
        for batch in range(batch0 if epoch == epoch0 else 0, batches):
            ids = order[batch * args.batch_size:(batch + 1) * args.batch_size]
            h = np.asarray(arrays['histories'][ids], dtype=np.int64)
            t = np.asarray(arrays['targets'][ids], dtype=np.int64)
            u = np.asarray(arrays['users'][ids], dtype=np.int64)
            rng = np.random.default_rng(args.seed + epoch * 1_000_003 + batch)
            neg = sample_negatives(eligible, arrays['user_keys'],
                                   arrays['positive_indptr'], arrays['positive_items'],
                                   u, t, rng, args.negatives)
            history = torch.as_tensor(h, device=args.device)
            candidates = torch.as_tensor(np.column_stack((t, neg)), device=args.device)
            logits = model.logits(history, candidates)
            loss = F.softplus(-logits[:, 0]).mean() + F.softplus(logits[:, 1:]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite SASRec loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            if steps % args.checkpoint_every == 0:
                save_torch(output / 'latest.pt', checkpoint(epoch, batch + 1))
            if args.max_steps and steps >= args.max_steps:
                break
        report = evaluate(model, validation, counts, args.history,
                          args.eval_batch, args.validation_cap)
        score = report['cold_macro_ndcg@10']
        improved = score > best + 1e-6
        best = max(best, score)
        stale = 0 if improved else stale + 1
        if improved:
            save_torch(output / 'best.pt', dict(model=model.state_dict(),
                       epoch=epoch + 1, validation=report, sources=sources))
        record = dict(epoch=epoch + 1, steps=steps,
                      train_loss=float(np.mean(losses)) if losses else None,
                      validation=report, best_cold_macro_ndcg_at_10=best,
                      elapsed_seconds=time.monotonic() - started)
        write_json(output / f'epoch_{epoch + 1:03d}.json', record)
        save_torch(output / 'latest.pt', checkpoint(epoch + 1, 0))
        print(json.dumps(record), flush=True)
        if stale >= args.patience or (args.max_steps and steps >= args.max_steps):
            break
    done = dict(epochs=epoch + 1, steps=steps, best_cold_macro_ndcg_at_10=best,
                smoke_run=bool(args.max_steps), test_evaluated=False,
                time_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'completed.json', done)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--legacy-cache', type=Path, default=LEGACY)
    parser.add_argument('--graph', type=Path, default=GRAPH)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/sasrec_s42_v1')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--history', type=int, default=20)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--negatives', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--eval-batch', type=int, default=128)
    parser.add_argument('--validation-cap', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (min(args.history, args.dim, args.heads, args.layers, args.batch_size,
            args.negatives, args.epochs, args.checkpoint_every, args.eval_batch,
            args.threads) < 1 or args.dim % args.heads or args.lr <= 0
            or args.patience < 1 or args.max_steps < 0 or args.validation_cap < 0):
        parser.error('Invalid positive sizes or attention head dimensions.')
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
