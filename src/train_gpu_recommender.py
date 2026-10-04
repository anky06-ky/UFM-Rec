"""Content-aware Transformer two-tower experiment, not a paper reproduction.

Train-only gradients; fixed validation candidates; test requires a separate command.
Uses existing TF-IDF vocabulary fitted on train items, plus train-only ID residuals.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F

from common import write_json
from evaluate_content_baseline import summarize, tie_noise

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/processed/toys_games_full_temporal'
VERSION = 1

def save_checkpoint(path, value):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def fingerprint(data):
    names = ['split_manifest.json', 'content/items.csv.gz', 'content/tfidf_all.npz',
             'train_with_history.csv.gz', 'evaluation/samples_validation.npz']
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        with (data / name).open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def prepare(data, cache, history_size, tokens):
    signature = fingerprint(data)
    wanted = {'version': VERSION, 'fingerprint': signature,
              'history_size': history_size, 'tokens': tokens}
    marker = cache / 'complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if any(saved.get(k) != v for k, v in wanted.items()):
            raise ValueError('Cache does not match source/config; choose a new --cache.')
        return saved
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError(f'Incomplete cache: {cache}. Choose a new --cache.')
    cache.mkdir(parents=True, exist_ok=True)
    ids, counts = {}, [0]
    with gzip.open(data / 'content/items.csv.gz', 'rt', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            if int(row['row_index']) != len(ids):
                raise ValueError('Non-contiguous item mapping')
            ids[row['parent_asin']] = len(ids) + 1
            counts.append(int(row['train_count']))
    counts = np.asarray(counts, dtype=np.int64)
    np.save(cache / 'train_counts.npy', counts)
    print(f'Preparing text features: {len(ids):,} items', flush=True)
    matrix = sparse.load_npz(data / 'content/tfidf_all.npz').tocsr()
    assert matrix.shape[0] == len(ids)
    term_ids = np.zeros((len(ids) + 1, tokens), dtype=np.int32)
    weights = np.zeros_like(term_ids, dtype=np.float32)
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i:i+2]
        values = matrix.data[start:end]
        cols = matrix.indices[start:end]
        select = np.argsort(-values, kind='stable')[:tokens]
        term_ids[i+1, :len(select)] = cols[select] + 1
        chosen = values[select]
        weights[i+1, :len(select)] = chosen / max(float(np.linalg.norm(chosen)), 1e-12)
    np.save(cache / 'term_ids.npy', term_ids)
    np.save(cache / 'term_weights.npy', weights)
    vocabulary = matrix.shape[1] + 1
    del matrix, term_ids, weights
    manifest = json.loads((data / 'split_manifest.json').read_text())
    size = int(manifest['history_rows']['train'])
    histories = np.lib.format.open_memmap(cache / 'histories.npy', mode='w+',
                                         dtype=np.int32, shape=(size, history_size))
    targets = np.lib.format.open_memmap(cache / 'targets.npy', mode='w+', dtype=np.int32, shape=(size,))
    users = np.lib.format.open_memmap(cache / 'users.npy', mode='w+', dtype=np.int64, shape=(size,))
    cutoff = int(__import__('datetime').datetime.fromisoformat(manifest['validation_start_utc']).timestamp() * 1000)
    n = 0
    with gzip.open(data / 'train_with_history.csv.gz', 'rt', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            if int(row['timestamp']) >= cutoff:
                raise ValueError('Validation/test timestamp in training file')
            history = [ids[x] for x in row['history'].split()][-history_size:]
            target = ids[row['parent_asin']]
            if not history or counts[target] == 0 or np.any(counts[history] == 0):
                raise ValueError('Training item/history invariant failed')
            if n >= size:
                raise ValueError('More train rows than manifest')
            histories[n] = 0
            histories[n, :len(history)] = history
            targets[n] = target
            users[n] = int.from_bytes(hashlib.blake2b(row['user_id'].encode(), digest_size=8).digest(), 'little', signed=True)
            n += 1
            if n % 500_000 == 0:
                print(f'Prepared histories {n:,}/{size:,}', flush=True)
    if n != size:
        raise ValueError(f'Train count mismatch: {n} != {size}')
    histories.flush(); targets.flush(); users.flush()
    wanted.update(rows=n, items=len(ids), vocabulary=vocabulary,
                  policy='Train-only parameters; histories supplied by temporal pipeline; top TF-IDF terms.')
    write_json(marker, wanted)
    return wanted


class TwoTower(nn.Module):
    def __init__(self, vocabulary, items, dim, history_size, id_dropout=0.5):
        super().__init__()
        self.terms = nn.EmbeddingBag(vocabulary, dim, mode='sum', padding_idx=0)
        self.item_ids = nn.Embedding(items + 1, dim, padding_idx=0)
        nn.init.normal_(self.terms.weight, std=0.03)
        nn.init.normal_(self.item_ids.weight, std=0.01)
        with torch.no_grad():
            self.terms.weight[0].zero_(); self.item_ids.weight[0].zero_()
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim), nn.LayerNorm(dim))
        self.position = nn.Embedding(history_size, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim*4,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.user_head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.id_dropout = id_dropout

    def encode_items(self, item_ids, terms, weights, train_counts):
        text = self.terms(terms[item_ids].long(), per_sample_weights=weights[item_ids])
        known = train_counts[item_ids].gt(0).float().unsqueeze(-1)
        if self.training:
            # Cold-item simulation; no inverted-dropout scaling at inference.
            known = known * torch.rand_like(known).ge(self.id_dropout)
        encoded = self.item_mlp(text + known * self.item_ids(item_ids))
        return F.normalize(encoded.float(), dim=-1) * item_ids.ne(0).unsqueeze(-1)

    def encode_user(self, vectors, histories):
        mask = histories.eq(0)
        safe_mask = mask.clone()
        safe_mask[:, 0] = False  # Empty histories get fallback scores during evaluation.
        positions = torch.arange(histories.shape[1], device=histories.device)
        x = vectors + self.position(positions)
        x = self.history_encoder(x, src_key_padding_mask=safe_mask)
        last = histories.ne(0).sum(-1).clamp_min(1) - 1
        x = x[torch.arange(len(histories), device=x.device), last]
        return F.normalize(self.user_head(x).float(), dim=-1)


def negative_mask(targets, histories, users):
    invalid = targets[:, None].eq(targets[None, :])
    invalid |= users[:, None].eq(users[None, :])
    invalid |= histories[:, :, None].eq(targets[None, None, :]).any(1)
    invalid.fill_diagonal_(False)
    return invalid


def selected_validation(samples, cap, seed):
    if not cap:
        return np.arange(len(samples['candidates']))
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(samples['regime_codes'] == code)) for code in range(4)]
    # Stratified subset, not a prefix (source arrays are sorted by regime).
    chosen = np.concatenate([group[:max(1, cap // 4)] for group in groups])
    return np.sort(chosen)


@torch.no_grad()
def evaluate(model, arrays, table, history_size, batch_size, cap, seed):
    terms, weights, counts = table
    model.eval()
    device = terms.device
    vectors = []
    for start in range(0, len(counts), 2048):
        ids = torch.arange(start, min(start+2048, len(counts)), device=device)
        vectors.append(model.encode_items(ids, terms, weights, counts))
    vectors = torch.cat(vectors)
    select = selected_validation(arrays, cap, seed)
    all_ranks = []
    all_lengths = []
    for start in range(0, len(select), batch_size):
        idx = select[start:start+batch_size]
        candidate_np = arrays['candidates'][idx]
        candidate = torch.as_tensor(candidate_np.astype(np.int64)+1, device=device)
        histories = np.zeros((len(idx), history_size), dtype=np.int64)
        lengths = []
        for j, source in enumerate(idx):
            a, b = arrays['history_indptr'][source:source+2]
            history = arrays['history_indices'][max(a, b-history_size):b] + 1
            histories[j, :len(history)] = history
            lengths.append(b-a)
        history_t = torch.as_tensor(histories, device=device)
        user = model.encode_user(vectors[history_t], history_t)
        scores = torch.einsum('bd,bkd->bk', user, vectors[candidate])
        empty = history_t.ne(0).sum(-1).eq(0)
        scores[empty] = counts[candidate[empty]].float().log1p()
        scores = scores.cpu().numpy().astype(np.float64)
        scores += tie_noise(candidate_np) * 1e-10
        ranks = 1 + (scores[:, 1:] > scores[:, :1]).sum(-1)
        all_ranks.extend(ranks.tolist()); all_lengths.extend(lengths)
    del vectors
    report = summarize(np.asarray(all_ranks), arrays['regime_codes'][select], np.asarray(all_lengths))
    return report


def self_test(device):
    torch.manual_seed(7)
    model = TwoTower(13, 8, 16, 3, id_dropout=0).to(device)
    terms = torch.randint(1, 13, (9, 4), device=device)
    weights = torch.ones((9, 4), device=device) / 2
    terms[0] = 0; weights[0] = 0
    counts = torch.tensor([0, 3, 2, 4, 2, 1, 0, 0, 0], device=device)
    model.eval()
    with torch.no_grad():
        before = model.encode_items(torch.tensor([6], device=device), terms, weights, counts)
        model.item_ids.weight[6].add_(100)
        after = model.encode_items(torch.tensor([6], device=device), terms, weights, counts)
        torch.testing.assert_close(before, after)  # Unseen item cannot use untrained ID.
        h = torch.tensor([[1, 2, 0], [0, 0, 0]], device=device)
        v = model.encode_items(h.flatten(), terms, weights, counts).reshape(2, 3, 16)
        assert torch.isfinite(model.encode_user(v, h)).all()
        # Padded item vectors must not affect a non-empty user's representation.
        changed = v.clone(); changed[0, 2] = 999
        torch.testing.assert_close(model.encode_user(v, h)[0], model.encode_user(changed, h)[0])
    t = torch.tensor([3, 3, 4], device=device)
    h = torch.tensor([[1, 4, 0], [1, 0, 0], [2, 0, 0]], device=device)
    u = torch.tensor([10, 11, 11], device=device)
    mask = negative_mask(t, h, u)
    assert mask[0, 1] and mask[0, 2] and mask[1, 2] and not mask.diag().any()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ids, inverse = torch.unique(torch.cat([h.flatten(), t]), return_inverse=True)
    emb = model.encode_items(ids, terms, weights, counts)[inverse]
    user = model.encode_user(emb[:h.numel()].reshape(3, 3, 16), h)
    logits = user @ emb[h.numel():].T
    loss = F.cross_entropy(logits.masked_fill(mask, -1e4), torch.arange(3, device=device))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    print('SELF TEST PASS: cold IDs, padding/empty history, false negatives, finite gradients', flush=True)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('This training command requires CUDA.')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    self_test('cuda')
    if args.mode == 'self-test':
        return
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    meta = prepare(args.data, args.cache, args.history, args.tokens)
    if args.mode == 'prepare':
        print(json.dumps(meta, indent=2), flush=True)
        return
    config = dict(vars(args)); config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config['cache_metadata'] = meta
    config['versions'] = {'torch': str(torch.__version__), 'numpy': np.__version__}
    model = TwoTower(meta['vocabulary'], meta['items'], args.dim, args.history).cuda()
    table = (
        torch.from_numpy(np.load(args.cache/'term_ids.npy')).cuda(),
        torch.from_numpy(np.load(args.cache/'term_weights.npy')).cuda(),
        torch.from_numpy(np.load(args.cache/'train_counts.npy')).cuda(),
    )
    if args.mode == 'test':
        state = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
        if state['fingerprint'] != meta['fingerprint']:
            raise ValueError('Checkpoint/data mismatch')
        model.load_state_dict(state['model'])
        report = evaluate(model, np.load(args.data/'evaluation/samples_test.npz'), table,
                          args.history, args.eval_batch, 0, args.seed)
        if (args.output/'test_metrics.json').exists():
            raise FileExistsError('Test already evaluated for this run.')
        write_json(args.output/'test_metrics.json', report)
        print(json.dumps(report, indent=2), flush=True)
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Use a new --output, or --resume for an interrupted run.')
    args.output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    first_epoch, first_batch, global_step, best, stale = 0, 0, 0, -1.0, 0
    if args.resume:
        state = torch.load(args.output/'latest.pt', map_location='cpu', weights_only=True)
        previous = json.loads((args.output/'config.json').read_text())
        for key in ('seed','dim','history','tokens','batch_size','lr','validation_cap','max_steps'):
            if previous[key] != config[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        if state['fingerprint'] != meta['fingerprint']:
            raise ValueError('Checkpoint/data mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        first_epoch, first_batch = state['epoch'], state['next_batch']
        global_step, best, stale = state['global_step'], state['best'], state['stale']
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        write_json(args.output/'config.json', config)
    histories = np.load(args.cache/'histories.npy')
    targets = np.load(args.cache/'targets.npy')
    users = np.load(args.cache/'users.npy')
    validation = np.load(args.data/'evaluation/samples_validation.npz')
    torch.cuda.reset_peak_memory_stats()
    def checkpoint(epoch, next_batch):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    epoch=epoch, next_batch=next_batch, global_step=global_step, best=best, stale=stale,
                    fingerprint=meta['fingerprint'], rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    total_batches = (len(targets) + args.batch_size - 1) // args.batch_size
    print(f'TRAIN START: {len(targets):,} history samples; {total_batches:,} batches/epoch; GPU {torch.cuda.get_device_name(0)}', flush=True)
    started = time.monotonic()
    for epoch in range(first_epoch, args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(targets))
        losses = []
        model.train()
        start_batch = first_batch if epoch == first_epoch else 0
        for batch in range(start_batch, total_batches):
            idx = order[batch*args.batch_size:(batch+1)*args.batch_size]
            if len(idx) < 2:
                continue
            h = torch.as_tensor(histories[idx].astype(np.int64), device='cuda')
            t = torch.as_tensor(targets[idx].astype(np.int64), device='cuda')
            u = torch.as_tensor(users[idx], device='cuda')
            unique, inverse = torch.unique(torch.cat([h.flatten(), t]), return_inverse=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                emb = model.encode_items(unique, *table)[inverse]
                hv = emb[:h.numel()].reshape(len(h), args.history, args.dim)
                user = model.encode_user(hv, h)
                scores = user.float() @ emb[h.numel():].float().T / 0.1
                scores = scores.masked_fill(negative_mask(t, h, u), -1e4)
                loss = F.cross_entropy(scores, torch.arange(len(h), device='cuda'))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach())); global_step += 1
            if global_step % 100 == 0 or global_step == 1:
                print(f'epoch={epoch+1} batch={batch+1}/{total_batches} step={global_step} loss={np.mean(losses[-100:]):.5f} elapsed_min={(time.monotonic()-started)/60:.1f} vram_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}', flush=True)
            if global_step % args.checkpoint_every == 0:
                save_checkpoint(args.output/'latest.pt', checkpoint(epoch, batch+1))
            if args.max_steps and global_step >= args.max_steps:
                break
        report = evaluate(model, validation, table, args.history, args.eval_batch,
                          args.validation_cap, args.seed)
        score = report['known_user']['ndcg@10']
        improved = score > best + 1e-5
        stale = 0 if improved else stale + 1
        if improved:
            best = score
            save_checkpoint(args.output/'best.pt', {'model':model.state_dict(), 'epoch':epoch+1,
                            'fingerprint':meta['fingerprint'], 'validation':report})
        write_json(args.output/f'validation_epoch_{epoch+1:03d}.json', report)
        record = {'epoch':epoch+1, 'step':global_step, 'train_loss':float(np.mean(losses)) if losses else None,
                  'validation':report, 'best_known_user_ndcg@10':best, 'early_stop_wait':stale,
                  'elapsed_seconds':time.monotonic()-started, 'partial_epoch':bool(args.max_steps and global_step>=args.max_steps)}
        with (args.output/'history.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        save_checkpoint(args.output/'latest.pt', checkpoint(epoch+1, 0))
        if stale >= args.patience or (args.max_steps and global_step >= args.max_steps):
            break
    write_json(args.output/'completed.json', {'best_known_user_ndcg@10':best, 'steps':global_step,
               'test_evaluated':False, 'smoke_run':bool(args.max_steps), 'elapsed_seconds':time.monotonic()-started})
    print(f'TRAIN COMPLETE: {args.output}; test untouched.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['self-test','prepare','train','test'], default='train')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--cache', type=Path, default=ROOT/'data/processed/gpu_cache_h20_t64_v1')
    p.add_argument('--output', type=Path, default=ROOT/'runs/content_transformer_v1')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--history', type=int, default=20)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--eval-batch', type=int, default=128)
    p.add_argument('--validation-cap', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0, help='Smoke run only; 0 uses the full train set.')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.dim % 4 or min(args.history,args.tokens,args.batch_size,args.epochs,args.checkpoint_every) < 1:
        p.error('Positive sizes required and dim must be divisible by 4.')
    train(args)


if __name__ == '__main__':
    main()
