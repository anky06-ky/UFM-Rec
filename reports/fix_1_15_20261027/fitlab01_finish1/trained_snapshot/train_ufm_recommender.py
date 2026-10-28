"""Full-catalog UFM training; validation only, no test command.

Reuse verified baseline histories without changing them. Train negatives exclude
ALL that user's train positives (including first events), never validation/test.
Frozen CLIP tables are inputs; only adapter/sequential/UGAF heads are learned.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from scipy import sparse
import torch

from prepare_ufm_catalog import sha256
from train_gpu_recommender import fingerprint, write_json, save_checkpoint, selected_validation
from evaluate_content_baseline import summarize, tie_noise, REGIMES
from ufm_model import UFMConfig, UFMRec, ufm_loss, candidate_calibration

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/toys_games_full_temporal'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'


def user_key(value):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), 'little', signed=True)


def prepare(args):
    """CPU-only graph preparation, safe during another GPU job."""
    legacy = json.loads((args.legacy_cache / 'complete.json').read_text())
    source_fingerprint = fingerprint(args.data)
    manifest = json.loads((args.data / 'split_manifest.json').read_text())
    if legacy['fingerprint'] != source_fingerprint or legacy['history_size'] != args.history:
        raise ValueError('Legacy histories/source fingerprint or history size mismatch.')
    if legacy['rows'] != manifest['history_rows']['train']:
        raise ValueError('Legacy history row count mismatch.')
    sources = {'legacy_source': source_fingerprint,
               'raw_train': sha256(args.data / 'interactions_train.csv.gz')}
    for name in ['histories.npy', 'targets.npy', 'users.npy', 'train_counts.npy']:
        sources[name] = sha256(args.legacy_cache / name)
    wanted = {'version': 1, 'sources': sources, 'history_size': args.history,
              'rows': legacy['rows'], 'items': legacy['items']}
    marker = args.cache / 'complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if any(saved.get(k) != v for k, v in wanted.items()):
            raise ValueError('UFM graph cache does not match sources/config.')
        for name, expected in saved['graph_sha256'].items():
            if sha256(args.cache / name) != expected:
                raise ValueError('Graph cache checksum mismatch: ' + name)
        return saved
    if args.cache.exists() and any(args.cache.iterdir()):
        raise FileExistsError('Incomplete graph cache: use a NEW --cache, never overwrite.')
    args.cache.mkdir(parents=True, exist_ok=True)
    ids, counts = {}, [0]
    with gzip.open(args.data / 'content/items.csv.gz', 'rt', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            if int(row['row_index']) != len(ids) or row['parent_asin'] in ids:
                raise ValueError('Non-contiguous/duplicate item mapping.')
            ids[row['parent_asin']] = len(ids) + 1
            counts.append(int(row['train_count']))
    counts = np.asarray(counts, dtype=np.int64)
    if len(ids) != legacy['items'] or not np.array_equal(counts, np.load(args.legacy_cache / 'train_counts.npy')):
        raise ValueError('Legacy counts are not the mapped train-only counts.')
    h = np.load(args.legacy_cache / 'histories.npy', mmap_mode='r')
    t = np.load(args.legacy_cache / 'targets.npy', mmap_mode='r')
    users = np.load(args.legacy_cache / 'users.npy')
    if h.shape != (legacy['rows'], args.history) or t.shape != (legacy['rows'],) or users.shape != t.shape:
        raise ValueError('Legacy array shape mismatch.')
    keys = np.unique(users)
    lookup = {int(key): i for i, key in enumerate(keys)}
    seen_names = {}
    observed = np.zeros_like(counts)
    cutoff = int(datetime.fromisoformat(manifest['validation_start_utc']).timestamp() * 1000)
    total = int(manifest['interactions']['train'])
    print(f'CPU GRAPH START: {total:,} train interactions; {len(keys):,} history users.', flush=True)
    # ponytail: one streamed pass; COO temporary storage <=8 bytes/raw row.
    # For larger-than-local-disk graphs, replace this with a two-pass CSR builder.
    with tempfile.TemporaryDirectory(prefix='ufm_graph_') as temporary:
        edges_u = np.memmap(Path(temporary) / 'u.bin', mode='w+', dtype=np.int32, shape=(total,))
        edges_i = np.memmap(Path(temporary) / 'i.bin', mode='w+', dtype=np.int32, shape=(total,))
        n = scanned = 0
        with gzip.open(args.data / 'interactions_train.csv.gz', 'rt', encoding='utf-8') as source:
            for row in csv.DictReader(source):
                if int(row['timestamp']) >= cutoff or float(row['rating']) < 4:
                    raise ValueError('Future/nonpositive interaction in train source.')
                item = ids[row['parent_asin']]
                observed[item] += 1
                key = user_key(row['user_id'])
                if key in lookup:
                    name = seen_names.setdefault(key, row['user_id'])
                    if name != row['user_id']:
                        raise ValueError('User hash collision: cannot safely reuse legacy user keys.')
                    if n >= total:
                        raise ValueError('More graph edges than source manifest.')
                    edges_u[n], edges_i[n] = lookup[key], item
                    n += 1
                scanned += 1
                if scanned % 1_000_000 == 0:
                    print(f'graph_scanned={scanned:,}/{total:,} retained_edges={n:,}', flush=True)
        if scanned != total or not np.array_equal(observed, counts) or len(seen_names) != len(keys):
            raise ValueError('Raw train count/user coverage does not match mapping/cache.')
        graph = sparse.csr_matrix((np.ones(n, dtype=np.bool_), (edges_u[:n], edges_i[:n])), shape=(len(keys), len(counts)))
        graph.sum_duplicates(); graph.sort_indices()
        del edges_u, edges_i
    # Validate every legacy target/history against the full train graph.
    for start in range(0, len(t), 100_000):
        end = min(start + 100_000, len(t))
        rows = np.searchsorted(keys, users[start:end])
        targets = t[start:end]
        history = h[start:end]
        if np.any(targets < 1) or np.any(targets >= len(counts)) or np.any(history < 0) or np.any(history >= len(counts)):
            raise ValueError('Legacy item IDs out of range.')
        if np.any((history[:, 1:] != 0) & (history[:, :-1] == 0)) or np.any(history[:, 0] == 0):
            raise ValueError('Legacy history must be nonempty and right-padded.')
        if not np.asarray(graph[rows, targets]).all():
            raise ValueError('Legacy target absent from that user\'s train positives.')
        rr, cc = np.nonzero(history)
        if not np.asarray(graph[rows[rr], history[rr, cc]]).all():
            raise ValueError('Legacy history absent from that user\'s train positives.')
    np.save(args.cache / 'user_keys.npy', keys)
    np.save(args.cache / 'positive_indptr.npy', graph.indptr.astype(np.int64))
    np.save(args.cache / 'positive_items.npy', graph.indices.astype(np.int32))
    wanted.update(users=len(keys), edges=int(graph.nnz),
                  policy='All train positives excluded; no validation/test label filtering; legacy histories read-only.')
    wanted['graph_sha256'] = {name: sha256(args.cache / name) for name in ['user_keys.npy', 'positive_indptr.npy', 'positive_items.npy']}
    write_json(marker, wanted)
    print('CPU GRAPH PASS: ' + json.dumps(wanted), flush=True)
    return wanted


def sample_negatives(eligible, keys, indptr, positives, users, targets, histories, rng, number):
    candidates = np.repeat(targets[:, None], number, axis=1).astype(np.int64)
    valid = np.zeros(candidates.shape, dtype=np.bool_)
    rows = np.searchsorted(keys, users)
    if np.any(rows >= len(keys)) or np.any(keys[rows] != users):
        raise ValueError('Training user missing from positive graph.')
    for j, row in enumerate(rows):
        excluded = set(map(int, positives[indptr[row]:indptr[row + 1]]))
        excluded.update(map(int, histories[j])); excluded.add(int(targets[j]))
        selected = []
        # Bounded rejection; dense synthetic users fall back to exact set difference.
        for _ in range(32):
            for value in rng.choice(eligible, size=max(16, number * 2)):
                value = int(value)
                if value not in excluded:
                    selected.append(value); excluded.add(value)
                    if len(selected) == number:
                        break
            if len(selected) == number:
                break
        if len(selected) < number:
            remaining = np.setdiff1d(eligible, np.fromiter(excluded, dtype=np.int64))
            selected.extend(rng.choice(remaining, size=min(number - len(selected), len(remaining)), replace=False).tolist())
        candidates[j, :len(selected)] = selected
        valid[j, :len(selected)] = True
    return candidates, valid


def feature_audit(args, items):
    marker = json.loads((args.features / 'complete.json').read_text())
    catalog = json.loads((args.data / 'foundation_catalog_v1/catalog_report.json').read_text())
    if (marker['rows'] != items or marker['limit'] != 0 or not marker['encoder_frozen']
            or marker['revision'] != REVISION or marker['model'] != 'openai/clip-vit-base-patch32'
            or marker['feature_dim'] != 512 or marker['padding_row'] != 0):
        raise ValueError('Full, frozen, pinned CLIP cache is required; smoke/partial caches cannot train full UFM.')
    if marker['catalog_sha256'] != catalog['catalog_sha256'] or marker['text_sha256'] != sha256(args.data / 'content/products_text.csv.gz'):
        raise ValueError('Foundation cache/catalog/text fingerprint mismatch.')
    if catalog['sources_sha256']['items'] != sha256(args.data / 'content/items.csv.gz'):
        raise ValueError('Foundation mapping fingerprint mismatch.')
    tables = []
    for name in ['text.npy', 'image.npy', 'modalities.npy']:
        if sha256(args.features / name) != marker['features_sha256'][name]:
            raise ValueError('Foundation array checksum mismatch: ' + name)
        tables.append(np.load(args.features / name, mmap_mode='r'))
    text, image, flags = tables
    if text.shape != (items + 1, 512) or image.shape != text.shape or flags.shape != (items + 1, 2):
        raise ValueError('Foundation array shape mismatch.')
    if text.dtype != np.float16 or image.dtype != np.float16 or flags.dtype != np.bool_ or flags[0].any():
        raise ValueError('Foundation dtype/padding mismatch.')
    for column, table in enumerate([text, image]):
        if np.any(table[0]):
            raise ValueError('Nonzero feature padding.')
        for start in range(1, items + 1, 32768):
            values = np.asarray(table[start:start + 32768], dtype=np.float32)
            available = flags[start:start + 32768, column]
            if not np.isfinite(values).all() or np.any(values[~available]):
                raise ValueError('Nonfinite/missing-modality features.')
            if np.any(np.abs(np.linalg.norm(values[available], axis=-1) - 1) > 0.002):
                raise ValueError('Feature normalization mismatch.')
    coverage = flags[1:].sum(0)
    if list(map(int, coverage)) != [marker['has_text'], marker['has_image']]:
        raise ValueError('Coverage marker/mask mismatch.')
    if coverage[0] / items < 0.99 or coverage[1] / items < 0.9:
        raise ValueError('Coverage too low for initial full multimodal run; diagnose, do not silently train text-only.')
    return marker, {'items': items, 'text': int(coverage[0]), 'image': int(coverage[1]),
                    'finite_normalized_masked': True, 'test_evaluated': False}


@torch.no_grad()
def evaluate(model, arrays, table, args):
    model.eval()
    select = selected_validation(arrays, args.validation_cap, args.seed)
    if len(select) == 0:
        raise ValueError('Empty validation set.')
    scores_all, ranks, lengths, gates = [], [], [], []
    device = table[0].device
    for start in range(0, len(select), args.eval_batch):
        idx = select[start:start + args.eval_batch]
        candidate = arrays['candidates'][idx].astype(np.int64) + 1
        history = np.zeros((len(idx), args.history), dtype=np.int64)
        for j, source in enumerate(idx):
            a, b = arrays['history_indptr'][source:source + 2]
            past = arrays['history_indices'][max(a, b - args.history):b] + 1
            history[j, :len(past)] = past
            lengths.append(int(b - a))
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=args.amp and device.type == 'cuda'):
            output = model(torch.as_tensor(history, device=device), torch.as_tensor(candidate, device=device), *table)
        scores = output['scores'].float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite validation scores.')
        adjusted = scores.astype(np.float64) + tie_noise(candidate - 1) * 1e-10
        ranks.extend((1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(-1)).tolist())
        scores_all.append(scores)
        gates.append(output['weights'][:, 0].float().cpu().numpy())
    report = summarize(np.asarray(ranks), arrays['regime_codes'][select], np.asarray(lengths))
    cold = [report['by_regime'][name] for name in REGIMES[:3]]
    if not all(group['samples'] for group in cold):
        raise ValueError('Need all three cold regimes to select a checkpoint.')
    report['cold_macro_ndcg@10'] = float(np.mean([group['ndcg@10'] for group in cold]))
    scores = np.concatenate(scores_all)
    report['calibration_unscaled'] = candidate_calibration(torch.from_numpy(scores))
    weights = np.concatenate(gates)
    report['positive_mean_branch_weights'] = {name: weights[arrays['regime_codes'][select] == code].mean(0).tolist()
        for code, name in enumerate(REGIMES) if np.any(arrays['regime_codes'][select] == code)}
    report['protocol'] = 'Fixed validation candidates; 1 positive + 99 negatives on real dataset; sampled ranking, not full catalog.'
    return report, scores, select


def training_config(args, meta, feature_marker):
    ignored = {'mode', 'resume', 'interrupt_after_step', 'output'}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ignored}
    config['cache'] = meta
    config['feature_marker_sha256'] = sha256(args.features / 'complete.json')
    config['code_sha256'] = {p.name: sha256(p) for p in [Path(__file__), ROOT / 'src/ufm_model.py', ROOT / 'src/train_gpu_recommender.py']}
    config['versions'] = {'torch': str(torch.__version__), 'numpy': np.__version__}
    config['selection'] = 'Arithmetic mean NDCG@10 of zero_shot/extreme_cold/cold; validation only.'
    config['encoder'] = {'name': feature_marker['model'], 'revision': feature_marker['revision'], 'frozen': True}
    return config


def train(args):
    meta = prepare(args)
    if args.mode == 'prepare':
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    if args.device == 'cpu' and args.max_steps == 0:
        raise ValueError('CPU allowed for bounded smoke only, not full production training.')
    if (args.output / 'completed.json').exists():
        raise FileExistsError('Run already complete; use a new output, do not retrain/overwrite.')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Use a new output or --resume.')
    feature_marker, audit = feature_audit(args, meta['items'])
    config = training_config(args, meta, feature_marker)
    if args.resume and json.loads((args.output / 'config.json').read_text()) != config:
        raise ValueError('Resume config/code/source/version mismatch.')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(args.output / 'config.json', config)
    write_json(args.output / 'feature_audit.json', audit)
    torch.set_num_threads(args.threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)
    model_config = UFMConfig(items=meta['items'], dim=args.dim, history_size=args.history,
        heads=args.heads, layers=args.layers, dropout=args.dropout, id_dropout=args.id_dropout,
        variant=args.variant, fixed_alpha=args.fixed_alpha)
    model = UFMRec(model_config).to(device)
    # Load once into RAM/VRAM; no random reads of remote mmap during training.
    table = tuple(torch.from_numpy(np.load(args.features / name)).to(device) for name in ['text.npy', 'image.npy', 'modalities.npy'])
    counts = np.load(args.legacy_cache / 'train_counts.npy')
    table += (torch.from_numpy(counts).to(device),)
    histories = np.load(args.legacy_cache / 'histories.npy')
    targets = np.load(args.legacy_cache / 'targets.npy')
    users = np.load(args.legacy_cache / 'users.npy')
    keys, indptr, positives = (np.load(args.cache / name) for name in ['user_keys.npy', 'positive_indptr.npy', 'positive_items.npy'])
    eligible = np.flatnonzero(counts > 0)
    with np.load(args.data / 'evaluation/samples_validation.npz') as source:
        validation = {name: source[name] for name in source.files}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    rng = np.random.default_rng(args.seed + 113)
    epoch0 = batch0 = step = stale = 0
    best = -1.0
    stats = {'examples': 0, 'updates': 0, 'skipped': 0, 'sums': {name: 0.0 for name in ['total', 'bpr', 'align', 'unc', 'cal']}}
    if args.resume:
        state = torch.load(args.output / 'latest.pt', map_location='cpu', weights_only=True)
        if state['config'] != config:
            raise ValueError('Checkpoint fingerprint/config mismatch.')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epoch0, batch0, step, best, stale, stats = (state[k] for k in ['epoch', 'next_batch', 'global_step', 'best', 'stale', 'loss_stats'])
        random.setstate(state['python_rng']); rng.bit_generator.state = state['negative_rng']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'RESUME PASS: epoch={epoch0+1} next_batch={batch0} step={step} accumulated_examples={stats["examples"]}', flush=True)
    batches = (len(targets) + args.batch_size - 1) // args.batch_size
    if epoch0 < 0 or epoch0 > args.epochs or not 0 <= batch0 <= batches:
        raise ValueError('Invalid checkpoint cursor.')
    started = time.monotonic()

    def checkpoint(epoch, next_batch):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            model_config=asdict(model_config), config=config, epoch=epoch, next_batch=next_batch,
            global_step=step, best=best, stale=stale, loss_stats=stats,
            python_rng=random.getstate(), negative_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])

    print(f'UFM TRAIN START: samples={len(targets):,} items={meta["items"]:,} batches={batches:,} variant={args.variant} device={device}', flush=True)
    print(f'TRAINABLE PARAMS={sum(p.numel() for p in model.parameters() if p.requires_grad):,}; encoder frozen at {REVISION}', flush=True)
    for epoch in range(epoch0, args.epochs):
        if epoch != epoch0:
            stats = {'examples': 0, 'updates': 0, 'skipped': 0, 'sums': {name: 0.0 for name in stats['sums']}}
        order = np.random.default_rng(args.seed + epoch).permutation(len(targets))
        model.train()
        first_batch = batch0 if epoch == epoch0 else 0
        last_batch = first_batch - 1
        # A saved last-batch/max-step cursor must evaluate, not repeat a step.
        remaining = range(first_batch, batches) if not (args.max_steps and step >= args.max_steps) else ()
        for batch in remaining:
            last_batch = batch
            idx = order[batch * args.batch_size:(batch + 1) * args.batch_size]
            h, t = histories[idx].astype(np.int64), targets[idx].astype(np.int64)
            negative, valid = sample_negatives(eligible, keys, indptr, positives, users[idx], t, h, rng, args.negatives)
            if not valid.any():
                stats['skipped'] += len(idx)
                continue
            candidate = np.concatenate([t[:, None], negative], 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=args.amp and device.type == 'cuda'):
                output = model(torch.as_tensor(h, device=device), torch.as_tensor(candidate, device=device), *table)
                loss = ufm_loss(output, torch.as_tensor(valid, device=device), lambda_align=args.lambda_align,
                                lambda_unc=args.lambda_unc, lambda_cal=args.lambda_cal)
            if not all(torch.isfinite(v) for v in loss.values()):
                raise FloatingPointError('Nonfinite UFM loss.')
            scaler.scale(loss['total']).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            if not torch.isfinite(norm):
                # AMP overflows are expected occasionally: GradScaler backs off its
                # scale and the optimizer must not see this batch's gradients.
                scaler.update()
                stats['skipped'] += len(idx)
                print(f'AMP gradient overflow at epoch={epoch+1} batch={batch+1} step={step}; '
                      f'scale reduced to {scaler.get_scale()}', flush=True)
                continue
            scaler.step(optimizer); scaler.update()
            stats['examples'] += len(idx); stats['updates'] += 1
            for name, value in loss.items():
                stats['sums'][name] += float(value.detach()) * len(idx)
            step += 1
            if step == 1 or step % 100 == 0:
                vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0
                print(f'ufm_epoch={epoch+1} batch={batch+1}/{batches} step={step} loss={float(loss["total"].detach()):.5f} grad_norm={float(norm):.3f} elapsed_min={(time.monotonic()-started)/60:.1f} vram_GiB={vram:.2f}', flush=True)
            if step % args.checkpoint_every == 0 or step == args.interrupt_after_step:
                save_checkpoint(args.output / 'latest.pt', checkpoint(epoch, batch + 1))
            if step == args.interrupt_after_step:
                print('CONTROLLED INTERRUPT: checkpoint saved; no completion marker.', flush=True)
                return
            if args.max_steps and step >= args.max_steps:
                break
        report, scores, select = evaluate(model, validation, table, args)
        metric = report['cold_macro_ndcg@10']
        improved = metric > best + 1e-6
        stale = 0 if improved else stale + 1
        if improved:
            best = metric
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'model_config': asdict(model_config),
                'config': config, 'epoch': epoch + 1, 'validation': report})
            np.savez_compressed(args.output / 'best_validation_predictions.npz', scores=scores, sample_indices=select)
        partial = bool(args.max_steps and step >= args.max_steps and last_batch + 1 < batches)
        record = {'epoch': epoch + 1, 'step': step, 'train_loss': {name: value / max(1, stats['examples']) for name, value in stats['sums'].items()},
            'loss_stats': stats, 'partial_epoch': partial, 'validation': report, 'best_cold_macro_ndcg@10': best, 'early_stop_wait': stale}
        write_json(args.output / f'epoch_{epoch+1:03d}.json', record)
        # Rebuild atomically: interruption between report and checkpoint cannot duplicate an epoch.
        history = args.output / 'history.jsonl'
        temporary = history.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(json.loads(p.read_text())) + '\n' for p in sorted(args.output.glob('epoch_*.json'))), encoding='utf-8')
        temporary.replace(history)
        print('UFM VALIDATION: ' + json.dumps(record), flush=True)
        stats = {'examples': 0, 'updates': 0, 'skipped': 0, 'sums': {name: 0.0 for name in stats['sums']}}
        save_checkpoint(args.output / 'latest.pt', checkpoint(epoch + 1, 0))
        if stale >= args.patience or (args.max_steps and step >= args.max_steps):
            break
    write_json(args.output / 'completed.json', {'best_cold_macro_ndcg@10': best, 'steps': step,
        'smoke_run': bool(args.max_steps), 'test_evaluated': False, 'elapsed_seconds_this_process': time.monotonic()-started})
    print('UFM TRAIN COMPLETE: validation selected checkpoint; test untouched.', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'train'], default='train')
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--legacy-cache', type=Path, default=ROOT / 'data/processed/gpu_cache_h20_t64_v1')
    p.add_argument('--cache', type=Path, default=ROOT / 'data/processed/ufm_positive_graph_h20_v1')
    p.add_argument('--features', type=Path, default=DATA / 'foundation_clip_b32_v1')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/ufm_full_v1')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--variant', choices=['full','no_uncertainty','fixed_fusion','no_cross_align','semantic_only','collaborative_only','text_only','image_only'], default='full')
    for name, default in [('epochs',20), ('batch-size',256), ('dim',128), ('history',20), ('heads',4), ('layers',2), ('negatives',8), ('patience',3), ('checkpoint-every',1000), ('eval-batch',64), ('validation-cap',0), ('max-steps',0), ('interrupt-after-step',0), ('threads',4), ('seed',42)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('lr',0.0003), ('weight-decay',0.0001), ('dropout',0.1), ('id-dropout',0.5), ('fixed-alpha',0.5), ('lambda-align',0.01), ('lambda-unc',0.01), ('lambda-cal',0.1)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if min(args.epochs,args.batch_size,args.history,args.heads,args.layers,args.negatives,args.patience,args.checkpoint_every,args.eval_batch,args.threads) < 1 or args.dim % args.heads or min(args.max_steps,args.validation_cap,args.interrupt_after_step) < 0 or args.lr <= 0 or min(args.weight_decay,args.lambda_align,args.lambda_unc,args.lambda_cal) < 0:
        p.error('Invalid sizes/learning parameters.')
    return args


if __name__ == '__main__':
    train(parse_args())
