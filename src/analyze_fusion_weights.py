"""Analyse learned fusion weights of the production UFM checkpoint on validation.

Reproduces the selected-checkpoint metric from the saved predictions first (so
the analysis is anchored to the audited run), then recomputes a full forward
pass to collect per-candidate gate outputs (w_cf, w_sem, u_cf, u_sem, branch
scores) and aggregates them by regime and history length, comparing the gate's
preference with branch-score agreement and ranking outcomes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from common import sha256, write_json
from evaluate_content_baseline import REGIMES, tie_noise
from train_ufm_recommender import ROOT, DATA, REVISION
from ufm_model import UFMConfig, UFMRec

HISTORY_BUCKETS = ((0, 0), (1, 1), (2, 5), (6, 10), (11, 20), (21, 10 ** 9))


def load_checkpoint(run: Path, config: dict):
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if state['config'] != config:
        raise ValueError('Checkpoint config fingerprint differs from run config.json.')
    model = UFMRec(UFMConfig(**state['model_config']))
    model.load_state_dict(state['model'])
    return model.eval(), state


def verify_lock(run: Path, config: dict, features: Path):
    expected = config['code_sha256']
    for path in [ROOT / 'src/train_ufm_recommender.py', ROOT / 'src/ufm_model.py',
                 ROOT / 'src/train_gpu_recommender.py']:
        if expected[path.name] != sha256(path):
            raise ValueError(f'Code drifted since the run: {path.name}')
    if config['feature_marker_sha256'] != sha256(features / 'complete.json'):
        raise ValueError('Feature marker does not match the locked run.')
    if config.get('encoder', {}).get('revision') != REVISION:
        raise ValueError('Encoder revision mismatch.')


def forward_pass(model, samples, table, device, amp, eval_batch, history_size):
    scores, weights, uncertainty, cf_scores, sem_scores = [], [], [], [], []
    candidates = samples['candidates']
    history_indices = samples['history_indices']
    history_indptr = samples['history_indptr']
    lengths = np.diff(history_indptr).tolist()
    with torch.no_grad():
        for start in range(0, len(candidates), eval_batch):
            idx = np.arange(start, min(start + eval_batch, len(candidates)))
            candidate = torch.as_tensor(candidates[idx].astype(np.int64) + 1, device=device)
            history = np.zeros((len(idx), history_size), dtype=np.int64)
            for j, source in enumerate(idx):
                a, b = history_indptr[source:source + 2]
                past = history_indices[max(a, b - history_size):b] + 1
                history[j, :len(past)] = past
            with torch.amp.autocast(device.type, dtype=torch.float16,
                                    enabled=amp and device.type == 'cuda'):
                output = model(torch.as_tensor(history, device=device), candidate, *table)
            scores.append(output['scores'].float().cpu().numpy())
            weights.append(output['weights'].float().cpu().numpy())
            uncertainty.append(output['uncertainty'].float().cpu().numpy())
            cf_scores.append(output['cf_scores'].float().cpu().numpy())
            sem_scores.append(output['semantic_scores'].float().cpu().numpy())
    return (np.concatenate(scores), np.concatenate(weights), np.concatenate(uncertainty),
            np.concatenate(cf_scores), np.concatenate(sem_scores), np.asarray(lengths))


def ndcg(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    return np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.0)


def group_stats(mask, ranks, w_cf_pos, w_sem_pos, w_cf_neg_mean, u_pos, cf_pos, sem_pos):
    selected = np.flatnonzero(mask)
    if not len(selected):
        return {'samples': 0}
    correct = ranks[selected] <= 10
    prefers_cf = w_cf_pos[selected] > w_sem_pos[selected]
    branch_higher_cf = cf_pos[selected] > sem_pos[selected]
    agree = prefers_cf == branch_higher_cf
    return {
        'samples': int(len(selected)),
        'ndcg@10': round(float(ndcg(ranks[selected]).mean()), 6),
        'hit@10': round(float(correct.mean()), 6),
        'mean_positive_w_cf': round(float(w_cf_pos[selected].mean()), 6),
        'mean_positive_w_sem': round(float(w_sem_pos[selected].mean()), 6),
        'mean_negative_w_cf': round(float(w_cf_neg_mean[selected].mean()), 6),
        'mean_positive_u_cf': round(float(u_pos[selected, 0].mean()), 6),
        'mean_positive_u_sem': round(float(u_pos[selected, 1].mean()), 6),
        'gate_prefers_cf': round(float(prefers_cf.mean()), 6),
        'gate_agrees_with_branch_scores': round(float(agree.mean()), 6),
        'hit_when_prefers_cf': round(float(correct[prefers_cf].mean()), 6) if prefers_cf.any() else None,
        'hit_when_prefers_sem': round(float(correct[~prefers_cf].mean()), 6) if (~prefers_cf).any() else None,
        'hit_when_gate_agrees': round(float(correct[agree].mean()), 6) if agree.any() else None,
        'hit_when_gate_disagrees': round(float(correct[~agree].mean()), 6) if (~agree).any() else None,
    }


def id_embedding_quality(model, counts, text_table, modalities_table, n_per_group=2000,
                         reference=50000, seed=42):
    """Compare learned ID vectors across train-count regimes.

    Zero-shot rows never receive gradient updates, so they serve as the random
    init baseline (init std 0.02, expected norm ~0.226 for dim 128). Alignment is
    the mean top-5 Jaccard overlap between an item's ID-space and text-space
    nearest neighbours over a shared reference pool.
    """
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        ids = model.item_ids.weight.detach().float().cpu().numpy()[1:]
    counts = np.asarray(counts)[1:]
    text = np.asarray(text_table[1:], dtype=np.float32)
    has_text = np.asarray(modalities_table[1:, 0], dtype=bool)
    if len(counts) != ids.shape[0] or text.shape[0] != ids.shape[0] or len(has_text) != ids.shape[0]:
        raise ValueError('Count/feature table sizes do not match the item table.')
    pool = rng.choice(ids.shape[0], size=min(reference, ids.shape[0]), replace=False)
    pool_ids = ids[pool]
    pool_text = text[pool]
    pool_text_has = has_text[pool]
    item_regime = np.zeros(ids.shape[0], dtype=np.uint8)
    item_regime[(counts >= 1) & (counts <= 5)] = 1
    item_regime[(counts >= 6) & (counts <= 20)] = 2
    item_regime[counts > 20] = 3
    groups = {}
    for code, name in enumerate(REGIMES):
        eligible = np.flatnonzero((item_regime == code) & has_text)
        chosen = rng.choice(eligible, size=min(n_per_group, len(eligible)), replace=False)
        norms = np.linalg.norm(ids[chosen], axis=1)
        overlaps = np.zeros(len(chosen), dtype=np.float64)
        pool_ids_n = pool_ids / np.linalg.norm(pool_ids, axis=1, keepdims=True)
        pool_text_n = pool_text / np.linalg.norm(pool_text, axis=1, keepdims=True)
        pool_text_n[~pool_text_has] = 0
        for start in range(0, len(chosen), 250):
            block = chosen[start:start + 250]
            id_n = ids[block] / np.linalg.norm(ids[block], axis=1, keepdims=True)
            text_n = text[block] / np.linalg.norm(text[block], axis=1, keepdims=True)
            id_top = np.argsort(-(id_n @ pool_ids_n.T), axis=1)[:, :5]
            text_top = np.argsort(-(text_n @ pool_text_n.T), axis=1)[:, :5]
            overlaps[start:start + len(block)] = [
                len(set(map(int, a)) & set(map(int, b))) / 5.0 for a, b in zip(id_top, text_top)]
        groups[name] = {
            'samples': int(len(chosen)),
            'mean_norm': round(float(norms.mean()), 6),
            'std_norm': round(float(norms.std()), 6),
            'mean_top5_jaccard_id_vs_text': round(float(overlaps.mean()), 6),
        }
    groups['init_expected_norm'] = round(float(np.sqrt(ids.shape[1]) * 0.02), 6)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/ufm_full_v1')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--features', type=Path, default=DATA / 'foundation_clip_b32_v1')
    parser.add_argument('--legacy-cache', type=Path, default=ROOT / 'data/processed/gpu_cache_h20_t64_v1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/fusion_weight_analysis_v1')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty output directory.')
    torch.set_num_threads(args.threads)
    config = json.loads((args.run / 'config.json').read_text())
    done = json.loads((args.run / 'completed.json').read_text())
    if done.get('smoke_run') is not False or config.get('validation_cap') != 0:
        raise ValueError('Require the completed full-validation production run.')
    if config['variant'] != 'full':
        raise ValueError('This analysis targets the full variant.')
    verify_lock(args.run, config, args.features)
    with np.load(args.data / 'evaluation/samples_validation.npz') as source:
        samples = {k: source[k] for k in source.files}
    with gzip.open(args.data / 'evaluation/samples_validation.csv.gz', 'rt', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(samples['candidates']) or any(int(r['sample_index']) != i for i, r in enumerate(rows)):
        raise ValueError('Validation metadata order differs from samples.')
    users = np.asarray([r['user_id'] for r in rows])
    timestamps = np.asarray([int(r['timestamp']) for r in rows], dtype=np.int64)
    with np.load(args.run / 'best_validation_predictions.npz') as source:
        saved_scores, saved_indices = source['scores'], source['sample_indices']
    if not np.array_equal(saved_indices, np.arange(len(samples['candidates']))) \
            or saved_scores.shape != samples['candidates'].shape:
        raise ValueError('Saved predictions do not cover the fixed validation candidates.')
    adjusted = saved_scores.astype(np.float64) + tie_noise(samples['candidates']) * 1e-10
    saved_ranks = 1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(1)
    reproduced = float(np.mean([ndcg(saved_ranks[samples['regime_codes'] == i]).mean() for i in range(3)]))
    if abs(reproduced - done['best_cold_macro_ndcg@10']) > 2e-6:
        raise ValueError('Saved predictions do not reproduce the selected checkpoint metric.')
    model, state = load_checkpoint(args.run, config)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu.')
    model = model.to(device)
    tables = [np.load(args.features / name) for name in ['text.npy', 'image.npy', 'modalities.npy']]
    counts = np.load(args.legacy_cache / 'train_counts.npy')
    table = tuple(torch.from_numpy(t).to(device) for t in [*tables, counts])
    id_quality = id_embedding_quality(model, counts, tables[0], tables[2])
    scores, weights, uncertainty, cf_scores, sem_scores, lengths = forward_pass(
        model, samples, table, device, bool(config['amp']), args.eval_batch, config['history'])
    max_diff = float(np.abs(scores - saved_scores).max())
    if max_diff > 5e-2:
        raise ValueError(f'Forward pass diverges from saved scores (max abs diff {max_diff}).')
    ranks = 1 + ((scores.astype(np.float64) + tie_noise(samples['candidates']) * 1e-10)[:, 1:]
                 > (scores.astype(np.float64) + tie_noise(samples['candidates']) * 1e-10)[:, :1]).sum(1)
    w_cf_pos = weights[:, 0, 0]
    w_sem_pos = weights[:, 0, 1]
    w_cf_neg_mean = weights[:, 1:, 0].mean(1)
    u_pos = uncertainty[:, 0]
    cf_pos = cf_scores[:, 0]
    sem_pos = sem_scores[:, 0]
    regimes = samples['regime_codes']
    groups = {'overall': group_stats(np.ones(len(ranks), bool), ranks, w_cf_pos, w_sem_pos,
                                     w_cf_neg_mean, u_pos, cf_pos, sem_pos)}
    for code, name in enumerate(REGIMES):
        groups[name] = group_stats(regimes == code, ranks, w_cf_pos, w_sem_pos,
                                   w_cf_neg_mean, u_pos, cf_pos, sem_pos)
    history_groups = {}
    for low, high in HISTORY_BUCKETS:
        mask = (lengths >= low) & (lengths <= high)
        label = '0' if high == 0 else (f'{low}' if low == high else f'{low}-{min(high, 20)}')
        history_groups[label] = group_stats(mask, ranks, w_cf_pos, w_sem_pos,
                                            w_cf_neg_mean, u_pos, cf_pos, sem_pos)
    both_available = weights[:, 0].sum(1)
    hard_cf = int(((w_cf_pos == 0) & (w_sem_pos > 0)).sum())
    hard_sem = int(((w_sem_pos == 0) & (w_cf_pos > 0)).sum())
    correlation = float(np.corrcoef(w_cf_pos, ranks.astype(np.float64))[0, 1]) if len(ranks) > 1 else None
    report = dict(run=str(args.run), sources=dict(
        checkpoint=sha256(args.run / 'best.pt'), config=sha256(args.run / 'config.json'),
        completion=sha256(args.run / 'completed.json'),
        predictions=sha256(args.run / 'best_validation_predictions.npz'),
        samples=sha256(args.data / 'evaluation/samples_validation.npz'),
        metadata=sha256(args.data / 'evaluation/samples_validation.csv.gz'),
        text=sha256(args.features / 'text.npy'), image=sha256(args.features / 'image.npy'),
        modalities=sha256(args.features / 'modalities.npy'),
        train_counts=sha256(args.legacy_cache / 'train_counts.npy')),
        protocol=dict(split='validation', cases=len(ranks), candidates=scores.shape[1],
                      positive_column=0, device=args.device),
        reproduced_cold_macro_ndcg10=reproduced,
        forward_vs_saved_max_abs_diff=max_diff,
        gate_weights_sum_to_one=bool(np.allclose(both_available[np.isfinite(both_available)], 1.0)),
        hard_gates={'w_cf_exactly_zero': hard_cf, 'w_sem_exactly_zero': hard_sem},
        correlation_w_cf_rank=correlation,
        by_regime=groups, by_history_length=history_groups,
        id_embedding_quality=id_quality,
        limitations=['Validation was already used for checkpoint selection.',
                     'Weights are conditional on sampled candidates, not catalog-wide.',
                     'One training seed.'])
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'report.json', report)
    lines = ['# Fusion weight analysis (validation)', '', f'Run: `{args.run.name}`.',
             '', '| Group | Cases | NDCG@10 | mean w_cf (pos) | mean w_cf (neg) | u_cf | u_sem | prefers CF | gate-branch agree |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, group in [('overall', groups['overall'])] + [(name, groups[name]) for name in REGIMES]:
        if group['samples']:
            lines.append(f"| {label} | {group['samples']} | {group['ndcg@10']} | {group['mean_positive_w_cf']} | {group['mean_negative_w_cf']} | {group['mean_positive_u_cf']} | {group['mean_positive_u_sem']} | {group['gate_prefers_cf']} | {group['gate_agrees_with_branch_scores']} |")
    lines += ['', 'By history length:', '']
    for label, group in history_groups.items():
        if group['samples']:
            lines.append(f"| {label} | {group['samples']} | {group['ndcg@10']} | {group['mean_positive_w_cf']} | {group['mean_negative_w_cf']} | {group['mean_positive_u_cf']} | {group['mean_positive_u_sem']} | {group['gate_prefers_cf']} | {group['gate_agrees_with_branch_scores']} |")
    lines += ['', f"Max |forward - saved| on scores: {max_diff:.2e}.",
              f"Hard gates: w_cf == 0 on {hard_cf} positives; w_sem == 0 on {hard_sem} positives.",
              '', 'ID embedding quality by train-count regime:',
              '', '| Regime | Items sampled | Mean norm | Std norm | Mean top-5 Jaccard (ID vs text NN) |',
              '|---|---:|---:|---:|---:|']
    for name in REGIMES:
        group = id_quality[name]
        lines.append(f"| {name} | {group['samples']} | {group['mean_norm']} | {group['std_norm']} | {group['mean_top5_jaccard_id_vs_text']} |")
    lines.append(f"| init expectation | - | {id_quality['init_expected_norm']} | - | - |")
    lines += ['', *report['limitations']]
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(cold_macro=reproduced, max_diff=max_diff,
                          by_regime=groups, correlation=correlation), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
