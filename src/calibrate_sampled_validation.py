"""Exploratory temporal validation calibration; never opens test samples."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_validation import temporal_folds, fit_temperature, calibration_report
from common import sha256, write_json
from trained_ufm import model_classes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/toys_games_full_temporal'


@torch.no_grad()
def compute_scores(model, arrays, table, history_size, batch_size, amp):
    model.eval()
    scores_all = []
    device = table[0].device
    for start in range(0, len(arrays['candidates']), batch_size):
        idx = np.arange(start, min(start + batch_size, len(arrays['candidates'])))
        history = np.zeros((len(idx), history_size), dtype=np.int64)
        for j, row in enumerate(idx):
            a, b = arrays['history_indptr'][row:row + 2]
            past = arrays['history_indices'][max(a, b - history_size):b] + 1
            history[j, :len(past)] = past
        candidates = arrays['candidates'][idx].astype(np.int64) + 1
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            output = model(torch.as_tensor(history, device=device), torch.as_tensor(candidates, device=device), *table)
        scores = output['scores'].float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite validation scores.')
        scores_all.append(scores)
    if not scores_all:
        raise ValueError('Validation samples are empty.')
    return np.concatenate(scores_all)


def audit_scores(scores, arrays, evaluation):
    scores = np.asarray(scores)
    if scores.shape != arrays['candidates'].shape or not np.isfinite(scores).all():
        raise ValueError('Scores do not cover every validation candidate.')
    timestamps = []
    with gzip.open(evaluation/'samples_validation.csv.gz', 'rt', encoding='utf-8', newline='') as source:
        for index, row in enumerate(csv.DictReader(source)):
            if int(row['sample_index']) != index:
                raise ValueError('Validation metadata order mismatch.')
            timestamps.append(int(row['timestamp']))
    if len(timestamps) != len(scores):
        raise ValueError('Validation metadata length mismatch.')
    timestamps = np.asarray(timestamps, dtype=np.int64)
    fit, audit, boundary = temporal_folds(timestamps)
    temperature = fit_temperature(scores[fit])
    return dict(test_evaluated=False, temperature=temperature, split='validation',
        protocol='Earlier validation timestamps fit; later timestamps audit; timestamp ties remain together.',
        scope='conditional_on_sampled_candidates_not_purchase_probability',
        fit_samples=len(fit), audit_samples=len(audit), boundary_timestamp_ms=boundary,
        fit_max_timestamp_ms=int(timestamps[fit].max()), audit_min_timestamp_ms=int(timestamps[audit].min()),
        before=calibration_report(scores[audit], 1., arrays['regime_codes'][audit]),
        after=calibration_report(scores[audit], temperature, arrays['regime_codes'][audit]),
        limitations=['Exploratory: checkpoint selection already used validation; these are not independent final-test results.',
                     'Temperature scaling does not improve ranking metrics.',
                     'Temporal folds can share users.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--checkpoint', type=Path)
    origin.add_argument('--predictions', type=Path, help='Scores NPZ with sample_indices and exact candidates.')
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--features', type=Path, default=DATA/'foundation_clip_b32_v1')
    parser.add_argument('--counts', type=Path, default=ROOT/'data/processed/gpu_cache_h20_t64_v1/train_counts.npy')
    parser.add_argument('--training-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite calibration evidence.')
    if args.batch < 1:
        parser.error('Batch must be positive.')
    sample_path = args.evaluation/'samples_validation.npz'
    with np.load(sample_path) as source:
        arrays = {key:source[key] for key in source.files}
    provenance = dict(samples_validation_sha256=sha256(sample_path),
        metadata_sha256=sha256(args.evaluation/'samples_validation.csv.gz'),
        scorer_sha256=sha256(Path(__file__)))
    if args.predictions:
        with np.load(args.predictions) as source:
            if ('candidates' not in source or not np.array_equal(source['candidates'], arrays['candidates'])
                or not np.array_equal(source['sample_indices'], np.arange(len(arrays['candidates'])))):
                raise ValueError('Saved scores are not bound to these exact validation candidates.')
            scores = source['scores']
        provenance['predictions_sha256'] = sha256(args.predictions)
    else:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        Config, Model = model_classes(state['config'], args.training_source)
        if sha256(args.features/'complete.json') != state['config'].get('feature_marker_sha256'):
            raise ValueError('Feature marker does not match checkpoint.')
        expected_counts = state['config'].get('cache', {}).get('sources', {}).get('train_counts.npy')
        if expected_counts and sha256(args.counts) != expected_counts:
            raise ValueError('Train counts do not match checkpoint.')
        model = Model(Config(**state['model_config']))
        model.load_state_dict(state['model'])
        device = torch.device(args.device)
        model = model.to(device)
        table = tuple(torch.from_numpy(np.load(args.features/name)).to(device)
                      for name in ['text.npy', 'image.npy', 'modalities.npy'])
        table += (torch.from_numpy(np.load(args.counts)).to(device),)
        scores = compute_scores(model, arrays, table, model.config.history_size, args.batch, args.amp)
        provenance.update(checkpoint_sha256=sha256(args.checkpoint),
                          recorded_code_sha256=state['config']['code_sha256'])
    report = audit_scores(scores, arrays, args.evaluation)
    report['sources'] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({key:report[key] for key in ['temperature','fit_samples','audit_samples','test_evaluated']}))
    return report


if __name__ == '__main__':
    main()
