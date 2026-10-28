"""One-shot final test evaluation of the locked production UFM run.

Run exactly once after the configuration is frozen: verifies the code and
feature fingerprints recorded in the run config, reproduces the recorded
validation metric from the saved predictions as a guard, fits the temperature
on the chronological early half of validation, then evaluates the held-out
test samples. Refuses to overwrite an existing final report.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_validation import fit_temperature, temporal_folds
from common import sha256, write_json
from evaluation_ledger import TestEvaluationLedger, import_known_evidence
from evaluate_content_baseline import REGIMES, summarize, tie_noise
from train_ufm_recommender import ROOT, DATA, REVISION
from ufm_model import UFMConfig, UFMRec, candidate_calibration


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
    return config['code_sha256']


def read_metadata(path: Path, n_expected: int, split: str):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if len(rows) != n_expected or any(int(r['sample_index']) != i for i, r in enumerate(rows)):
        raise ValueError(f'{split} metadata order differs from samples.')
    return np.asarray([r['user_id'] for r in rows]), \
        np.asarray([int(r['timestamp']) for r in rows], dtype=np.int64)


@torch.no_grad()
def evaluate_test(model, samples, table, device, amp, eval_batch, history_size):
    candidates = samples['candidates']
    history_indices = samples['history_indices']
    history_indptr = samples['history_indptr']
    scores_all, lengths = [], []
    for start in range(0, len(candidates), eval_batch):
        idx = np.arange(start, min(start + eval_batch, len(candidates)))
        candidate = torch.as_tensor(candidates[idx].astype(np.int64) + 1, device=device)
        history = np.zeros((len(idx), history_size), dtype=np.int64)
        for j, source in enumerate(idx):
            a, b = history_indptr[source:source + 2]
            past = history_indices[max(a, b - history_size):b] + 1
            history[j, :len(past)] = past
            lengths.append(int(b - a))
        with torch.amp.autocast(device.type, dtype=torch.float16,
                                enabled=amp and device.type == 'cuda'):
            output = model(torch.as_tensor(history, device=device), candidate, *table)
        scores = output['scores'].float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite test scores.')
        scores_all.append(scores)
    return np.concatenate(scores_all), np.asarray(lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/ufm_full_v1')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--features', type=Path, default=DATA / 'foundation_clip_b32_v1')
    parser.add_argument('--legacy-cache', type=Path, default=ROOT / 'data/processed/gpu_cache_h20_t64_v1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--evaluation-ledger', type=Path, default=None,
                        help='JSON ledger recording prior test evaluations; defaults to reports/test_evaluation_ledger.json.')
    parser.add_argument('--output', type=Path, default=None,
                        help='Final report path; defaults to <run>/final_test.json.')

    parser.add_argument('--allow-seen-test', action='store_true')
    parser.add_argument('--test-purpose', help='Reason for explicitly exploratory re-evaluation.')
    args = parser.parse_args()
    output = args.output or args.run / 'final_test.json'
    if output.exists():
        raise FileExistsError('A final test report already exists; the protocol allows one test evaluation.')
    ledger_path = args.evaluation_ledger or (ROOT / 'reports' / 'test_evaluation_ledger.json')
    ledger = TestEvaluationLedger(ledger_path)
    import_known_evidence(ROOT, ledger)
    identity = dict(checkpoint_sha256=sha256(args.run / 'best.pt'),
                    samples_test_sha256=sha256(args.data / 'evaluation/samples_test.npz'),
                    dataset_sha256=sha256(args.data / 'split_manifest.json'),
                    scorer_sha256=sha256(Path(__file__)), output=str(output))
    with ledger.evaluation(identity, allow_seen=args.allow_seen_test,
                           purpose=args.test_purpose) as evaluation:
        evaluate_locked_run(args, output, evaluation)


def evaluate_locked_run(args, output, evaluation):
    first_evaluation = evaluation['first_evaluation']
    torch.set_num_threads(args.threads)
    config = json.loads((args.run / 'config.json').read_text())
    done = json.loads((args.run / 'completed.json').read_text())
    if done.get('smoke_run') is not False or config.get('validation_cap') != 0:
        raise ValueError('Require the completed full-validation production run.')
    if config['variant'] != 'full':
        raise ValueError('Final test targets the production full variant.')
    if done.get('test_evaluated') is not False and not args.allow_seen_test:
        raise ValueError('Test was already evaluated for this run.')
    code_lock = verify_lock(args.run, config, args.features)
    with np.load(args.run / 'best_validation_predictions.npz') as source:
        val_scores, val_indices = source['scores'], source['sample_indices']
    with np.load(args.data / 'evaluation/samples_validation.npz') as source:
        validation = {k: source[k] for k in source.files}
    if not np.array_equal(val_indices, np.arange(len(validation['candidates']))) \
            or val_scores.shape != validation['candidates'].shape:
        raise ValueError('Saved validation predictions do not cover the fixed candidates.')
    val_ranks = 1 + ((val_scores.astype(np.float64) + tie_noise(validation['candidates']) * 1e-10)[:, 1:]
                     > (val_scores.astype(np.float64) + tie_noise(validation['candidates']) * 1e-10)[:, :1]).sum(1)
    reproduced = float(np.mean([
        np.where(val_ranks[validation['regime_codes'] == i] <= 10,
                 1 / np.log2(val_ranks[validation['regime_codes'] == i] + 1), 0.0).mean()
        for i in range(3)]))
    if abs(reproduced - done['best_cold_macro_ndcg@10']) > 2e-6:
        raise ValueError('Validation predictions do not reproduce the selected checkpoint metric.')
    val_users, val_timestamps = read_metadata(
        args.data / 'evaluation/samples_validation.csv.gz', len(validation['candidates']), 'validation')
    fit, holdout, boundary = temporal_folds(val_timestamps)
    temperature = fit_temperature(val_scores[fit])
    state = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=True)
    if state['config'] != config:
        raise ValueError('Checkpoint config fingerprint differs from run config.json.')
    model = UFMRec(UFMConfig(**state['model_config']))
    model.load_state_dict(state['model'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu.')
    model = model.to(device).eval()
    tables = [np.load(args.features / name) for name in ['text.npy', 'image.npy', 'modalities.npy']]
    counts = np.load(args.legacy_cache / 'train_counts.npy')
    table = tuple(torch.from_numpy(t).to(device) for t in [*tables, counts])
    with np.load(args.data / 'evaluation/samples_test.npz') as source:
        test = {k: source[k] for k in source.files}
    test_users, test_timestamps = read_metadata(
        args.data / 'evaluation/samples_test.csv.gz', len(test['candidates']), 'test')
    scores, lengths = evaluate_test(model, test, table, device, bool(config['amp']),
                                    args.eval_batch, config['history'])
    adjusted = scores.astype(np.float64) + tie_noise(test['candidates']) * 1e-10
    ranks = 1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(1)
    metrics = summarize(ranks, test['regime_codes'], lengths,
                        test.get('repeat_purchases'))
    cold = [metrics['by_regime'][name] for name in REGIMES[:3]]
    if not all(group['samples'] for group in cold):
        raise ValueError('Test split lacks a cold regime.')
    cold_macro = float(np.mean([group['ndcg@10'] for group in cold]))
    calibration = {
        'before': {**candidate_calibration(torch.from_numpy(scores), temperature=1.0)},
        'after': {**candidate_calibration(torch.from_numpy(scores), temperature=temperature)},
    }
    calibration['before']['by_regime'] = {}
    calibration['after']['by_regime'] = {}
    for code, name in enumerate(REGIMES):
        selected = test['regime_codes'] == code
        calibration['before']['by_regime'][name] = candidate_calibration(
            torch.from_numpy(scores[selected]), temperature=1.0)
        calibration['after']['by_regime'][name] = candidate_calibration(
            torch.from_numpy(scores[selected]), temperature=temperature)
    report = dict(run=str(args.run), test_evaluated=True,
                  sources=dict(checkpoint=sha256(args.run / 'best.pt'),
                               config=sha256(args.run / 'config.json'),
                               completion=sha256(args.run / 'completed.json'),
                               validation_predictions=sha256(args.run / 'best_validation_predictions.npz'),
                               samples_validation=sha256(args.data / 'evaluation/samples_validation.npz'),
                               metadata_validation=sha256(args.data / 'evaluation/samples_validation.csv.gz'),
                               samples_test=sha256(args.data / 'evaluation/samples_test.npz'),
                               metadata_test=sha256(args.data / 'evaluation/samples_test.csv.gz'),
                               items=sha256(args.data / 'content/items.csv.gz'),
                               text=sha256(args.features / 'text.npy'),
                               image=sha256(args.features / 'image.npy'),
                               modalities=sha256(args.features / 'modalities.npy')),
                  code_sha256=code_lock,
                  protocol=dict(split='test', cases=int(len(test['candidates'])),
                                candidates=int(scores.shape[1]), positive_column=0,
                                device=args.device, first_evaluation=first_evaluation),
                  validation_guard=dict(reproduced_cold_macro_ndcg10=reproduced,
                                        temperature=temperature,
                                        temperature_fit_samples=int(len(fit)),
                                        boundary_timestamp_ms=int(boundary)),
                  metrics=metrics,
                  cold_macro_ndcg10=cold_macro,
                  calibration=calibration,
                  limitations=['One training seed.',
                               'Test candidate pool is a popularity sample, not the full catalog.',
                               'Test set contains later purchases by the same user; see candidate validity report.'])
    write_json(output, report)
    evaluation.update(report_sha256=sha256(output), temperature=temperature, code_sha256=code_lock, test_metrics=metrics['overall'])
    print(json.dumps(dict(cold_macro=cold_macro, temperature=temperature,
                          overall_ndcg=metrics['overall']['ndcg@10'],
                          ece_before=calibration['before']['ece'],
                          ece_after=calibration['after']['ece'],
                          output=str(output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
