"""Audit saved UFM validation predictions: ranking, calibration and paired user CI.

No GPU or test labels. Calibration folds are exploratory because checkpoint
selection already used the whole validation set. Positive is candidate column 0.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_validation import temporal_folds, fit_temperature, calibration_report
from evaluate_content_baseline import REGIMES, tie_noise
from train_sasrec import DATA, ROOT, sha256, write_json


def ndcg(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    return np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.)


def paired_user_ci(delta, users, regimes, repeats=2000, seed=42):
    """Cluster bootstrap; one shared user weight across all three cold regimes."""
    delta, regimes = np.asarray(delta), np.asarray(regimes)
    _, inverse = np.unique(users, return_inverse=True)
    n = int(inverse.max()) + 1
    sums, counts = [], []
    for code in range(3):
        selected = regimes == code
        if not selected.any():
            raise ValueError('All cold regimes required for a macro interval.')
        sums.append(np.bincount(inverse[selected], weights=delta[selected], minlength=n))
        counts.append(np.bincount(inverse[selected], minlength=n))
    sums, counts = np.asarray(sums), np.asarray(counts)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(n, size=n), minlength=n)
        denominator = counts @ weights
        if np.all(denominator):
            values.append(float(np.mean((sums @ weights) / denominator)))
    if len(values) < repeats * .99:
        raise ValueError('Too many resamples lack a cold regime.')
    return dict(delta=float(np.mean(sums.sum(1) / counts.sum(1))),
                ci95=np.quantile(values, [.025, .975]).tolist(),
                users=n, resamples=len(values), seed=seed,
                method='paired percentile bootstrap clustered by user; cold macro NDCG@10')


def audit(data, run, output, repeats=2000):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty audit directory.')
    done = json.loads((run / 'completed.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if done.get('smoke_run') is not False or config.get('validation_cap') != 0:
        raise ValueError('Require completed full-validation production run.')
    with np.load(data / 'evaluation/samples_validation.npz') as source:
        samples = {k: source[k] for k in source.files}
    with np.load(run / 'best_validation_predictions.npz') as source:
        scores, indices = source['scores'], source['sample_indices']
    n = len(samples['candidates'])
    if (not np.array_equal(indices, np.arange(n)) or scores.shape != samples['candidates'].shape
            or not np.isfinite(scores).all()):
        raise ValueError('Saved predictions do not cover the fixed validation candidates.')
    adjusted = scores.astype(np.float64) + tie_noise(samples['candidates']) * 1e-10
    ranks = 1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(1)
    regimes = samples['regime_codes']
    reproduced = np.mean([ndcg(ranks[regimes == i]).mean() for i in range(3)])
    if abs(reproduced - done['best_cold_macro_ndcg@10']) > 2e-6:
        raise ValueError('Prediction scores do not reproduce the selected checkpoint metric.')
    with gzip.open(data / 'evaluation/samples_validation.csv.gz', 'rt', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != n or any(int(r['sample_index']) != i for i, r in enumerate(rows)):
        raise ValueError('Validation metadata order differs from predictions.')
    users = np.asarray([r['user_id'] for r in rows])
    timestamps = np.asarray([int(r['timestamp']) for r in rows], dtype=np.int64)
    lengths = np.diff(samples['history_indptr'])
    with np.load(data / 'models/content_baseline/ranks_validation.npz') as source:
        baseline = source['content_ranks']
    if baseline.shape != ranks.shape:
        raise ValueError('TF-IDF ranks differ in shape.')
    groups = {}
    for code, name in enumerate(REGIMES):
        groups[name] = {}
        for label, condition in [('all', np.ones(n, bool)), ('known_history', lengths > 0),
                                 ('empty_history', lengths == 0)]:
            selected = (regimes == code) & condition
            groups[name][label] = dict(samples=int(selected.sum()))
            if selected.any():
                groups[name][label].update(ufm_ndcg10=float(ndcg(ranks[selected]).mean()),
                    tfidf_ndcg10=float(ndcg(baseline[selected]).mean()),
                    delta=float((ndcg(ranks[selected])-ndcg(baseline[selected])).mean()))
    fit, holdout, boundary = temporal_folds(timestamps)
    temperature = fit_temperature(scores[fit])
    files = {'predictions': run / 'best_validation_predictions.npz', 'config': run / 'config.json',
             'completion': run / 'completed.json', 'samples': data / 'evaluation/samples_validation.npz',
             'metadata': data / 'evaluation/samples_validation.csv.gz',
             'baseline': data / 'models/content_baseline/ranks_validation.npz'}
    report = dict(run=str(run), sources={k: sha256(v) for k, v in files.items()},
        protocol=dict(split='validation', cases=n, candidates=scores.shape[1], positive_column=0),
        test_evaluated=False, prediction_metric_reproduced=True, cold_macro_ndcg10=float(reproduced),
        by_regime_and_history=groups,
        paired_comparison=paired_user_ci(ndcg(ranks)-ndcg(baseline), users, regimes, repeats),
        calibration=dict(temperature=temperature, fit_samples=len(fit), audit_samples=len(holdout),
            boundary_timestamp_ms=boundary,
            before=calibration_report(scores[holdout], 1., regimes[holdout]),
            after=calibration_report(scores[holdout], temperature, regimes[holdout])),
        limitations=['Validation was already used for checkpoint selection; calibration and CI are exploratory.',
                    'Confidence is conditional on sampled candidates, not purchase probability.',
                    'One training seed; user bootstrap does not measure training-seed variability.'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'report.json', report)
    lines = ['# UFM validation audit', '', f'Run: `{run.name}`. Validation only; test untouched.', '',
             '| Regime | History | Cases | UFM NDCG@10 | TF-IDF NDCG@10 | Difference |',
             '|---|---|---:|---:|---:|---:|']
    for regime, sections in groups.items():
        for history, row in sections.items():
            if row['samples']:
                lines.append(f"| {regime} | {history} | {row['samples']} | {row['ufm_ndcg10']:.6f} | {row['tfidf_ndcg10']:.6f} | {row['delta']:+.6f} |")
    ci = report['paired_comparison']
    lines += ['', f"Cold macro difference: {ci['delta']:+.6f}; exploratory user-bootstrap 95% CI {ci['ci95']}.",
              '', f'Temperature: {temperature:.6f}; fit={len(fit)}, audit={len(holdout)}.', '',
              '| Audit metric | Before | After |', '|---|---:|---:|']
    for key in ('ece', 'nll', 'brier_multiclass'):
        lines.append(f"| {key} | {report['calibration']['before']['overall'][key]:.6f} | {report['calibration']['after']['overall'][key]:.6f} |")
    lines += ['', *report['limitations']]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--run', type=Path, default=ROOT / 'runs/ufm_full_v1')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/ufm_validation_audit_v1')
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error('At least 100 bootstrap repetitions required.')
    torch.set_num_threads(4)
    result = audit(args.data, args.run, args.output, args.bootstrap)
    print(json.dumps(dict(cold_macro=result['cold_macro_ndcg10'],
                         comparison=result['paired_comparison'],
                         temperature=result['calibration']['temperature'])), flush=True)
