"""Direction 1: recent-popularity (trend) signal for the item side, no training.

Candidates of validation v3: the positive is drawn from validation interactions, negatives by
train+validation popularity inside the target's regime. Lifetime train popularity therefore
favours negatives; what predicts the positive is recent activity relative to lifetime count.
The trend score uses train interactions only (timestamps before validation start):
    trend(i) = log1p(recent_D(i)) - beta * log1p(train_count(i))
Everything is chosen on the earlier half of validation (fit fold, by time) and reported on the
later half (audit fold), exactly like the published hybrid. Test data are never read.

Inputs: UFM scores from best_validation_predictions.npz of a run (validation order),
TF-IDF scores recomputed as in src/evaluate_content_baseline.py.
"""
import argparse, csv, gzip, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from calibrate_validation import temporal_folds  # noqa: E402
from evaluate_content_baseline import REGIMES, load_train_counts, summarize, tie_noise  # noqa: E402

WINDOWS = [30, 90, 180, 365]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
GAMMAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
ALPHAS = [round(0.05 * s, 2) for s in range(21)]
BOOTSTRAP = 2000


_NOISE = {}


def ranks(scores, candidates):
    key = id(candidates)
    if key not in _NOISE:
        _NOISE[key] = tie_noise(candidates) * 1e-10
    adjusted = scores.astype(np.float64) + _NOISE[key]
    return 1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(1)


def ndcg(r):
    return np.where(r <= 10, 1 / np.log2(r + 1), 0.0)


def z(x):
    std = x.std(1, keepdims=True)
    return np.where(std > 0, (x - x.mean(1, keepdims=True)) / np.where(std > 0, std, 1), 0.0)


def cold_macro(r, regimes, mask):
    return float(np.mean([ndcg(r[mask & (regimes == c)]).mean() for c in range(3)]))


def report(r, samples, lengths, mask):
    out = summarize(r[mask], samples['regime_codes'][mask], lengths[mask], samples['repeat_purchases'][mask])
    out['cold_macro_ndcg@10'] = round(float(np.mean([out['by_regime'][n]['ndcg@10'] for n in REGIMES[:3]])), 6)
    return out


def ci(gain, regimes, mask, seed=42):
    rng = np.random.default_rng(seed)
    groups = [gain[mask & (regimes == c)] for c in range(3)]
    draws = [np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups]) for _ in range(BOOTSTRAP)]
    return [round(float(v), 6) for v in np.percentile(draws, [2.5, 97.5])]


def recent_counts(data, ids):
    manifest = json.loads((data / 'split_manifest.json').read_text())
    end = int(datetime.fromisoformat(manifest['validation_start_utc']).timestamp() * 1000)
    starts = {d: end - d * 86400000 for d in WINDOWS}
    out = {d: np.zeros(len(ids), dtype=np.int64) for d in WINDOWS}
    with gzip.open(data / 'interactions_train.csv.gz', 'rt', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = int(row['timestamp'])
            if t >= end:
                raise ValueError('Train interaction after validation start.')
            if t >= starts[365]:
                i = ids[row['parent_asin']]
                for d in WINDOWS:
                    if t >= starts[d]:
                        out[d][i] += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed/toys_games_full_temporal')
    p.add_argument('--run', type=Path, default=ROOT / 'runs/ufm_full_v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    t0 = time.monotonic()
    ev = args.data / 'evaluation'
    with np.load(ev / 'samples_validation.npz') as f:
        samples = {k: f[k] for k in f.files}
    with gzip.open(ev / 'samples_validation.csv.gz', 'rt', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    fit, audit, boundary = temporal_folds([int(r['timestamp']) for r in rows])
    n = len(samples['candidates'])
    fit_mask = np.zeros(n, bool); fit_mask[fit] = True
    audit_mask = np.zeros(n, bool); audit_mask[audit] = True
    cand = samples['candidates']
    regimes = samples['regime_codes']
    lengths = np.diff(samples['history_indptr'])
    empty = lengths == 0

    pred = np.load(args.run / 'best_validation_predictions.npz')
    if not np.array_equal(pred['sample_indices'], np.arange(n)):
        raise ValueError('Expected UFM predictions for every validation case in order.')
    ufm = pred['scores'].astype(np.float64)

    from hybrid_rebuild import tfidf_scores  # same TF-IDF scoring as the published hybrid
    matrix = sparse.load_npz(args.data / 'content/tfidf_all.npz').tocsr()
    counts = load_train_counts(args.data)
    tfidf = tfidf_scores(samples, matrix, counts)
    del matrix

    ids = {}
    with gzip.open(args.data / 'content/items.csv.gz', 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ids[r['parent_asin']] = int(r['row_index'])
    print('Counting recent train interactions...', flush=True)
    recent = recent_counts(args.data, ids)
    log_total = np.log1p(counts)

    def trend(d, beta):
        return (np.log1p(recent[d]) - beta * log_total)[cand]

    # 1) Item-only score for users without history (UFM and TF-IDF fall back to log popularity there).
    base_empty = ranks(np.log1p(counts)[cand], cand)
    grid_empty = {}
    for d in WINDOWS:
        for beta in BETAS:
            grid_empty[f'{d}d_beta{beta}'] = cold_macro(ranks(trend(d, beta), cand), regimes, fit_mask & empty)
    best_empty = max(grid_empty, key=grid_empty.get)
    d_e, b_e = int(best_empty.split('d_')[0]), float(best_empty.split('beta')[1])
    print('empty-history best', best_empty, grid_empty[best_empty], 'popularity', cold_macro(base_empty, regimes, fit_mask & empty), flush=True)

    # 2) Users with history: alpha*z(UFM) + (1-alpha)*z(TF-IDF) + gamma*z(trend).
    zu, zt = z(ufm), z(tfidf)
    known = ~empty
    grid_known = {}
    trends = {(d, beta): z(trend(d, beta)) for d in WINDOWS for beta in BETAS}
    for (d, beta), ztr in trends.items():
        for alpha in ALPHAS:
            mix = alpha * zu + (1 - alpha) * zt
            for gamma in GAMMAS:
                grid_known[f'{d}d_beta{beta}_alpha{alpha}_gamma{gamma}'] = cold_macro(
                    ranks(mix + gamma * ztr, cand), regimes, fit_mask & known)
    best_known = max(grid_known, key=grid_known.get)
    parts = best_known.split('_')
    d_k, b_k = int(parts[0][:-1]), float(parts[1][4:])
    a_k, g_k = float(parts[2][5:]), float(parts[3][5:])
    print('known-history best', best_known, grid_known[best_known], flush=True)

    # Published-style hybrid (alpha on all fit cases, no trend) as the reference.
    grid_h = {a: cold_macro(ranks(a * zu + (1 - a) * zt, cand), regimes, fit_mask) for a in ALPHAS}
    a_h = max(ALPHAS, key=grid_h.get)

    final = np.where(empty[:, None], z(trend(d_e, b_e)), a_k * zu + (1 - a_k) * zt + g_k * trends[(d_k, b_k)])
    r = {'tfidf': ranks(tfidf, cand), 'ufm': ranks(ufm, cand),
         'hybrid': ranks(a_h * zu + (1 - a_h) * zt, cand), 'hybrid_trend': ranks(final, cand)}
    audit_reports = {k: report(v, samples, lengths, audit_mask) for k, v in r.items()}
    subsets = {}
    for name, mask in [('empty_history', empty), ('known_history', known)]:
        subsets[name] = {k: cold_macro(v, regimes, audit_mask & mask) for k, v in r.items()}
    gain = ndcg(r['hybrid_trend']) - ndcg(r['hybrid'])
    gain_tfidf = ndcg(r['hybrid_trend']) - ndcg(r['tfidf'])
    result = dict(
        protocol='validation v3; parameters chosen on the earlier half by time (fit), reported on the later half (audit); '
                 'trend uses train interactions only; test never read.',
        ufm_run=str(args.run), folds=dict(boundary_timestamp_ms=boundary, fit=int(len(fit)), audit=int(len(audit))),
        selected=dict(empty_history=dict(window_days=d_e, beta=b_e),
                      known_history=dict(window_days=d_k, beta=b_k, alpha=a_k, gamma=g_k), hybrid_alpha=a_h),
        fit_best=dict(empty=grid_empty[best_empty], known=grid_known[best_known]),
        audit_cold_macro={k: v['cold_macro_ndcg@10'] for k, v in audit_reports.items()},
        audit_by_regime={k: {g: v['by_regime'][g]['ndcg@10'] for g in REGIMES} for k, v in audit_reports.items()},
        audit_cold_macro_by_history=subsets,
        gain_vs_hybrid=dict(value=round(audit_reports['hybrid_trend']['cold_macro_ndcg@10'] - audit_reports['hybrid']['cold_macro_ndcg@10'], 6),
                            ci95=ci(gain, regimes, audit_mask)),
        gain_vs_tfidf=dict(value=round(audit_reports['hybrid_trend']['cold_macro_ndcg@10'] - audit_reports['tfidf']['cold_macro_ndcg@10'], 6),
                           ci95=ci(gain_tfidf, regimes, audit_mask)),
        grid_empty_fit=grid_empty,
        audit=audit_reports,
        limitations=['Validation v3 was already used to select checkpoints; the audit half is not an independent holdout.',
                     'Part of the gain comes from how negatives are sampled (train+validation popularity); recent '
                     'popularity is still a legitimate train-only signal.'],
        runtime_seconds=round(time.monotonic() - t0, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ['selected', 'audit_cold_macro', 'audit_cold_macro_by_history', 'gain_vs_hybrid', 'gain_vs_tfidf']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
