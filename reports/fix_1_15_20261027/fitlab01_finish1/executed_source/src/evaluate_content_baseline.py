"""Evaluate popularity and TF-IDF user-profile baselines on shared samples."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from common import sha256
from contextlib import nullcontext
from evaluation_ledger import TestEvaluationLedger, import_known_evidence


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"
CONTENT = DATA / "content"
EVALUATION = DATA / "evaluation"
OUTPUT = DATA / "models/content_baseline"
SPLITS = ("validation", "test")
REGIMES = ("zero_shot", "extreme_cold", "cold", "warm")
PROFILE_ITEMS = 10
K = 10


def load_train_counts(data=None) -> np.ndarray:
    content = CONTENT if data is None else Path(data) / 'content'
    counts = []
    with gzip.open(content / "items.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if int(row["row_index"]) != len(counts):
                raise AssertionError("Item mapping is not contiguous")
            counts.append(int(row["train_count"]))
    return np.asarray(counts, dtype=np.float32)


def tie_noise(candidates: np.ndarray) -> np.ndarray:
    values = candidates.astype(np.uint64) * np.uint64(2_654_435_761)
    return (values & np.uint64(0xFFFFFFFF)).astype(np.float64) / 2**32


def rank_of_target(scores: np.ndarray, candidates: np.ndarray) -> int:
    adjusted = scores.astype(np.float64, copy=False) + tie_noise(candidates) * 1e-10
    return 1 + int(np.count_nonzero(adjusted[1:] > adjusted[0]))


def metrics(ranks: np.ndarray, mask: np.ndarray) -> dict:
    # Cast before arithmetic: NumPy may evaluate log2 on uint8 at reduced
    # precision, which makes serialized and in-memory rank metrics disagree.
    selected = ranks[mask].astype(np.int64, copy=False)
    if not len(selected):
        return {"samples": 0}
    hits = selected <= K
    return {
        "samples": int(len(selected)),
        f"recall@{K}": round(float(hits.mean()), 6),
        f"ndcg@{K}": round(
            float(np.where(hits, 1.0 / np.log2(selected + 1), 0.0).mean()), 6
        ),
        f"mrr@{K}": round(float(np.where(hits, 1.0 / selected, 0.0).mean()), 6),
        "mean_rank": round(float(selected.mean()), 3),
    }


def summarize(ranks, regime_codes, history_lengths, repeat_purchases=None):
    groups = {
        "overall": metrics(ranks, np.ones(len(ranks), dtype=bool)),
        "known_user": metrics(ranks, history_lengths > 0),
        "empty_history": metrics(ranks, history_lengths == 0),
        "by_regime": {},
    }
    for code, name in enumerate(REGIMES):
        groups["by_regime"][name] = metrics(ranks, regime_codes == code)
    if repeat_purchases is not None:
        groups["repeat_purchases"] = metrics(ranks, repeat_purchases == 1)
        groups["new_purchases"] = metrics(ranks, repeat_purchases == 0)
    return groups


def evaluate_split(split, matrix, train_counts, has_vector, evaluation=None, output=None):
    evaluation = EVALUATION if evaluation is None else Path(evaluation)
    output = OUTPUT if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / f'ranks_{split}.npz').exists():
        raise FileExistsError('Refusing to overwrite prior baseline ranks.')
    samples = np.load(evaluation / f"samples_{split}.npz")
    candidates = samples["candidates"]
    history_indices = samples["history_indices"]
    history_indptr = samples["history_indptr"]
    regime_codes = samples["regime_codes"]
    history_lengths = np.diff(history_indptr)
    popularity_ranks = np.empty(len(candidates), dtype=np.int32)
    content_ranks = np.empty(len(candidates), dtype=np.int32)

    for index, candidate_rows in enumerate(candidates):
        popularity = np.log1p(train_counts[candidate_rows])
        popularity_ranks[index] = rank_of_target(popularity, candidate_rows)

        start, end = history_indptr[index : index + 2]
        history = history_indices[max(start, end - PROFILE_ITEMS) : end]
        if len(history):
            history = history[has_vector[history]]
        if len(history):
            profile = np.asarray(matrix[history].sum(axis=0)).ravel()
            scores = np.asarray(matrix[candidate_rows].dot(profile)).ravel()
            scale = float(popularity.max()) or 1.0
            scores = scores + popularity / scale * 1e-7
        else:
            scores = popularity
        content_ranks[index] = rank_of_target(scores, candidate_rows)

        if (index + 1) % 5_000 == 0:
            print(f"  {split}: {index + 1:,}/{len(candidates):,}", flush=True)

    ranks_path = output / f"ranks_{split}.npz.tmp"
    with ranks_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            popularity_ranks=popularity_ranks,
            content_ranks=content_ranks,
            regime_codes=regime_codes,
            history_lengths=history_lengths,
        )
    ranks_path.replace(output / f"ranks_{split}.npz")
    result = {
        "popularity": summarize(popularity_ranks, regime_codes, history_lengths, samples.get("repeat_purchases")),
        "tfidf_profile": summarize(content_ranks, regime_codes, history_lengths, samples.get("repeat_purchases")),
    }
    for model in result.values():
        cold = [model['by_regime'][name] for name in REGIMES[:3]]
        model['cold_macro_ndcg@10'] = (round(float(np.mean([g['ndcg@10'] for g in cold])), 6)
                                        if all(g['samples'] for g in cold) else None)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, default=None,
                        help='Directory holding samples_{split}.npz.')
    parser.add_argument('--output', type=Path, default=OUTPUT,
                        help='Directory for ranks and metrics (must be empty or new).')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=['validation'])
    parser.add_argument('--evaluation-ledger', type=Path, default=ROOT/'reports/test_evaluation_ledger.json')
    parser.add_argument('--allow-seen-test', action='store_true')
    parser.add_argument('--test-purpose')
    args = parser.parse_args()
    if (args.data / 'demo_manifest.json').exists():
        raise ValueError('Synthetic demo data cannot be reported as benchmark results.')
    evaluation = args.evaluation or args.data / 'evaluation'
    content = args.data / 'content'
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading TF-IDF catalog matrix...", flush=True)
    matrix = sparse.load_npz(content / "tfidf_all.npz").tocsr()
    train_counts = load_train_counts(args.data)
    has_vector = np.diff(matrix.indptr) > 0
    if matrix.shape[0] != len(train_counts):
        raise AssertionError("TF-IDF and item mapping sizes differ")

    report = {
        "candidate_protocol": "One positive plus 99 fixed sampled negatives shared across models.",
        "content_model": f"Sum TF-IDF vectors of the latest {PROFILE_ITEMS} history items; cosine dot-product against L2-normalized item vectors; popularity fallback/tie-break.",
        "k": K,
        "data": str(args.data),
        "evaluation": str(evaluation),
        "test_evaluated": 'test' in args.splits,
        "sources": {name: sha256(content / name) for name in ('items.csv.gz', 'tfidf_all.npz')},
        "splits": {},
    }
    for split in args.splits:
        report['sources'][f'samples_{split}'] = sha256(evaluation / f'samples_{split}.npz')
        print(f"Evaluating {split}...", flush=True)
        ledger = TestEvaluationLedger(args.evaluation_ledger)
        if split == 'test':
            import_known_evidence(ROOT, ledger)
        identity = dict(model='popularity_and_tfidf', samples_test_sha256=report['sources'][f'samples_{split}'],
                        dataset_sha256=sha256(args.data/'split_manifest.json'),
                        scorer_sha256=sha256(Path(__file__)), model_sources=report['sources'], output=str(args.output))
        context = ledger.evaluation(identity, allow_seen=args.allow_seen_test, purpose=args.test_purpose) if split == 'test' else nullcontext(None)
        with context as entry:
            report['splits'][split] = evaluate_split(split, matrix, train_counts, has_vector, evaluation, args.output)
            if entry is not None:
                entry['test_metrics'] = report['splits'][split]
    temporary = args.output / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output / "metrics.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"PASS: content baseline evaluated at {args.output}", flush=True)


if __name__ == "__main__":
    main()
