"""Evaluate popularity and TF-IDF user-profile baselines on shared samples."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"
CONTENT = DATA / "content"
EVALUATION = DATA / "evaluation"
OUTPUT = DATA / "models/content_baseline"
SPLITS = ("validation", "test")
REGIMES = ("zero_shot", "extreme_cold", "cold", "warm")
PROFILE_ITEMS = 10
K = 10


def load_train_counts() -> np.ndarray:
    counts = []
    with gzip.open(CONTENT / "items.csv.gz", "rt", encoding="utf-8", newline="") as source:
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
    selected = ranks[mask]
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


def summarize(ranks, regime_codes, history_lengths):
    groups = {
        "overall": metrics(ranks, np.ones(len(ranks), dtype=bool)),
        "known_user": metrics(ranks, history_lengths > 0),
        "empty_history": metrics(ranks, history_lengths == 0),
        "by_regime": {},
    }
    for code, name in enumerate(REGIMES):
        groups["by_regime"][name] = metrics(ranks, regime_codes == code)
    return groups


def evaluate_split(split, matrix, train_counts, has_vector):
    samples = np.load(EVALUATION / f"samples_{split}.npz")
    candidates = samples["candidates"]
    history_indices = samples["history_indices"]
    history_indptr = samples["history_indptr"]
    regime_codes = samples["regime_codes"]
    history_lengths = np.diff(history_indptr)
    popularity_ranks = np.empty(len(candidates), dtype=np.uint8)
    content_ranks = np.empty(len(candidates), dtype=np.uint8)

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

    ranks_path = OUTPUT / f"ranks_{split}.npz.tmp"
    with ranks_path.open("wb") as output:
        np.savez_compressed(
            output,
            popularity_ranks=popularity_ranks,
            content_ranks=content_ranks,
            regime_codes=regime_codes,
            history_lengths=history_lengths,
        )
    ranks_path.replace(OUTPUT / f"ranks_{split}.npz")
    return {
        "popularity": summarize(popularity_ranks, regime_codes, history_lengths),
        "tfidf_profile": summarize(content_ranks, regime_codes, history_lengths),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Loading TF-IDF catalog matrix...", flush=True)
    matrix = sparse.load_npz(CONTENT / "tfidf_all.npz").tocsr()
    train_counts = load_train_counts()
    has_vector = np.diff(matrix.indptr) > 0
    if matrix.shape[0] != len(train_counts):
        raise AssertionError("TF-IDF and item mapping sizes differ")

    report = {
        "candidate_protocol": "One positive plus 99 fixed sampled negatives shared across models.",
        "content_model": f"Sum TF-IDF vectors of the latest {PROFILE_ITEMS} history items; cosine dot-product against L2-normalized item vectors; popularity fallback/tie-break.",
        "k": K,
        "splits": {},
    }
    for split in SPLITS:
        print(f"Evaluating {split}...", flush=True)
        report["splits"][split] = evaluate_split(
            split, matrix, train_counts, has_vector
        )
    temporary = OUTPUT / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "metrics.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"PASS: content baseline evaluated at {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
