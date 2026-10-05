"""Tune and evaluate candidate-aware reciprocal-rank fusion."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy import sparse

from evaluate_content_baseline import (
    K,
    PROFILE_ITEMS,
    REGIMES,
    load_train_counts,
    summarize,
    tie_noise,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"
CONTENT = DATA / "content"
EVALUATION = DATA / "evaluation"
COLLABORATIVE = DATA / "models/collaborative_svd"
OUTPUT = DATA / "models/hybrid_rrf"
SPLITS = ("validation", "test")
COMPONENTS = ("popularity", "content", "collaborative")
RRF_CONSTANT = 60
WEIGHT_STEP = 0.1
COORDINATE_PASSES = 2




def rank_all(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return deterministic one-based ranks for every candidate."""
    adjusted = scores.astype(np.float64, copy=False) + tie_noise(candidates) * 1e-10
    order = np.argsort(-adjusted, kind="stable")
    ranks = np.empty(len(scores), dtype=np.uint8)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.uint8)
    return ranks


def build_component_ranks(
    split: str,
    matrix: sparse.csr_matrix,
    factors: np.ndarray,
    train_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recreate all candidate ranks for the three already-evaluated baselines."""
    samples = np.load(EVALUATION / f"samples_{split}.npz")
    candidates = samples["candidates"]
    history_indices = samples["history_indices"]
    history_indptr = samples["history_indptr"]
    regime_codes = samples["regime_codes"]
    history_lengths = np.diff(history_indptr)
    has_content = np.diff(matrix.indptr) > 0
    has_factor = np.linalg.norm(factors, axis=1) > 0

    popularity_ranks = np.empty(candidates.shape, dtype=np.uint8)
    content_ranks = np.empty(candidates.shape, dtype=np.uint8)
    collaborative_ranks = np.empty(candidates.shape, dtype=np.uint8)

    for index, candidate_rows in enumerate(candidates):
        popularity = np.log1p(train_counts[candidate_rows])
        popularity_ranks[index] = rank_all(popularity, candidate_rows)

        start, end = history_indptr[index : index + 2]
        history = history_indices[max(start, end - PROFILE_ITEMS) : end]

        content_history = history[has_content[history]]
        if len(content_history):
            profile = np.asarray(matrix[content_history].sum(axis=0)).ravel()
            content_scores = np.asarray(matrix[candidate_rows].dot(profile)).ravel()
            scale = float(popularity.max()) or 1.0
            content_scores += popularity / scale * 1e-7
        else:
            content_scores = popularity
        content_ranks[index] = rank_all(content_scores, candidate_rows)

        collaborative_history = history[has_factor[history]]
        if len(collaborative_history):
            profile = factors[collaborative_history].sum(axis=0)
            collaborative_scores = factors[candidate_rows] @ profile
            scale = float(popularity.max()) or 1.0
            collaborative_scores += popularity / scale * 1e-7
        else:
            collaborative_scores = popularity
        collaborative_ranks[index] = rank_all(collaborative_scores, candidate_rows)

        if (index + 1) % 5_000 == 0:
            print(f"  {split}: {index + 1:,}/{len(candidates):,}", flush=True)

    return {
        "candidates": candidates,
        "regime_codes": regime_codes,
        "history_lengths": history_lengths,
        "popularity": popularity_ranks,
        "content": content_ranks,
        "collaborative": collaborative_ranks,
    }


def verify_component_targets(split: str, data: dict[str, np.ndarray]) -> None:
    """Ensure recomputed full rankings reproduce the published target ranks."""
    content = np.load(DATA / f"models/content_baseline/ranks_{split}.npz")
    collaborative = np.load(COLLABORATIVE / f"ranks_{split}.npz")
    expected = {
        "popularity": content["popularity_ranks"],
        "content": content["content_ranks"],
        "collaborative": collaborative["collaborative_ranks"],
    }
    for name, target_ranks in expected.items():
        if not np.array_equal(data[name][:, 0], target_ranks):
            differences = int(np.count_nonzero(data[name][:, 0] != target_ranks))
            raise AssertionError(
                f"{split} {name} ranks differ from baseline for {differences} samples"
            )


def weight_grid() -> list[np.ndarray]:
    units = round(1 / WEIGHT_STEP)
    return [
        np.asarray((a, b, units - a - b), dtype=np.float64) / units
        for a, b in product(range(units + 1), repeat=2)
        if a + b <= units
    ]


def reciprocal_components(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [1.0 / (RRF_CONSTANT + data[name].astype(np.float64)) for name in COMPONENTS],
        axis=0,
    )


def target_ranks(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    adjusted = scores + tie_noise(candidates) * 1e-12
    return 1 + np.count_nonzero(adjusted[:, 1:] > adjusted[:, :1], axis=1)


def objective(ranks: np.ndarray) -> tuple[float, float, float, float]:
    hits = ranks <= K
    ndcg = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0).mean()
    recall = hits.mean()
    mrr = np.where(hits, 1.0 / ranks, 0.0).mean()
    return float(ndcg), float(recall), float(mrr), -float(ranks.mean())


def tune_global_weights(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tune a single global weight vector on validation overall NDCG@10."""
    components = reciprocal_components(data)
    candidates = data["candidates"]
    grid = weight_grid()

    best = None
    for weights in grid:
        scores = np.tensordot(weights, components, axes=(0, 0))
        ranks = target_ranks(scores, candidates)
        candidate = (objective(ranks), weights.copy(), ranks)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None

    tuning = {
        "global_weights": dict(zip(COMPONENTS, best[1].tolist())),
        "global_validation": summarize(
            best[2], data["regime_codes"], data["history_lengths"]
        ),
        "method": "Single global weight vector; no per-candidate regime gating.",
    }
    return best[1], best[2], tuning


def evaluate_weights(
    data: dict[str, np.ndarray], weights: np.ndarray
) -> tuple[np.ndarray, dict]:
    components = reciprocal_components(data)
    scores = np.tensordot(weights, components, axes=(0, 0))
    ranks = target_ranks(scores, data["candidates"])
    return ranks, summarize(ranks, data["regime_codes"], data["history_lengths"])


def save_ranks(split: str, ranks: np.ndarray, data: dict[str, np.ndarray]) -> None:
    path = OUTPUT / f"ranks_{split}.npz.tmp"
    with path.open("wb") as output:
        np.savez_compressed(
            output,
            hybrid_ranks=ranks.astype(np.uint8),
            regime_codes=data["regime_codes"],
            history_lengths=data["history_lengths"],
        )
    path.replace(OUTPUT / f"ranks_{split}.npz")


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    required = [
        CONTENT / "tfidf_all.npz",
        COLLABORATIVE / "item_factors.npy",
        *(EVALUATION / f"samples_{split}.npz" for split in SPLITS),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisite artifacts: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading content matrix and collaborative factors...", flush=True)
    matrix = sparse.load_npz(CONTENT / "tfidf_all.npz").tocsr()
    factors = np.load(COLLABORATIVE / "item_factors.npy", mmap_mode="r")
    train_counts = load_train_counts()
    if matrix.shape[0] != len(factors) or len(factors) != len(train_counts):
        raise AssertionError("Content, collaborative and item mapping sizes differ")

    print("Building validation component ranks...", flush=True)
    validation = build_component_ranks("validation", matrix, factors, train_counts)
    verify_component_targets("validation", validation)
    print("Tuning global weights on validation only...", flush=True)
    weights, validation_ranks, tuning = tune_global_weights(validation)
    validation_metrics = summarize(
        validation_ranks,
        validation["regime_codes"],
        validation["history_lengths"],
    )
    save_ranks("validation", validation_ranks, validation)

    print("Building test component ranks...", flush=True)
    test = build_component_ranks("test", matrix, factors, train_counts)
    verify_component_targets("test", test)
    test_ranks, test_metrics = evaluate_weights(test, weights)
    save_ranks("test", test_ranks, test)

    report = {
        "model": "Global reciprocal-rank fusion of popularity, TF-IDF content and collaborative SVD.",
        "leakage_policy": "Weights selected on validation only. A single weight vector scores all candidates uniformly. No per-candidate regime gating.",
        "components": list(COMPONENTS),
        "candidate_protocol": "Same fixed one-positive plus 99-negative samples as all baselines.",
        "parameters": {
            "rrf_constant": RRF_CONSTANT,
            "weight_step": WEIGHT_STEP,
            "coordinate_passes": COORDINATE_PASSES,
            "optimization_metric": f"ndcg@{K}",
            "tie_breakers": [f"recall@{K}", f"mrr@{K}", "mean_rank"],
        },
        "weights": dict(zip(COMPONENTS, weights.tolist())),
        "tuning": tuning,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }
    temporary = OUTPUT / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "report.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"PASS: hybrid fusion trained and evaluated at {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
