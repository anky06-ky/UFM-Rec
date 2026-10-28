"""Audit saved v3 predictions by positive-item train count; never retrains or opens test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REGIMES = ("zero_shot", "extreme_cold", "cold", "warm")
MODELS = {
    "full": "ufm_full_v1",
    "semantic_only": "ufm_ablation_semantic_only_s42_v1",
    "fixed_fusion": "ufm_ablation_fixed_fusion_s42_v1",
}
EXPECTED_CANDIDATES_SHA256 = "9387e66e9f1905c44d02b65c3809f5fa8b6c022f6abf0c1a37d52cb57767a60e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranks_for(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if scores.shape != candidates.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid or nonfinite saved scores")
    tie = (candidates.astype(np.uint64) * np.uint64(2_654_435_761) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 2**32
    adjusted = scores.astype(np.float64) + tie * 1e-10
    return 1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(axis=1)


def ndcg10(ranks: np.ndarray) -> np.ndarray:
    return np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.0)


def summarize(mask: np.ndarray, ndcg: dict[str, np.ndarray]) -> dict:
    n = int(mask.sum())
    result = {"cases": n}
    if n:
        result["mean_ndcg10"] = {name: float(values[mask].mean()) for name, values in ndcg.items()}
        result["paired_full_minus_semantic"] = float((ndcg["full"][mask] - ndcg["semantic_only"][mask]).mean())
        result["paired_full_minus_fixed"] = float((ndcg["full"][mask] - ndcg["fixed_fusion"][mask]).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--train-counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sample_sha = sha256(args.samples)
    if sample_sha != EXPECTED_CANDIDATES_SHA256:
        raise ValueError("Expected locked validation v3 candidate file")
    with np.load(args.samples) as source:
        candidates = source["candidates"]
        regimes = source["regime_codes"]
        history_lengths = np.diff(source["history_indptr"])
    counts = np.load(args.train_counts)
    if candidates.shape != (40000, 100) or len(counts) != 767046:
        raise ValueError("Unexpected candidate or count table size")
    if candidates.min() < 0 or candidates.max() >= len(counts) - 1:
        raise ValueError("Candidate index outside count table")
    positive_counts = counts[candidates[:, 0] + 1]
    ndcg = {}
    score_hashes = {}
    for label, run in MODELS.items():
        path = args.scores_dir / f"{run}_scores" / "scores_validation.npz"
        with np.load(path) as source:
            if not np.array_equal(source["sample_indices"], np.arange(len(candidates))):
                raise ValueError(f"Scores for {label} do not cover all cases in order")
            if not np.array_equal(source["candidates"], candidates):
                raise ValueError(f"Candidate mismatch for {label}")
            ndcg[label] = ndcg10(ranks_for(source["scores"], candidates))
        score_hashes[label] = sha256(path)
    cold = regimes < 3
    bins = {
        "0": positive_counts == 0,
        "1-5": (positive_counts >= 1) & (positive_counts <= 5),
        "6-20": (positive_counts >= 6) & (positive_counts <= 20),
        "21+": positive_counts >= 21,
    }
    report = {
        "protocol": "validation_v3_saved_predictions_only",
        "sources_sha256": {"samples": sample_sha, "train_counts": sha256(args.train_counts), **score_hashes},
        "training_seeds": [42],
        "cases": int(len(candidates)),
        "cold_macro_ndcg10": {name: float(np.mean([values[regimes == i].mean() for i in range(3)])) for name, values in ndcg.items()},
        "by_regime": {name: summarize(regimes == i, ndcg) for i, name in enumerate(REGIMES)},
        "by_positive_train_count": {name: summarize(mask, ndcg) for name, mask in bins.items()},
        "by_positive_train_count_and_history": {
            name: {"empty_history": summarize(mask & (history_lengths == 0), ndcg),
                   "has_history": summarize(mask & (history_lengths > 0), ndcg)}
            for name, mask in bins.items()
        },
        "notes": [
            "Positive-item train counts are train-only; no test labels were opened.",
            "These are paired descriptive differences on validation candidates, not new model runs.",
            "Validation population was used for historical checkpoint selection; not an independent holdout.",
            "One training seed; inference here does not isolate a causal architecture component.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cold_macro_ndcg10": report["cold_macro_ndcg10"],
                      "count_bin_cases": {k: v["cases"] for k, v in report["by_positive_train_count"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
