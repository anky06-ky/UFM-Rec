"""Blend UFM full with the TF-IDF profile baseline on validation v3; nothing is trained.

Both models score the fixed v3 candidates as in the published runs: TF-IDF as in
src/evaluate_content_baseline.py, UFM as in the executed reevaluate_ufm_checkpoints.py
(last 20 history items, IDs shifted by one) with the archived training source. The
recomputed metrics must match the published ones before any blend is reported.
Each model's 100 scores per case are standardized and mixed as
alpha * z(UFM) + (1 - alpha) * z(TF-IDF). Alpha is chosen by cold macro NDCG@10 on
the earlier validation timestamps (the calibration folds) and reported on the later
ones, so the reported fold never influences the choice.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrate_validation import temporal_folds  # noqa: E402
from common import sha256, write_json  # noqa: E402
from evaluate_content_baseline import PROFILE_ITEMS, REGIMES, load_train_counts, summarize, tie_noise  # noqa: E402
from trained_ufm import model_classes  # noqa: E402

ALPHAS = [round(0.05 * step, 2) for step in range(21)]
BOOTSTRAP = 2000
TOLERANCE = 5e-4  # largest allowed gap to a published NDCG@10 (CPU here, GPU on FITLAB)


def ranks_of_positive(scores, candidates):
    adjusted = scores.astype(np.float64) + tie_noise(candidates) * 1e-10
    return 1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(1)


def ndcg10(ranks):
    return np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)


def tfidf_scores(samples, matrix, train_counts):
    """Scores of src/evaluate_content_baseline.py, kept for every candidate."""
    has_vector = np.diff(matrix.indptr) > 0
    indptr, indices = samples["history_indptr"], samples["history_indices"]
    out = np.empty(samples["candidates"].shape, dtype=np.float64)
    for case, rows in enumerate(samples["candidates"]):
        popularity = np.log1p(train_counts[rows])
        start, end = indptr[case:case + 2]
        history = indices[max(start, end - PROFILE_ITEMS):end]
        if len(history):
            history = history[has_vector[history]]
        if len(history):
            profile = np.asarray(matrix[history].sum(axis=0)).ravel()
            scores = np.asarray(matrix[rows].dot(profile)).ravel()
            scores = scores + popularity / (float(popularity.max()) or 1.0) * 1e-7
        else:
            scores = popularity
        out[case] = scores
    return out


@torch.no_grad()
def ufm_scores(samples, model, table, history_size, batch):
    """Scores of the executed reevaluate_ufm_checkpoints.py, computed on CPU."""
    candidates = samples["candidates"].astype(np.int64) + 1
    indptr, indices = samples["history_indptr"], samples["history_indices"]
    out = np.empty(candidates.shape, dtype=np.float32)
    for start in range(0, len(candidates), batch):
        stop = min(start + batch, len(candidates))
        history = np.zeros((stop - start, history_size), dtype=np.int64)
        for row, case in enumerate(range(start, stop)):
            a, b = indptr[case:case + 2]
            past = indices[max(a, b - history_size):b] + 1
            history[row, :len(past)] = past
        output = model(torch.from_numpy(history), torch.from_numpy(candidates[start:stop]), *table)
        out[start:stop] = output["scores"].float().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite UFM scores.")
    return out


def standardize(scores):
    std = scores.std(1, keepdims=True)
    return np.where(std > 0, (scores - scores.mean(1, keepdims=True)) / np.where(std > 0, std, 1.0), 0.0)


def report(ranks, samples, lengths, mask):
    result = summarize(ranks[mask], samples["regime_codes"][mask], lengths[mask], samples["repeat_purchases"][mask])
    result["cold_macro_ndcg@10"] = round(float(np.mean(
        [result["by_regime"][name]["ndcg@10"] for name in REGIMES[:3]])), 6)
    return result


def cold_gain_ci(gain, regimes, mask, seed=42):
    """Bootstrap the cold macro NDCG@10 difference, resampling cases within each cold regime."""
    rng = np.random.default_rng(seed)
    groups = [gain[mask & (regimes == code)] for code in range(3)]
    draws = [np.mean([group[rng.integers(0, len(group), len(group))].mean() for group in groups])
             for _ in range(BOOTSTRAP)]
    return [round(float(value), 6) for value in np.percentile(draws, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="toys_games_full_temporal directory")
    parser.add_argument("--evaluation", type=Path, required=True, help="evaluation_v3 directory")
    parser.add_argument("--run", type=Path, required=True, help="runs/ufm_full_v1")
    parser.add_argument("--counts", type=Path, required=True, help="gpu_cache_h20_t64_v1/train_counts.npy")
    parser.add_argument("--features", type=Path, help="Defaults to <data>/foundation_clip_b32_v1.")
    parser.add_argument("--training-source", type=Path,
                        default=ROOT / "reports/fix_1_15_20261027/fitlab01_finish1/trained_snapshot")
    parser.add_argument("--published", type=Path,
                        default=ROOT / "reports/continuation_20261028/validation_v3_results.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior evidence.")
    started = time.monotonic()
    features = args.features or args.data / "foundation_clip_b32_v1"
    published = json.loads(args.published.read_text(encoding="utf-8"))["files"]
    reference = {"tfidf": published["baseline/metrics.json"]["splits"]["validation"]["tfidf_profile"],
                 "ufm_full": published["ufm_full_v1.json"]["splits"]["validation"]}
    samples_path = args.evaluation / "samples_validation.npz"
    if sha256(samples_path) != published["ufm_full_v1.json"]["sources"]["samples_validation"]:
        raise ValueError("Candidates differ from the published validation v3 set.")
    with np.load(samples_path) as source:
        samples = {name: source[name] for name in source.files}
    with gzip.open(args.evaluation / "samples_validation.csv.gz", "rt", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if [int(row["sample_index"]) for row in rows] != list(range(len(samples["candidates"]))):
        raise ValueError("Validation metadata order mismatch.")
    fit, audit, boundary = temporal_folds([int(row["timestamp"]) for row in rows])
    candidates, lengths = samples["candidates"], np.diff(samples["history_indptr"])

    print("Scoring TF-IDF...", flush=True)
    matrix = sparse.load_npz(args.data / "content/tfidf_all.npz").tocsr()
    tfidf = tfidf_scores(samples, matrix, load_train_counts(args.data))
    del matrix
    gc.collect()

    print("Scoring UFM full...", flush=True)
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if config["feature_marker_sha256"] != sha256(features / "complete.json"):
        raise ValueError("Feature marker changed since training.")
    # Memory-map the checkpoint and features so the run fits beside other programs.
    state = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True, mmap=True)
    if state["config"] != config:
        raise ValueError("Best checkpoint/config mismatch.")
    UFMConfig, UFMRec = model_classes(config, args.training_source)
    model_config = UFMConfig(**state["model_config"])
    model = UFMRec(model_config).eval()
    model.load_state_dict(state["model"])
    del state
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        table = tuple(torch.from_numpy(np.load(features / name, mmap_mode="r"))
                      for name in ("text.npy", "image.npy", "modalities.npy"))
    table += (torch.from_numpy(np.load(args.counts)),)
    ufm = ufm_scores(samples, model, table, model_config.history_size, args.batch)
    del model, table
    gc.collect()

    everything = np.ones(len(candidates), dtype=bool)
    fit_mask, audit_mask = np.zeros_like(everything), np.zeros_like(everything)
    fit_mask[fit], audit_mask[audit] = True, True
    rank = {"tfidf": ranks_of_positive(tfidf, candidates), "ufm_full": ranks_of_positive(ufm, candidates)}

    # Trust the blend only if both single models reproduce the published v3 numbers.
    reproduction = {}
    for name, ranks in rank.items():
        mine = report(ranks, samples, lengths, everything)
        gaps = {regime: abs(mine["by_regime"][regime]["ndcg@10"] - reference[name]["by_regime"][regime]["ndcg@10"])
                for regime in REGIMES}
        reproduction[name] = dict(
            published_ndcg10={regime: reference[name]["by_regime"][regime]["ndcg@10"] for regime in REGIMES},
            recomputed_ndcg10={regime: mine["by_regime"][regime]["ndcg@10"] for regime in REGIMES},
            max_abs_gap=round(max(gaps.values()), 6))
        print(name, reproduction[name], flush=True)
        if name == "tfidf" and max(gaps.values()) > TOLERANCE:  # rebuilt UFM is a new checkpoint
            raise AssertionError(f"{name} does not reproduce the published validation v3 metrics.")

    z_tfidf, z_ufm = standardize(tfidf), standardize(ufm.astype(np.float64))

    def blend(alpha):
        return ranks_of_positive(alpha * z_ufm + (1 - alpha) * z_tfidf, candidates)

    if not (np.array_equal(blend(0.0), rank["tfidf"]) and np.array_equal(blend(1.0), rank["ufm_full"])):
        raise AssertionError("Blend endpoints must reproduce the single-model rankings.")
    grid = {alpha: report(blend(alpha), samples, lengths, fit_mask)["cold_macro_ndcg@10"] for alpha in ALPHAS}
    alpha = max(ALPHAS, key=lambda value: grid[value])
    rank["hybrid"] = blend(alpha)

    audit_reports = {name: report(ranks, samples, lengths, audit_mask) for name, ranks in rank.items()}
    gain = ndcg10(rank["hybrid"]) - ndcg10(rank["tfidf"])
    result = dict(
        protocol=("validation_v3: 1 positive + 99 shared negatives per case. Per-case z-score blend "
                  "alpha * z(UFM full) + (1 - alpha) * z(TF-IDF profile); alpha chosen by cold macro NDCG@10 "
                  "on the earlier timestamps (fit fold) and reported on the later timestamps (audit fold)."),
        test_evaluated=False,
        inputs={
            "samples_validation": sha256(samples_path),
            "samples_metadata": sha256(args.evaluation / "samples_validation.csv.gz"),
            "tfidf_all": sha256(args.data / "content/tfidf_all.npz"),
            "items": sha256(args.data / "content/items.csv.gz"),
            "checkpoint": sha256(args.run / "best.pt"),
            "model_source_ufm_model_py": config["code_sha256"]["ufm_model.py"],
            "features_complete": sha256(features / "complete.json"),
            "text_npy": sha256(features / "text.npy"),
            "image_npy": sha256(features / "image.npy"),
            "modalities_npy": sha256(features / "modalities.npy"),
            "train_counts": sha256(args.counts),
        },
        reproduction=reproduction,
        folds=dict(boundary_timestamp_ms=boundary, fit_samples=int(len(fit)), audit_samples=int(len(audit))),
        alpha_grid_fit_cold_macro_ndcg10={str(value): score for value, score in grid.items()},
        selected_alpha=alpha,
        audit=audit_reports,
        audit_cold_macro_gain_vs_tfidf=dict(
            value=round(audit_reports["hybrid"]["cold_macro_ndcg@10"] - audit_reports["tfidf"]["cold_macro_ndcg@10"], 6),
            ci95=cold_gain_ci(gain, samples["regime_codes"], audit_mask),
            bootstrap_draws=BOOTSTRAP, resampling="cases within each cold regime of the audit fold"),
        all_cases={name: report(ranks, samples, lengths, everything) for name, ranks in rank.items()},
        limitations=[
            "Exploratory and post hoc: the checkpoint was selected on validation v1 and validation v3 is not an "
            "independent holdout; test data were not opened.",
            "Alpha is selected on the earlier half of validation v3; all_cases includes that half and is optimistic.",
            "Single training seed (42); sampled ranking over 100 candidates, not full-catalog or online results.",
            "UFM scores were recomputed on CPU; the published v3 run used FITLAB, so near-ties may differ slightly.",
        ],
        runtime_seconds=round(time.monotonic() - started, 1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"selected_alpha": alpha, "fit_grid": grid,
                      "audit_cold_macro": {name: value["cold_macro_ndcg@10"] for name, value in audit_reports.items()},
                      "gain_vs_tfidf": result["audit_cold_macro_gain_vs_tfidf"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
