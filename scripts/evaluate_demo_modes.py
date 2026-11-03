"""Measure the demo's ranking modes on the real catalog; nothing is trained.

1. Replays the six API cases recorded on 29/10 in the original mode (UFM, full catalog) and
   checks that the current implementation returns the same items, scores and branch weights.
2. Times every mode: UFM or UFM + TF-IDF, scoring the full catalog or the fast scope.
3. Measures how much of the full-catalog top 10 the fast scope returns, using histories of
   validation v3 cases with a known user (histories only; target items are never read).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import write_json  # noqa: E402
from recommend_catalog import FAST_CANDIDATES, HYBRID_ALPHA, Catalog, UFMBackend  # noqa: E402

REGIMES = ("all", "zero_shot", "extreme_cold", "cold", "warm")
MODES = [("ufm", "full"), ("ufm", "fast"), ("hybrid", "full"), ("hybrid", "fast")]


def timed(backend, history, k, regime, model, scope):
    started = time.perf_counter()
    rows = backend.recommend(history, k, regime, model=model, scope=scope)
    return rows, time.perf_counter() - started


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(n=int(len(values)), mean=round(float(values.mean()), 4), min=round(float(values.min()), 4),
                p50=round(float(np.percentile(values, 50)), 4), max=round(float(values.max()), 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="toys_games_full_temporal directory")
    parser.add_argument("--run", type=Path, required=True, help="runs/ufm_full_v1")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--training-source", type=Path,
                        default=ROOT / "reports/fix_1_15_20261027/fitlab01_finish1/trained_snapshot")
    parser.add_argument("--evaluation", type=Path, required=True, help="evaluation_v3 directory; histories only")
    parser.add_argument("--recorded", type=Path,
                        default=ROOT / "reports/nontraining_20261029/demo_api_verification.json")
    parser.add_argument("--histories", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior evidence.")
    import psutil

    started = time.perf_counter()
    catalog = Catalog(args.data)
    backend = UFMBackend(catalog, args.run, features=args.features, training_source=args.training_source)
    load_seconds = time.perf_counter() - started
    rss_mib = psutil.Process().memory_info().rss / 2**20
    print(f"Loaded in {load_seconds:.1f}s, RSS {rss_mib:.0f} MiB", flush=True)

    # 1. The faster implementation must reproduce the recorded full-catalog answers.
    replay = []
    for case in json.loads(args.recorded.read_text(encoding="utf-8"))["cases"]:
        request, before = case["request"], case["response"]["results"]
        rows, seconds = timed(backend, request["history"], request["k"], request["regime"], "ufm", "full")
        same = [r["asin"] for r in rows] == [r["asin"] for r in before]
        replay.append(dict(
            name=case["name"], request=request, same_items=same,
            max_score_gap=max(abs(a["score"] - b["score"]) for a, b in zip(rows, before)),
            max_weight_gap=max(abs(x - y) for a, b in zip(rows, before) for x, y in zip(a["weights"], b["weights"])),
            seconds_29_10=round(case["wall_seconds"], 2), seconds_now=round(seconds, 2)))
        print("replay", replay[-1], flush=True)
    if not all(case["same_items"] and case["max_score_gap"] < 1e-4 for case in replay):
        raise AssertionError("Full-catalog mode no longer reproduces the recorded API cases.")

    # 2. Latency of every mode on the recorded history.
    latency = {}
    for model, scope in MODES:
        timed(backend, ["B004S8F7QM"], 5, "all", model, scope)  # warm-up
        values = [timed(backend, ["B004S8F7QM"], 5, "all", model, scope)[1] for _ in range(args.repeats)]
        latency[f"{model}_{scope}"] = dict(summary(values), candidates=backend.last_stats["candidates"])
        print("latency", model, scope, latency[f"{model}_{scope}"], flush=True)

    # 3. Fast scope against the full catalog on real histories (no target items are read).
    with np.load(args.evaluation / "samples_validation.npz") as source:
        indptr, indices = source["history_indptr"], source["history_indices"]
    known = np.flatnonzero(np.diff(indptr) > 0)
    chosen = np.random.default_rng(42).choice(known, size=args.histories, replace=False)
    histories = [[catalog.asins[r] for r in indices[indptr[i]:indptr[i + 1]][-20:]] for i in sorted(chosen)]
    fidelity = {}
    for model in ("ufm", "hybrid"):
        for regime in REGIMES:
            overlap, top1, seconds = [], [], {"full": [], "fast": []}
            for history in histories:
                full, full_s = timed(backend, history, 10, regime, model, "full")
                fast, fast_s = timed(backend, history, 10, regime, model, "fast")
                if not full:
                    continue
                full_items = [r["asin"] for r in full]
                overlap.append(len(set(full_items) & {r["asin"] for r in fast}) / len(full_items))
                top1.append(bool(fast) and fast[0]["asin"] == full_items[0])
                seconds["full"].append(full_s)
                seconds["fast"].append(fast_s)
            fidelity[f"{model}_{regime}"] = dict(
                histories=len(overlap), mean_overlap_at_10=round(float(np.mean(overlap)), 4),
                min_overlap_at_10=round(float(np.min(overlap)), 4), same_top1=round(float(np.mean(top1)), 4),
                full_seconds_p50=round(float(np.median(seconds["full"])), 3),
                fast_seconds_p50=round(float(np.median(seconds["fast"])), 3))
            print("fidelity", model, regime, fidelity[f"{model}_{regime}"], flush=True)

    result = dict(
        observed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scope=("In-process ranking on CPU (no HTTP); full catalog of 767,045 items, checkpoint epoch 9. "
               "Fast scope: UFM rescores the union of the nearest items by its own semantic and behaviour "
               "embeddings, TF-IDF and popularity."),
        hardware=dict(platform=platform.platform(), logical_cpu=os.cpu_count(),
                      physical_ram_mib=round(psutil.virtual_memory().total / 2**20, 1), torch_threads=4),
        load_seconds=round(load_seconds, 1), rss_after_load_mib=round(rss_mib, 1),
        fast_candidates_per_source=FAST_CANDIDATES, hybrid_alpha=HYBRID_ALPHA,
        replay_29_10_full_catalog=replay, latency_seconds=latency,
        fast_vs_full_top10=fidelity, sampled_histories=len(histories),
        limitations=[
            "Latency on one Windows laptop with other programs running; not a concurrent benchmark.",
            "Overlap compares the fast scope with the full-catalog ranking of the same model; "
            "it is not recommendation-quality evidence.",
            "The hybrid alpha was chosen on 100-candidate validation cases; it is not tuned for "
            "full-catalog or fast-scope ranking.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(latency=latency, replay_ok=True), indent=2), flush=True)


if __name__ == "__main__":
    main()
