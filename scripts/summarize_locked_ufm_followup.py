"""Summarize all 15 locked v3 validation reports; refuse incomplete evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/multimodal_followup_20261028.json"
REGIMES = ("zero_shot", "extreme_cold", "cold", "warm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_sample = protocol["sources_sha256"][
        "reports/fix_1_15_20261027/fitlab01_finish1/evaluation_v3/samples_validation.npz"]
    variants = [variant for stage in protocol["stages"] for variant in stage["variants"]]
    summary = {"protocol": protocol["name"], "candidate_sha256": expected_sample,
               "evaluation": "validation_v3_exploratory_not_independent_holdout",
               "variants": {}, "test_evaluated": False}
    for variant in variants:
        per_seed = {}
        for seed in protocol["seeds"]:
            path = args.reports / f"{variant}_s{seed}_v3.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            result = report["splits"]["validation"]
            if (report.get("test_evaluated") is not False or set(report["splits"]) != {"validation"}
                    or report["sources"]["samples_validation"] != expected_sample
                    or report["checkpoint_variant"] != variant
                    or report["checkpoint_seed"] != seed
                    or report["model_config"]["variant"] != variant
                    or report["recorded_code_sha256"] != {
                        Path(relative).name: expected for relative, expected in protocol["sources_sha256"].items()
                        if relative.startswith("src/")
                    }
                    or report["sources"]["features_complete"] != protocol["sources_sha256"][
                        "data/processed/toys_games_full_temporal/foundation_clip_b32_v1/complete.json"]
                    or report["sources"]["train_counts"] != protocol["sources_sha256"][
                        "data/processed/gpu_cache_h20_t64_v1/train_counts.npy"]):
                raise ValueError(f"Protocol mismatch: {path}")
            if result["overall"]["samples"] != 40000 or any(
                result["by_regime"][regime]["samples"] != 10000 for regime in REGIMES
            ):
                raise ValueError(f"Unexpected validation case count: {path}")
            per_seed[str(seed)] = {
                "checkpoint_sha256": report["checkpoint_sha256"],
                "cold_macro_ndcg10": result["cold_macro_ndcg@10"],
                "overall_ndcg10": result["overall"]["ndcg@10"],
                "regime_ndcg10": {r: result["by_regime"][r]["ndcg@10"] for r in REGIMES},
                "history_ndcg10": {"known_user": result["known_user"]["ndcg@10"],
                                   "empty_history": result["empty_history"]["ndcg@10"]},
            }
        vals = [per_seed[str(seed)]["cold_macro_ndcg10"] for seed in protocol["seeds"]]
        summary["variants"][variant] = {"per_seed": per_seed,
                                        "cold_macro_ndcg10_mean": statistics.mean(vals),
                                        "cold_macro_ndcg10_sample_std": statistics.stdev(vals)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"mean": v["cold_macro_ndcg10_mean"],
                          "std": v["cold_macro_ndcg10_sample_std"]}
                      for k, v in summary["variants"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
