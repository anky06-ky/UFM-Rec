"""Verify a frozen Toys protocol and run one GPU training job at a time.

Default is a read-only preflight. --execute trains/re-evaluates all declared jobs
in stage order, with checkpoints and logs outside Git. This script never opens test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/multimodal_followup_20261028.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(protocol: dict, source_root: Path, workspace: Path) -> dict:
    checked = {}
    for relative, expected in protocol["sources_sha256"].items():
        path = (source_root if relative.startswith("src/") else workspace) / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or changed locked input: {path}")
        checked[relative] = expected
    if protocol["seeds"] != [42, 7, 2026]:
        raise ValueError("Unexpected seed list")
    if [[*stage["variants"]] for stage in protocol["stages"]] != [
        ["full", "semantic_only", "fixed_fusion"], ["count_aware_fusion"], ["adaptive_dropout"]
    ]:
        raise ValueError("Unexpected model comparison order")
    if protocol["training"]["max_steps"] != 0 or protocol["training"]["validation_cap"] != 0:
        raise ValueError("Partial train/validation is not part of the locked protocol")
    return checked


def command_for(protocol: dict, source_root: Path, workspace: Path, variant: str, seed: int,
                output: Path, resume: bool) -> list[str]:
    data = workspace / "data/processed/toys_games_full_temporal"
    command = [sys.executable, "-u", str(source_root / "src/train_ufm_recommender.py"),
               "--data", str(data), "--legacy-cache", str(workspace / "data/processed/gpu_cache_h20_t64_v1"),
               "--cache", str(workspace / "data/processed/ufm_positive_graph_h20_v1"),
               "--features", str(data / "foundation_clip_b32_v1"),
               "--output", str(output), "--device", "cuda", "--variant", variant, "--seed", str(seed)]
    for key, value in protocol["training"].items():
        flag = "--" + key.replace("_", "-")
        command += [flag if value else "--no-amp"] if key == "amp" else [flag, str(value)]
    if resume:
        command.append("--resume")
    return command


def verify_completed(output: Path, variant: str, seed: int, protocol: dict) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    marker = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if (config.get("variant") != variant or config.get("seed") != seed
            or config.get("device") != "cuda" or marker.get("smoke_run") is not False
            or marker.get("test_evaluated") is not False or marker.get("steps", 0) < 1
            or not (output / "best.pt").is_file()):
        raise ValueError(f"Completed run is invalid: {output}")
    for key, value in protocol["training"].items():
        if config.get(key) != value:
            raise ValueError(f"Training setting changed in {output}: {key}")
    for relative, expected in protocol["sources_sha256"].items():
        if relative.startswith("src/") and config.get("code_sha256", {}).get(Path(relative).name) != expected:
            raise ValueError(f"Training source changed in {output}: {relative}")
    if config.get("feature_marker_sha256") != protocol["sources_sha256"][
        "data/processed/toys_games_full_temporal/foundation_clip_b32_v1/complete.json"]:
        raise ValueError("Feature marker changed")


def gpu_idle() -> bool:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; full runs require a GPU host")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return not any(line.strip().isdigit() for line in result.stdout.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--workspace", type=Path, default=ROOT,
                        help="Complete FITLAB workspace or verified local restore")
    parser.add_argument("--output-root", type=Path,
                        help="Fresh experiment root, defaults to source-root/runs/ufm_followup_20261028")
    parser.add_argument("--report-root", type=Path,
                        help="Fresh report root, defaults to source-root/reports/ufm_followup_20261028")
    parser.add_argument("--execute", action="store_true", help="Start declared GPU jobs sequentially")
    parser.add_argument("--max-new-jobs", type=int, default=0,
                        help="Limit newly trained runs per invocation so results can be backed up between batches; 0 means all")
    args = parser.parse_args()
    if args.max_new_jobs < 0:
        parser.error("--max-new-jobs must be nonnegative")
    protocol = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_root, workspace = args.source_root.resolve(), args.workspace.resolve()
    checked = preflight(protocol, source_root, workspace)
    output_root = args.output_root or source_root / "runs/ufm_followup_20261028"
    report_root = args.report_root or source_root / "reports/ufm_followup_20261028"
    jobs = [(stage["name"], variant, seed) for stage in protocol["stages"]
            for variant in stage["variants"] for seed in protocol["seeds"]]
    print(json.dumps({"preflight": "passed", "locked_inputs": len(checked),
                      "jobs": len(jobs), "mode": "execute" if args.execute else "dry_run",
                      "output_root": str(output_root), "report_root": str(report_root)}), flush=True)
    if not args.execute:
        for stage, variant, seed in jobs:
            print(f"{stage}: {variant} seed={seed}")
        return
    pending = sum(not ((output_root / f"{variant}_s{seed}") / "completed.json").is_file()
                  for _, variant, seed in jobs)
    planned = min(pending, args.max_new_jobs) if args.max_new_jobs else pending
    disk_path = output_root
    while not disk_path.exists():
        disk_path = disk_path.parent
    free = shutil.disk_usage(disk_path).free
    required = (planned * 2 + 2) * 2**30
    if free < required:
        raise RuntimeError(f"Only {free / 2**30:.1f} GiB free at {disk_path}; "
                           f"need about {required / 2**30:.1f} GiB for {planned} new runs plus reserve. "
                           "Use a larger --output-root or --max-new-jobs 1 and back up each batch locally.")
    started = 0
    for stage, variant, seed in jobs:
        label = f"{variant}_s{seed}"
        output = output_root / label
        report = report_root / f"{label}_v3.json"
        predictions = report_root / f"{label}_v3_scores"
        if not (output / "completed.json").exists():
            if args.max_new_jobs and started >= args.max_new_jobs:
                print("Batch limit reached; back up new artifacts locally before resuming", flush=True)
                return
            if not gpu_idle():
                raise RuntimeError(f"GPU busy before {label}; no other job was stopped")
            resume = (output / "latest.pt").is_file()
            if output.exists() and any(output.iterdir()) and not resume:
                raise FileExistsError(f"Incomplete output with no resumable checkpoint: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with (output.parent / f"{label}.log").open("a", encoding="utf-8") as log:
                subprocess.run(command_for(protocol, source_root, workspace, variant, seed, output, resume),
                               cwd=source_root, stdout=log, stderr=subprocess.STDOUT, check=True)
            started += 1
        verify_completed(output, variant, seed, protocol)
        if report.exists() or predictions.exists():
            if not (report.is_file() and (predictions / "scores_validation.npz").is_file()):
                raise FileExistsError(f"Partial evaluation needs manual audit: {label}")
            print(f"verified existing {label}; report already present", flush=True)
            continue
        command = [sys.executable, "-u", str(source_root / "src/reevaluate_ufm_checkpoints.py"),
                   "--checkpoint", str(output / "best.pt"), "--evaluation",
                   str(workspace / "reports/fix_1_15_20261027/fitlab01_finish1/evaluation_v3"),
                   "--features", str(workspace / "data/processed/toys_games_full_temporal/foundation_clip_b32_v1"),
                   "--counts", str(workspace / "data/processed/gpu_cache_h20_t64_v1/train_counts.npy"),
                   "--output", str(report), "--save-predictions", str(predictions),
                   "--device", "cuda", "--splits", "validation"]
        report_root.mkdir(parents=True, exist_ok=True)
        with (report_root / f"{label}_v3.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=source_root, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not report.is_file() or not (predictions / "scores_validation.npz").is_file():
            raise RuntimeError(f"Missing v3 evidence for {label}")
        print(f"complete {stage}: {label}", flush=True)


if __name__ == "__main__":
    main()
