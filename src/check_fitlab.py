"""Read-only checks before training the full temporal dataset on FITLAB."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "toys_games_full_temporal"


def main():
    import torch

    print(f"Project: {ROOT}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: select the FITLAB GPU Python environment.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}; VRAM: {props.total_memory / 2**30:.1f} GiB", flush=True)
    with torch.no_grad():
        x = torch.randn(1024, 1024, device=device)
        y = x @ x
        torch.cuda.synchronize()
        assert torch.isfinite(y).all().item(), "Non-finite GPU result"
    print("GPU smoke test: PASS", flush=True)
    print(f"Storage free: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB", flush=True)

    manifest = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))
    print(f"Split policy: {manifest['policy']['split']}", flush=True)
    print(f"Interaction counts (manifest): {manifest['interactions']}", flush=True)
    for split in ("train", "validation", "test"):
        for prefix in ("interactions_", ""):
            name = f"interactions_{split}.csv.gz" if prefix else f"{split}_with_history.csv.gz"
            path = DATA / name
            with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                required = {"user_id", "parent_asin", "timestamp"}
                if not prefix:
                    required.add("history")
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Unexpected columns in {name}: {reader.fieldnames}")
                if next(reader, None) is None:
                    raise ValueError(f"Empty dataset: {name}")
            print(f"Readable: {name} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    for name in ("content/items.csv.gz", "content/products_text.csv.gz", "content/tfidf_all.npz",
                 "evaluation/samples_validation.npz", "evaluation/samples_test.npz"):
        path = DATA / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty file: {path}")
    print("FITLAB preflight: PASS (GPU + files + sample headers; not a full leakage audit)", flush=True)


if __name__ == "__main__":
    main()
