"""Small synthetic CUDA check for the five locked UFM variants.

This does not read the recommendation dataset, open test labels, or report
ranking metrics. It verifies one forward/backward/optimizer step per variant.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ufm_model import UFMConfig, UFMRec, ufm_loss  # noqa: E402


VARIANTS = (
    "full",
    "semantic_only",
    "fixed_fusion",
    "count_aware_fusion",
    "adaptive_dropout",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON receipt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke check")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    text = torch.randn(9, 12, device=device)
    image = torch.randn(9, 12, device=device)
    modalities = torch.ones(9, 2, dtype=torch.bool, device=device)
    modalities[0] = False
    modalities[8] = False
    counts = torch.tensor([0, 40, 3, 2, 1, 1, 0, 0, 0], device=device)
    histories = torch.tensor([[1, 2, 0], [2, 3, 4], [0, 0, 0]], device=device)
    candidates = torch.tensor([[3, 6, 8], [5, 6, 7], [3, 6, 8]], device=device)
    valid_negatives = torch.tensor(
        [[True, False], [True, True], [False, False]], device=device
    )
    base = UFMConfig(items=8, feature_dim=12, dim=16, history_size=3,
                     heads=4, layers=2, dropout=0.1, id_dropout=0.5)
    checks = {}
    for variant in VARIANTS:
        model = UFMRec(replace(base, variant=variant)).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        output = model(histories, candidates, text, image, modalities, counts)
        loss = ufm_loss(output, valid_negatives)["total"]
        if not torch.isfinite(output["scores"]).all() or not torch.isfinite(loss):
            raise AssertionError(f"Non-finite CUDA forward/loss: {variant}")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f"Missing or non-finite CUDA gradients: {variant}")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            after = model(histories, candidates, text, image, modalities, counts)
        if not torch.isfinite(after["scores"]).all():
            raise AssertionError(f"Non-finite CUDA scores after update: {variant}")
        checks[variant] = {
            "forward_finite": True,
            "loss_finite": True,
            "backward_finite": True,
            "optimizer_step_finite": True,
        }
    torch.cuda.synchronize()
    receipt = {
        "scope": "synthetic_cuda_smoke_only",
        "dataset_opened": False,
        "test_labels_opened": False,
        "ranking_metric_reported": False,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
        "model_source_sha256": hashlib.sha256((ROOT / "src/ufm_model.py").read_bytes()).hexdigest(),
        "variants": checks,
    }
    rendered = json.dumps(receipt, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
