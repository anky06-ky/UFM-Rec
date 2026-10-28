"""Re-evaluate a trained UFM checkpoint against a given evaluation directory.

Same protocol as train_ufm_recommender.evaluate (last-20 history, +1 ID shift,
tie_noise ranking, summarize + cold macro + calibration + branch weights), but
read-only: no training state is touched and no output is overwritten.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from common import sha256, write_json
from evaluation_ledger import TestEvaluationLedger, import_known_evidence
from contextlib import nullcontext
from evaluate_content_baseline import REGIMES, summarize, tie_noise
from ufm_model import candidate_calibration
from trained_ufm import model_classes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"


def selected_validation(samples, cap, seed):
    if not cap:
        return np.arange(len(samples["candidates"]))
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(samples["regime_codes"] == code)) for code in range(4)]
    chosen = np.concatenate([group[:max(1, cap // 4)] for group in groups])
    return np.sort(chosen)


@torch.no_grad()
def evaluate(model, arrays, table, history_size, batch_size, cap, seed, amp, prediction_output=None):
    model.eval()
    select = selected_validation(arrays, cap, seed)
    if len(select) == 0:
        raise ValueError("Empty validation set.")
    scores_all, ranks, lengths, gates = [], [], [], []
    no_signal_pairs = 0
    device = table[0].device
    for start in range(0, len(select), batch_size):
        idx = select[start:start + batch_size]
        candidate = arrays["candidates"][idx].astype(np.int64) + 1
        history = np.zeros((len(idx), history_size), dtype=np.int64)
        for j, source in enumerate(idx):
            a, b = arrays["history_indptr"][source:source + 2]
            past = arrays["history_indices"][max(a, b - history_size):b] + 1
            history[j, :len(past)] = past
            lengths.append(int(b - a))
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            output = model(torch.as_tensor(history, device=device), torch.as_tensor(candidate, device=device), *table)
        scores = output["scores"].float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite validation scores.")
        no_signal_pairs += int((~output["available"].any(-1)).sum())
        adjusted = scores.astype(np.float64) + tie_noise(candidate - 1) * 1e-10
        ranks.extend((1 + (adjusted[:, 1:] > adjusted[:, :1]).sum(-1)).tolist())
        scores_all.append(scores)
        gates.append(output["weights"][:, 0].float().cpu().numpy())
    report = summarize(np.asarray(ranks), arrays["regime_codes"][select],
                       np.asarray(lengths), arrays.get("repeat_purchases")[select]
                       if "repeat_purchases" in arrays else None)
    cold = [report["by_regime"][name] for name in REGIMES[:3]]
    if not all(group["samples"] for group in cold):
        raise ValueError("Need all three cold regimes to select a checkpoint.")
    report["cold_macro_ndcg@10"] = float(np.mean([group["ndcg@10"] for group in cold]))
    scores = np.concatenate(scores_all)
    if prediction_output is not None:
        prediction_output = Path(prediction_output)
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        with prediction_output.open('xb') as target:
            np.savez_compressed(target, scores=scores, sample_indices=select,
                                candidates=arrays['candidates'][select])
    report["calibration_unscaled"] = candidate_calibration(torch.from_numpy(scores))
    weights = np.concatenate(gates)
    report["positive_mean_branch_weights"] = {name: weights[arrays["regime_codes"][select] == code].mean(0).tolist()
        for code, name in enumerate(REGIMES) if np.any(arrays["regime_codes"][select] == code)}
    report["no_signal_pairs"] = no_signal_pairs
    report["protocol"] = "Fixed validation candidates; 1 positive + 99 negatives on real dataset; sampled ranking, not full catalog."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt or latest.pt")
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="Directory holding samples_{split}.npz.")
    parser.add_argument("--features", type=Path, default=DATA / "foundation_clip_b32_v1")
    parser.add_argument("--counts", type=Path,
                        default=ROOT / "data/processed/gpu_cache_h20_t64_v1/train_counts.npy")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--splits", default="validation")
    parser.add_argument("--evaluation-ledger", type=Path, default=ROOT / "reports/test_evaluation_ledger.json")
    parser.add_argument("--allow-seen-test", action="store_true")
    parser.add_argument("--test-purpose")
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--training-source", type=Path,
                        help="Verified archived training source for relocated checkpoints.")
    parser.add_argument("--save-predictions", type=Path,
                        help="New directory for scores/sample indices/candidates for each split.")
    args = parser.parse_args()
    if args.batch < 1 or args.cap < 0:
        parser.error('Batch must be positive and cap nonnegative.')

    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior evaluation evidence.")
    if args.save_predictions is not None and args.save_predictions.exists():
        raise FileExistsError('Prediction directory must be new.')
    if any(split not in {"validation", "test"} for split in args.splits.split(",")):
        parser.error("Splits must be validation and/or test.")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    UFMConfig, UFMRec = model_classes(state['config'], args.training_source)
    model_config = UFMConfig(**state["model_config"])
    model = UFMRec(model_config)
    model.load_state_dict(state["model"])
    device = torch.device(args.device)
    model = model.to(device)

    counts = np.load(args.counts)
    expected_marker = state['config'].get('feature_marker_sha256')
    if not expected_marker or sha256(args.features/'complete.json') != expected_marker:
        raise ValueError('Feature marker does not match checkpoint.')
    expected_counts = state['config'].get('cache', {}).get('sources', {}).get('train_counts.npy')
    if expected_counts and sha256(args.counts) != expected_counts:
        raise ValueError('Train counts do not match checkpoint.')
    table = (torch.from_numpy(np.load(args.features / "text.npy")).to(device),
             torch.from_numpy(np.load(args.features / "image.npy")).to(device),
             torch.from_numpy(np.load(args.features / "modalities.npy")).to(device),
             torch.from_numpy(counts).to(device))

    recorded_config = state.get("config", {})
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_variant": recorded_config.get("variant"),
        "checkpoint_seed": recorded_config.get("seed"),
        "recorded_code_sha256": recorded_config.get("code_sha256"),
        "training_source": str(args.training_source) if args.training_source else str(Path(__file__).parent),
        "model_config": asdict(model_config),
        "evaluation": str(args.evaluation),
        "sources": {
            "features_complete": sha256(args.features / "complete.json"),
            "text_npy": sha256(args.features / "text.npy"),
            "image_npy": sha256(args.features / "image.npy"),
            "modalities_npy": sha256(args.features / "modalities.npy"),
            "train_counts": sha256(args.counts),
        },
        "splits": {},
    }
    for split in args.splits.split(","):
        arrays_path = args.evaluation / f"samples_{split}.npz"
        report["sources"][f"samples_{split}"] = sha256(arrays_path)
        identity = dict(checkpoint_sha256=report['checkpoint_sha256'],
                        samples_test_sha256=sha256(arrays_path),
                        dataset_sha256=sha256(args.evaluation.parent/'split_manifest.json')
                        if (args.evaluation.parent/'split_manifest.json').exists() else None,
                        scorer_sha256=sha256(Path(__file__)), output=str(args.output))
        ledger = TestEvaluationLedger(args.evaluation_ledger)
        if split == 'test':
            import_known_evidence(ROOT, ledger)
        context = ledger.evaluation(identity, allow_seen=args.allow_seen_test,
                                    purpose=args.test_purpose) if split == 'test' else nullcontext(None)
        with context as evaluation:
            with np.load(arrays_path) as source:
                arrays = {name: source[name] for name in source.files}
            report['splits'][split] = evaluate(
                model, arrays, table, model_config.history_size, args.batch, args.cap, args.seed, args.amp,
                args.save_predictions / f'scores_{split}.npz' if args.save_predictions else None)
            if evaluation is not None:
                evaluation['test_metrics'] = report['splits'][split]['overall']
                report['test_history'] = dict(id=evaluation['id'], first_evaluation=evaluation['first_evaluation'])
    report['test_evaluated'] = 'test' in report['splits']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({split: {"cold_macro": value["cold_macro_ndcg@10"],
                              "overall": value["overall"]["ndcg@10"],
                              "no_signal_pairs": value["no_signal_pairs"]}
                      for split, value in report["splits"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
