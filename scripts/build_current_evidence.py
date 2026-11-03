"""Build report tables and figures from verified saved v3 evidence; no training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/continuation_20261028"
OUT = ROOT / "reports/nontraining_20261029"
FIG = ROOT / "output/figures/current_v3_20261029"
NAMES = {
    "tfidf_profile": "TF-IDF", "ufm_ablation_semantic_only_s42_v1": "Semantic-only",
    "ufm_ablation_no_cross_align_s42_v1": "No cross-align",
    "ufm_ablation_no_uncertainty_s42_v1": "No uncertainty",
    "ufm_full_v1": "UFM full", "ufm_ablation_text_only_s42_v1": "Text-only",
    "ufm_ablation_fixed_fusion_s42_v1": "Fixed fusion",
    "ufm_ablation_image_only_s42_v1": "Image-only",
    "ufm_ablation_collaborative_only_s42_v1": "Collaborative-only", "popularity": "Popularity",
}


def read(name):
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def number(value):
    return f"{value:.6f}".replace(".", ",")


def build():
    summary = read("summary.json")
    files = read("validation_v3_results.json")["files"]
    counts = read("count_effect_v3.json")
    benchmark = read("windows_ufm_benchmark.json")
    candidate = summary["candidate_sha256"]
    assert counts["sources_sha256"]["samples"] == candidate
    models = {}
    for key in NAMES:
        source = files["baseline/metrics.json"] if key in {"tfidf_profile", "popularity"} else files[key + ".json"]
        assert source["test_evaluated"] is False
        assert source["sources"]["samples_validation"] == candidate
        metrics = source["splits"]["validation"]
        if key in {"tfidf_profile", "popularity"}:
            metrics = metrics[key]
        assert metrics["overall"]["samples"] == 40000
        assert all(group["samples"] == 10000 for group in metrics["by_regime"].values())
        assert abs(metrics["cold_macro_ndcg@10"] - summary["cold_macro_ndcg10"][key]) < 1e-6
        models[key] = {"name": NAMES[key], "metrics": metrics}
    evidence = {
        "as_of": "2026-10-29", "operation": "summarize_saved_evidence_no_training",
        "protocol": "validation_v3", "candidate_sha256": candidate,
        "samples": 40000, "candidates_per_case": 100, "training_seeds": [42],
        "models": models, "count_effect": counts,
        "calibration": summary["full_calibration"], "windows_benchmark": benchmark,
        "limitations": summary["limitations"],
        "sources_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                           for name in ["summary.json", "validation_v3_results.json",
                                        "count_effect_v3.json", "windows_ufm_benchmark.json"]},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["| Mô hình | Cold macro NDCG@10 | Overall NDCG@10 | Recall@10 | MRR@10 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for model in models.values():
        m = model["metrics"]
        vals = [m["cold_macro_ndcg@10"]] + [m["overall"][key] for key in ["ndcg@10", "recall@10", "mrr@10"]]
        table.append("| " + model["name"] + " | " + " | ".join(map(number, vals)) + " |")
    (OUT / "ranking_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    labels = [item["name"] for item in models.values()]
    values = [item["metrics"]["cold_macro_ndcg@10"] for item in models.values()]
    ax.barh(labels[::-1], values[::-1], color=["#176b73" if x == "UFM full" else "#a5b3bf" for x in labels[::-1]])
    for i, value in enumerate(values[::-1]):
        ax.text(value + 0.002, i, f"{value:.6f}", va="center", fontsize=9)
    ax.set(xlim=(0, 0.235), xlabel="Cold macro NDCG@10 (0–1)", ylabel="Mô hình",
           title="Validation v3 • So sánh cold macro NDCG@10")
    fig.text(0.12, -0.035, "Nguồn: validation_v3_results.json • 28/10/2026 • 40.000 ca, 100 ứng viên/ca\nMột seed UFM 42; checkpoint chọn bằng v1; kết quả thăm dò.", fontsize=8)
    fig.savefig(FIG / "cold_macro.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / "cold_macro.svg", bbox_inches="tight")
    plt.close(fig)
    labels = ["0", "1–5", "6–20", "21+"]
    groups = list(counts["by_positive_train_count"].values())
    delta = [group["paired_full_minus_semantic"] for group in groups]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
    ax.bar(labels, delta, color=["#9a624d" if x < 0 else "#176b73" for x in delta], width=0.5)
    ax.axhline(0, color="#555555", linewidth=0.8)
    for i, value in enumerate(delta):
        ax.text(i, value + (0.001 if value >= 0 else -0.001), f"{value:+.6f}",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    ax.set(ylim=(-0.032, 0.023), xlabel="Số tương tác của sản phẩm trong train",
           ylabel="Chênh lệch NDCG@10\nFull − semantic-only",
           title="Chênh lệch thay đổi theo mức độ thưa dữ liệu")
    fig.text(0.12, -0.05, "Nguồn: count_effect_v3.json • 28/10/2026 • 10.000 ca/nhóm • seed 42\nSo sánh dự đoán đã lưu; chưa xác định nguyên nhân của chênh lệch.", fontsize=8)
    fig.savefig(FIG / "count_effect.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / "count_effect.svg", bbox_inches="tight")
    plt.close(fig)
    print("Verified 10 models on shared v3 candidates; created evidence, table and two figures.")


if __name__ == "__main__":
    build()
