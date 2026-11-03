"""Draw the figures of the full report from saved evidence only (no training).

Outputs PNGs into output/figures/report_20261103/. Every number comes from a
JSON file in the repository; the cropped demo views come from the 29/10 UI
acceptance screenshots.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures/report_20261103"
TOYS = ROOT / "data/processed/toys_games_full_temporal"
BEAUTY = ROOT / "data/processed/all_beauty_full_temporal"
EVIDENCE = ROOT / "reports/nontraining_20261029/evidence.json"
HISTORY = ROOT / "reports/final_20261103/ufm_full_v1_training_history.json"
SHOTS = ROOT / "reports/nontraining_20261029"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
# Fixed categorical order of the validated palette; colour follows the entity in every chart.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ENTITY = {"TF-IDF": BLUE, "Semantic-only": ORANGE, "UFM full": AQUA, "Popularity": YELLOW}
REGIMES = [("zero_shot", "Zero-shot\n(0)"), ("extreme_cold", "Extreme cold\n(1–5)"),
           ("cold", "Cold\n(6–20)"), ("warm", "Warm\n(>20)")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False, "figure.dpi": 100,
})


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def vn(value: float, digits: int = 0) -> str:
    """Vietnamese number format: dot thousands, comma decimals."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


def finish(fig, name: str, note: str | None = None) -> None:
    if note:  # place the source line just below everything already drawn, never over tick labels
        fig.canvas.draw()
        bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
        fig.text(0.01, bottom - 0.015, note, fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / name)


def bar_labels(ax, bars, values, fmt, inside=False):
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, bar.get_height(), fmt(value), ha="center", va="bottom", fontsize=8, color=INK)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def data_figures() -> None:
    split = load(TOYS / "split_manifest.json")
    raw = split["raw"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    stages = [("Đánh giá thô", raw["raw_records"]),
              ("Mua đã xác minh,\nrating ≥ 4", raw["positive_verified_records"]),
              ("Sau khử trùng", split["retained_unique_records"])]
    y = range(len(stages))[::-1]
    bars = a.barh(list(y), [v for _, v in stages], color=BLUE, height=0.55)
    a.set_yticks(list(y), [s for s, _ in stages])
    for bar, (_, v) in zip(bars, stages):
        a.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + vn(v), va="center", fontsize=8.5, color=INK)
    a.set_xlim(0, raw["raw_records"] * 1.32)
    a.set_xticks([0, 4e6, 8e6, 12e6, 16e6, 20e6])
    a.set_title("(a) Lọc tương tác dương", loc="left")
    a.xaxis.set_major_formatter(lambda v, _: vn(v / 1e6, 0) + " tr")
    grid(a, "x")
    parts = [("Train", split["interactions"]["train"]), ("Validation", split["interactions"]["validation"]),
             ("Test", split["interactions"]["test"])]
    bars = b.bar([p for p, _ in parts], [v for _, v in parts], color=BLUE, width=0.55)
    for bar, (_, v) in zip(bars, parts):
        b.text(bar.get_x() + bar.get_width() / 2, v, vn(v) + "\n" + vn(100 * v / split["retained_unique_records"], 1) + "%",
               ha="center", va="bottom", fontsize=8.5, color=INK)
    b.set_ylim(0, split["interactions"]["train"] * 1.25)
    b.yaxis.set_major_formatter(lambda v, _: vn(v / 1e6, 0) + " tr")
    b.set_title("(b) Chia theo thời gian 80/10/10", loc="left")
    grid(b)
    finish(fig, "data_funnel_split.png", "Nguồn: data/processed/toys_games_full_temporal/split_manifest.json")

    catalog = load(TOYS / "foundation_catalog_v1/catalog_report.json")["regimes"]
    rows = split["regime_rows"]["validation"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, values, title in ((a, [catalog[r]["items"] for r, _ in REGIMES], "(a) Sản phẩm trong catalog theo nhóm"),
                              (b, [rows[r] for r, _ in REGIMES], "(b) Tương tác validation theo nhóm của sản phẩm đích")):
        bars = ax.bar([label for _, label in REGIMES], values, color=BLUE, width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, vn(v), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_ylim(0, max(values) * 1.18)
        ax.yaxis.set_major_formatter(lambda v, _: vn(v / 1e3, 0) + "k")
        ax.set_title(title, loc="left")
        grid(ax)
    finish(fig, "data_regimes.png", "Nhóm tính theo số tương tác của sản phẩm trong tập train. Nguồn: catalog_report.json, split_manifest.json")


def training_figure() -> None:
    history = load(HISTORY)["epochs"]
    epochs = [e["epoch"] for e in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, label, color in (("total", "Tổng loss", BLUE), ("bpr", "Thành phần xếp hạng", ORANGE)):
        values = [e["train_loss"][key] for e in history]
        a.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=4, label=label)
    a.set_xlim(0.5, 12.5)
    a.set_title("(a) Loss huấn luyện trung bình mỗi epoch", loc="left")
    a.set_xlabel("Epoch")
    a.legend(loc="upper right")
    grid(a)
    cold = [e["validation_v1"]["cold_macro_ndcg@10"] for e in history]
    b.plot(epochs, cold, color=BLUE, linewidth=2, marker="o", markersize=4)
    best = max(range(len(cold)), key=cold.__getitem__)
    b.plot([epochs[best]], [cold[best]], marker="o", markersize=9, markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=2)
    b.annotate(f"Epoch {epochs[best]} được chọn: {vn(cold[best], 4)}", (epochs[best], cold[best]),
               xytext=(1.0, cold[best] - 0.0005), fontsize=8.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    b.axvline(epochs[-1], color=MUTED, linewidth=0.8, linestyle="--")
    b.text(epochs[-1] - 0.15, min(cold) + 0.0005, "Dừng sớm\n(patience 3)", ha="right", fontsize=8, color=MUTED)
    b.set_title("(b) Cold macro NDCG@10 trên validation v1", loc="left")
    b.set_xlabel("Epoch")
    grid(b)
    finish(fig, "training_curve.png",
           "Một seed (42), 198.232 bước. Validation v1 là giao thức cũ chỉ dùng để chọn checkpoint. Nguồn: ufm_full_v1_training_history.json")


def regime_figures() -> None:
    models = load(EVIDENCE)["models"]
    by_name = {m["name"]: m["metrics"] for m in models.values()}
    names = ["TF-IDF", "Semantic-only", "UFM full", "Popularity"]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    width = 0.19
    for i, name in enumerate(names):
        values = [by_name[name]["by_regime"][r]["ndcg@10"] for r, _ in REGIMES]
        xs = [k + (i - 1.5) * (width + 0.012) for k in range(len(REGIMES))]
        ax.bar(xs, values, width=width, color=ENTITY[name], label=name)
    ax.set_xticks(range(len(REGIMES)), [label for _, label in REGIMES])
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    grid(ax)
    finish(fig, "regime_ndcg_v3.png", "Validation v3: 10.000 ca mỗi nhóm, 100 ứng viên mỗi ca, seed 42. Giá trị chính xác ở Bảng 5.3. Nguồn: evidence.json")

    order = sorted(by_name, key=lambda n: by_name[n]["known_user"]["ndcg@10"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    values = [by_name[n]["known_user"]["ndcg@10"] for n in order]
    # Only the four named models keep their entity colour; ablation variants stay neutral grey.
    bars = ax.barh(order, values, color=[ENTITY.get(n, "#b4b2ab") for n in order], height=0.6)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, " " + vn(v, 4), va="center", fontsize=8.5, color=INK)
    fallback = by_name["UFM full"]["empty_history"]["ndcg@10"]
    ax.axvline(fallback, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylim(-0.6, len(order) + 0.5)
    ax.text(fallback, len(order) - 0.1, f"  Ca không có lịch sử (mọi mô hình dùng popularity): {vn(fallback, 4)}",
            fontsize=8, color=MUTED, va="center")
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("NDCG@10 trên 20.601 ca có lịch sử")
    grid(ax, "x")
    finish(fig, "history_split_v3.png", "Validation v3; 19.399 ca không có lịch sử cho cùng điểm ở mọi mô hình. Nguồn: evidence.json")

    beauty = load(BEAUTY / "models/content_baseline/metrics.json")["splits"]["validation"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    width = 0.32
    groups = [(key, label) for key, label in REGIMES]
    for i, (name, key) in enumerate((("Popularity", "popularity"), ("TF-IDF", "tfidf_profile"))):
        values = [beauty[key]["by_regime"][r]["ndcg@10"] for r, _ in groups]
        xs = [k + (i - 0.5) * (width + 0.02) for k in range(len(groups))]
        bars = ax.bar(xs, values, width=width, color=ENTITY[name], label=name)
        bar_labels(ax, bars, values, lambda v: vn(v, 3))
    ax.set_xticks(range(len(groups)), [label for _, label in groups])
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper left")
    grid(ax)
    finish(fig, "all_beauty_baseline.png", "All Beauty, validation, 100 ứng viên/ca lấy đều; cold có 7.381 ca. Không so trực tiếp với Toys. Nguồn: metrics.json")


def box(ax, x, y, w, h, title, lines, fill="#eef3f9", edge="#5a7da8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.78, "\n".join(lines), ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.45)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#5f6f82",
                                 linewidth=1.3, linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


def diagram_figures() -> None:
    split = load(TOYS / "split_manifest.json")
    fig, ax = plt.subplots(figsize=(11, 8.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    w, h, xs, ys = 3.5, 1.75, (0.2, 4.25, 8.3), (7.6, 5.35, 3.1, 0.85)
    rows = [
        [("Amazon Reviews 2023", ["Toys & Games: đánh giá + metadata", f"{vn(split['raw']['raw_records'])} bản ghi thô"]),
         ("Lọc và khử trùng", ["Mua đã xác minh, rating ≥ 4", f"Còn {vn(split['retained_unique_records'])} tương tác"]),
         ("Chia thời gian 80/10/10", ["Ranh giới ngày UTC", "Validation từ 11/08/2021", "Test từ 16/07/2022"])],
        [("Lịch sử chỉ dùng quá khứ", ["Sự kiện trước đích, giữ 50 gần nhất", f"{vn(split['history_rows']['train'])} dòng train có lịch sử"]),
         ("Catalog 767.045 sản phẩm", ["Tên/mô tả + URL ảnh chính", "TF-IDF 20.000 đặc trưng"]),
         ("Bộ ứng viên validation v3", ["40.000 ca, 10.000 ca mỗi nhóm", "1 dương + 99 âm mỗi ca"])],
        [("CLIP ViT-B/32 đóng băng", ["767.044 vector văn bản", "758.692 vector ảnh (98,91%)"]),
         ("Huấn luyện UFM-Rec", ["GPU FITLAB, AMP, 12 epoch", "Chọn epoch 9 theo validation v1"]),
         ("7 biến thể ablation", ["Một seed 42, cùng ngân sách", "Đánh giá cùng candidate hash"])],
        [("Đánh giá và hiệu chuẩn", ["NDCG/Recall/MRR@10 theo nhóm", "Temperature scaling thăm dò"]),
         ("Demo UFM toàn catalog", ["API + giao diện web", "6/6 ca API, 4/4 ca giao diện"]),
         ("Báo cáo và bàn giao", ["Bằng chứng có SHA-256", "120 test tự động đạt"])],
    ]
    for r, row in enumerate(rows):
        for c, (title, lines) in enumerate(row):
            box(ax, xs[c], ys[r], w, h, title, lines)
    for r in range(4):
        for c in range(2):
            arrow(ax, (xs[c] + w, ys[r] + h / 2), (xs[c + 1], ys[r] + h / 2))
    for r in range(3):
        for c in range(3):
            arrow(ax, (xs[c] + w / 2, ys[r]), (xs[c] + w / 2, ys[r + 1] + h))
    finish(fig, "pipeline.png", "Mũi tên ngang: thứ tự trong một giai đoạn; mũi tên dọc: đầu ra dùng cho giai đoạn sau.")

    fig, ax = plt.subplots(figsize=(11, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.3)
    ax.axis("off")
    left, right, w = 0.3, 6.3, 5.4
    box(ax, left, 7.55, w, 1.45, "Đầu vào hành vi", ["ID lịch sử + ID ứng viên", "Số tương tác train, mặt nạ ID chưa biết"])
    box(ax, right, 7.55, w, 1.45, "Đầu vào nội dung", ["Văn bản 512 chiều + ảnh 512 chiều từ CLIP", "Encoder đóng băng, mặt nạ phương thức"])
    box(ax, left, 5.45, w, 1.7, "Nhánh hành vi", ["ID + vị trí → Transformer nhân quả", "2 lớp, 4 head, d = 128", "Vector cặp người dùng–ứng viên"])
    box(ax, right, 5.45, w, 1.7, "Nhánh ngữ nghĩa", ["Adapter → d = 128, chuẩn hoá L2", "Hồ sơ = trung bình lịch sử khả dụng", "Thiếu phương thức: mặt nạ, không thêm nhiễu"])
    box(ax, left, 3.7, w, 1.35, "Vector hành vi + u_cf", ["MLP + Softplus; cờ khả dụng"])
    box(ax, right, 3.7, w, 1.35, "Vector ngữ nghĩa + u_sem", ["MLP + Softplus; cờ khả dụng"])
    box(ax, left, 1.95, w, 1.35, "Trọng số độ tin cậy", ["w = softmax(−u) trên nhánh khả dụng"])
    box(ax, right, 1.95, w, 1.35, "Căn chỉnh chéo có cổng", ["Chỉ bật khi cả hai nhánh khả dụng"])
    box(ax, 2.6, 0.2, 5.0, 1.35, "Hợp nhất và chấm điểm", ["h = w_cf·h_cf + w_sem·h_sem + g_align → MLP"])
    box(ax, 8.3, 0.2, 3.4, 1.35, "Không có tín hiệu", ["Popularity của tập train"], fill="#f1f1ef", edge="#8a8a86")
    for x in (left + w / 2, right + w / 2):
        arrow(ax, (x, 7.55), (x, 7.15))
        arrow(ax, (x, 5.45), (x, 5.05))
        arrow(ax, (x, 3.7), (x, 3.3))
    arrow(ax, (left + w, 4.35), (right + 0.6, 3.3))
    arrow(ax, (right, 4.35), (left + w - 0.6, 3.3))
    arrow(ax, (left + w / 2, 1.95), (4.0, 1.55))
    arrow(ax, (right + w / 2, 1.95), (6.4, 1.55))
    arrow(ax, (8.3, 0.87), (7.6, 0.87))
    finish(fig, "architecture.png", "u là proxy độ tin cậy học được (Softplus), không phải hậu nghiệm Bayes. Theo src/ufm_model.py.")


def demo_figures() -> None:
    crops = {"demo_overview.png": ("demo_ui_zero_shot_20261029.png", (0, 0, 1403, 1480)),
             "demo_offline.png": ("demo_ui_offline_20261029.png", (0, 640, 1403, 1880))}
    for name, (source, area) in crops.items():
        Image.open(SHOTS / source).convert("RGB").crop(area).save(OUT / name, optimize=True)
        print(OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_figures()
    training_figure()
    regime_figures()
    diagram_figures()
    demo_figures()


if __name__ == "__main__":
    main()
