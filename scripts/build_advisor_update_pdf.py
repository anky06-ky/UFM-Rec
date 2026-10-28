"""Build a short, evidence-backed Vietnamese progress PDF for the advisor."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/continuation_20261028"
OUTPUT = ROOT / "output/pdf/UFM_Rec_Tien_Do_2026_10_28.pdf"


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def main() -> None:
    summary = load("summary.json")
    count = load("count_effect_v3.json")
    demo = load("windows_ufm_verification.json")
    modality = load("windows_ufm_missing_modalities.json")
    benchmark = load("windows_ufm_benchmark.json")
    protocol = json.loads((ROOT / "configs/multimodal_followup_20261028.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 40000 and summary["candidates_per_case"] == 100
    assert summary["candidate_sha256"] == count["sources_sha256"]["samples"]
    assert demo["phase"] == "complete" and len(demo["cases"]) == 6
    assert demo["status"]["model"]["checkpoint_sha256"] == benchmark["status"]["model"]["checkpoint_sha256"]
    assert modality["checkpoint_sha256"] == demo["status"]["model"]["checkpoint_sha256"]
    assert protocol["status"] == "protocol_locked_no_new_training_results"

    font = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if not font.is_file() or not bold.is_file():
        raise FileNotFoundError("Arial TTF needed for Vietnamese PDF")
    pdfmetrics.registerFont(TTFont("ArialVN", str(font)))
    pdfmetrics.registerFont(TTFont("ArialVNBold", str(bold)))
    pdfmetrics.registerFontFamily("ArialVN", normal="ArialVN", bold="ArialVNBold")
    style = ParagraphStyle("body", fontName="ArialVN", fontSize=9.4, leading=13.6,
                           spaceAfter=5, alignment=TA_LEFT, textColor=colors.HexColor("#20252b"))
    title = ParagraphStyle("title", parent=style, fontName="ArialVNBold", fontSize=17, leading=22,
                           spaceAfter=8, textColor=colors.black)
    subtitle = ParagraphStyle("subtitle", parent=style, fontName="ArialVNBold", fontSize=10.5,
                              leading=15, spaceBefore=10, spaceAfter=5, textColor=colors.black)
    small = ParagraphStyle("small", parent=style, fontSize=8.2, leading=11.2, textColor=colors.HexColor("#444b55"))
    cell = ParagraphStyle("cell", parent=style, fontSize=8.8, leading=12, spaceAfter=0)
    header_cell = ParagraphStyle("headercell", parent=cell, fontName="ArialVNBold", textColor=colors.white)
    story = []

    def p(text: str, kind=style) -> None:
        story.append(Paragraph(text, kind))

    def section(text: str) -> None:
        p(text, subtitle)

    def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        cells = [[Paragraph(x, header_cell) for x in headers]]
        cells += [[Paragraph(str(x), cell) for x in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#213c57")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9dfe5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 5 * mm))

    p("Báo cáo tiến độ hệ khuyến nghị sản phẩm UFM Rec", title)
    p("Cập nhật 28/10/2026 · Tài liệu trao đổi với giảng viên hướng dẫn", small)
    p("Theo xác nhận của nhóm, phạm vi đã chốt là hệ gợi ý sản phẩm kết hợp <b>lịch sử hành vi, hình ảnh và tên/mô tả</b>. Nhóm dùng Amazon Reviews 2023 Toys &amp; Games để thực nghiệm, tập trung phân tích sản phẩm ít hoặc chưa có tương tác. Người dùng không có lịch sử hiện chỉ nhận fallback; nhóm chưa chứng minh cá nhân hóa người dùng mới.")

    section("1. Kết quả có thể trình bày")
    p("Validation v3 gồm 40.000 ca, mỗi ca 1 sản phẩm đúng và 99 ứng viên âm; các mô hình trong bảng dùng cùng candidate SHA-256. Các checkpoint UFM hiện có <b>một seed 42</b> và từng được chọn trên validation v1. V3 dùng lại quần thể validation nên <b>chưa phải holdout độc lập</b>; test v2 đã được xem trong lịch sử.")
    scores = summary["cold_macro_ndcg10"]
    table(["Mô hình", "Cold macro NDCG@10", "Diễn giải"], [
        ["TF-IDF profile", f"{scores['tfidf_profile']:.6f}", "Mốc hệ thống"],
        ["CLIP semantic-only", f"{scores['ufm_ablation_semantic_only_s42_v1']:.6f}", "Đối chứng cùng CLIP"],
        ["UFM full", f"{scores['ufm_full_v1']:.6f}", "Dung hợp hành vi và nội dung"],
    ], [155, 145, 170])
    p("UFM full chưa vượt semantic-only hoặc TF-IDF theo chỉ số này. Không kết luận cơ chế dung hợp đã cải thiện cold-start.")

    section("2. Chênh lệch theo số tương tác train của sản phẩm đúng")
    table(["Train count", "Số ca", "Full − semantic-only NDCG@10"], [
        [label, str(count["by_positive_train_count"][label]["cases"]),
         f"{count['by_positive_train_count'][label]['paired_full_minus_semantic']:+.6f}"]
        for label in ("0", "1-5", "6-20", "21+")
    ], [145, 95, 230])
    p("Chênh lệch bất lợi lớn nhất là nhóm 1–5 tương tác; full nhỉnh hơn ở nhóm 21+. Đây là phân tích cặp trên dự đoán đã lưu, không chứng minh nhánh ID hay gate là nguyên nhân. Nhóm đã tách biến thể <b>count-aware fusion</b> khỏi <b>adaptive dropout</b> trong mã để đánh giá riêng.")

    story.append(PageBreak())
    section("3. Trạng thái thực nghiệm tiếp theo")
    p("Giao thức đã khóa <b>15 lượt train</b>: full, semantic-only, fixed-fusion, count-aware và adaptive dropout, mỗi cấu hình ở seed 42, 7, 2026; cùng CLIP, dữ liệu, ngân sách và cách chọn checkpoint. Dry-run trên bản sao FITLAB kiểm tra 9 đầu vào bằng SHA. <b>Chưa có lượt train mới</b>: máy Windows không có CUDA và phiên FITLAB chưa truy cập được trong lượt này. Chỉ sau khi có GPU mới báo điểm từng seed cùng trung bình ± độ lệch chuẩn mẫu.")

    section("4. Demo và giới hạn vận hành")
    p("Checkpoint UFM thật từ bản sao đã khôi phục chạy trên Windows với <b>767.045 sản phẩm</b> và train count thật; 6/6 ca API đạt. Item chỉ có text hoặc chỉ có ảnh vẫn tìm kiếm và dùng làm lịch sử để trả Top 3; sản phẩm đã chọn bị loại. Cache thực không có item thiếu đồng thời cả hai tín hiệu. JPEG hiển thị có thể lỗi nhưng mô hình dùng CLIP embedding đã lưu.")
    latency = benchmark["latency_seconds"]
    rss = benchmark["server_rss_mib"]["observed_max"]
    p(f"Benchmark CPU Windows trên toàn catalog, một item lịch sử, K=5: 2 warm-up và 10 request đo; p50 <b>{latency['p50']:.2f} giây</b>, p95 <b>{latency['p95']:.2f} giây</b>; RSS server cao nhất quan sát <b>{rss:,.1f} MiB</b>. Hai request đồng thời cho một HTTP 200 và một HTTP 429 vì server chỉ cho một lượt suy luận. Đây là giới hạn rõ của demo hiện tại, chưa phải benchmark phục vụ nhiều người dùng.")

    section("5. Nội dung xin góp ý")
    p("(1) Trong phạm vi ba nguồn tín hiệu đã chốt, thầy có cần kết quả UFM trên All Beauty hay Toys &amp; Games đủ làm tập chính? (2) Ngoài Cold macro NDCG@10, thầy muốn xem chỉ số nghiệm thu nào? (3) Với test v2 đã được xem và negative sampling hồi cứu, cách đánh giá tiếp theo nào phù hợp để kết luận đóng góp?")
    p("Nguồn đối chiếu: reports/continuation_20261028/summary.json, count_effect_v3.json, windows_ufm_verification.json, windows_ufm_missing_modalities.json, windows_ufm_benchmark.json; configs/multimodal_followup_20261028.json. Word/PDF báo cáo toàn văn cũ chưa đồng bộ, không dùng làm bản nộp cuối.", small)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=17 * mm,
                                 rightMargin=17 * mm, topMargin=15 * mm, bottomMargin=16 * mm,
                                 title="Báo cáo tiến độ hệ khuyến nghị sản phẩm UFM Rec",
                                 author="Nhóm UFM Rec")

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("ArialVN", 8)
        canvas.setFillColor(colors.HexColor("#565e66"))
        canvas.drawString(17 * mm, 11 * mm, "UFM Rec · Báo cáo tiến độ 28/10/2026")
        canvas.drawRightString(A4[0] - 17 * mm, 11 * mm, str(doc.page))
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
