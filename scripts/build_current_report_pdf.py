"""Render the current Markdown report to PDF with embedded fonts and equations."""
from __future__ import annotations

import hashlib
import html
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import math_to_image
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/final_report.md"
OUT = ROOT / "output/history/pdf/UFM_Rec_Bao_Cao_Hien_Trang_20261029.pdf"
TMP = ROOT / "tmp/current_report_pdf"
WIDTH = A4[0] - 104


def font_setup():
    choices = [(Path("C:/Windows/Fonts/times.ttf"), Path("C:/Windows/Fonts/timesbd.ttf")),
               (Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"))]
    for normal, bold in choices:
        if normal.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Report", str(normal)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold",
                                          italic="Report", boldItalic="ReportBold")
            return
    raise FileNotFoundError("A Vietnamese-capable report font is required")


def equation(tex, display=False):
    tex = tex.replace(r"\text{", r"\mathrm{")
    tex = re.sub(r"\\(le|ge|ne)(?![a-zA-Z])", lambda m: "\\" + m[1] + "q", tex)
    path = TMP / (hashlib.sha256(tex.encode()).hexdigest()[:20] + ".png")
    if not path.exists():
        math_to_image("$" + tex + "$", str(path), dpi=200, format="png", color="black")
    with PILImage.open(path) as im:
        width, height = im.size
    width, height = width * 72 / 200, height * 72 / 200
    factor = min(1, WIDTH / width) if display else min(1, 18 / height)
    return path, width * factor, height * factor


def inline(text):
    expressions = []
    def math_replace(match):
        path, width, height = equation(match[1])
        expressions.append(f'<img src="{path.as_posix()}" width="{width:.2f}" height="{height:.2f}" valign="middle"/>')
        return f"EQUATIONTOKEN{len(expressions)-1}ENDTOKEN"
    text = re.sub(r"\$([^$]+)\$", math_replace, text)
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: f'<a href="{m[2]}" color="#284b63">{m[1]}</a>' if '://' in m[2] else m[1], text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")
    for i, expression in enumerate(expressions):
        text = text.replace(f"EQUATIONTOKEN{i}ENDTOKEN", expression)
    return text


def build():
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    font_setup()
    body = ParagraphStyle("body", fontName="Report", fontSize=11, leading=15.6,
                          spaceAfter=7, textColor=colors.black)
    heading = ParagraphStyle("heading", parent=body, fontName="ReportBold", fontSize=15,
                             leading=19, spaceBefore=12, spaceAfter=10, keepWithNext=True)
    subheading = ParagraphStyle("subheading", parent=heading, fontSize=12.5, leading=16,
                                spaceBefore=10, spaceAfter=7)
    caption = ParagraphStyle("caption", parent=body, fontSize=9, leading=12, spaceAfter=10,
                            textColor=colors.HexColor("#454545"))
    table_style = ParagraphStyle("table", parent=body, fontSize=9.5, leading=12.5, spaceAfter=0)
    title = ParagraphStyle("title", parent=heading, fontSize=23, leading=29, spaceAfter=20)
    story = [Spacer(1, 35), Paragraph("Báo cáo đồ án UFM Rec", title),
             Paragraph("Hệ gợi ý sản phẩm từ hành vi và nội dung đa phương thức", heading),
             Spacer(1, 16), Paragraph("Học phần Học máy ứng dụng", body),
             Paragraph("Giảng viên hướng dẫn Phan Hồ Viết Trường", body),
             Paragraph("Nhóm thực hiện Nguyễn Thành Phong, Trần An Kỳ, Nguyễn Hoàng Hải, Hân", body),
             Spacer(1, 24), Paragraph("Bản hiện trạng ngày 29 tháng 10 năm 2026", subheading),
             Paragraph("Hệ thống đã có checkpoint, đối chứng một seed và demo UFM thật. "
                       "Trên validation v3, full UFM chưa vượt TF-IDF hoặc semantic-only ở Cold macro NDCG@10. "
                       "Bản này tổng hợp bằng chứng đã lưu và nghiệm thu suy luận; chưa có kết quả train mới.", body),
             Paragraph("Phạm vi kết luận: đánh giá thăm dò trên 40.000 ca và 100 ứng viên mỗi ca. "
                       "Test v2 đã được xem; validation v3 không phải holdout độc lập.", body),
             Spacer(1, 18), Paragraph("Thông tin thành viên đầy đủ và hình thức theo mẫu trường cần được nhóm đối chiếu trước bản nộp chính thức.", caption),
             PageBreak()]
    raw = SOURCE.read_text(encoding="utf-8")
    raw = raw[raw.index("### TÓM TẮT"):]
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---" or line.startswith("<!--"):
            i += 1; continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("# ")
            if text.startswith("CHƯƠNG 1:"):
                story.append(PageBreak())
            story.append(Paragraph(inline(text), heading if level <= 3 else subheading))
            i += 1; continue
        if line.startswith("$$"):
            tex = line.strip("$")
            path, width, height = equation(tex, display=True)
            story.append(Image(str(path), width=width, height=height, hAlign="CENTER"))
            story.append(Spacer(1, 12)); i += 1; continue
        match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if match:
            path = (SOURCE.parent / match[2]).resolve()
            with PILImage.open(path) as im:
                width, height = im.size
            display_width = min(WIDTH, width * 0.25)
            picture = Image(str(path), width=display_width, height=height/width*display_width)
            story.append(KeepTogether([picture, Spacer(1, 6), Paragraph(inline(match[1]), caption)]))
            i += 1; continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", x) for x in cells):
                    rows.append([Paragraph(inline(x), table_style) for x in cells])
                i += 1
            columns = len(rows[0])
            first = 128 if columns >= 5 else 173
            widths = [first] + [(WIDTH-first)/(columns-1)]*(columns-1)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5ebee")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7f8")]),
                ("GRID", (0,0), (-1,-1), 0.45, colors.HexColor("#d9d9d9")),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
            ]))
            story.extend([table, Spacer(1, 10)]); continue
        text = line
        i += 1
        if not re.match(r"^(?:[-*]|\d+\.) ", text):
            while i < len(lines) and lines[i].strip() and not re.match(r"^(?:#|\||!\[|\$\$|[-*] |\d+\. )",lines[i].strip()):
                text += " " + lines[i].strip(); i += 1
        if text.startswith("- "): text = "• " + text[2:]
        story.append(Paragraph(inline(text), body))

    def footer(canvas, doc):
        canvas.setTitle("Báo cáo đồ án UFM Rec ngày 29 tháng 10 năm 2026")
        canvas.setAuthor("Nhóm UFM Rec")
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.HexColor("#454545"))
        canvas.drawString(52, 28, "UFM Rec | Bản hiện trạng 29/10/2026 | Validation v3 thăm dò")
        canvas.drawRightString(A4[0]-52, 28, str(doc.page))
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=52, rightMargin=52,
                      topMargin=43, bottomMargin=48).build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
