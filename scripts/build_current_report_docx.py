"""Build an editable current-state Word report from docs/final_report.md.

Uses a verified Pandoc executable for Markdown and native Word equations, then
the bundled python-docx runtime for report styles. It does not train a model.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/final_report.md"
BUILD = ROOT / "tmp/current_docx_20261029"
OUTPUT = ROOT / "output/history/docx/UFM_Rec_Bao_Cao_Hien_Trang_20261029.docx"
DEFAULT_PANDOC = Path(
    r"D:\Nam3_ki1\UFM-Rec_Backups\colab_training_inputs_20261029"
    r"\document_runtime\pandoc_extracted\Pandoc\pandoc.exe"
)


def report_markdown() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    abstract_start = next(i for i, s in enumerate(lines) if s.startswith("### TÓM TẮT"))
    toc_start = next(i for i, s in enumerate(lines) if s.startswith("### MỤC LỤC"))
    chapter_start = next(i for i, s in enumerate(lines) if s.startswith("### CHƯƠNG 1"))
    selected = lines[abstract_start:toc_start] + lines[chapter_start:]
    out = [
        "---",
        'title: "Hệ khuyến nghị sản phẩm cold start đa phương thức"',
        'subtitle: "Báo cáo đồ án UFM Rec"',
        'author: "Nguyễn Thành Phong, Trần An Kỳ, Nguyễn Hoàng Hải, Hân"',
        'date: "29/10/2026"',
        "---",
        "",
        "Học phần: Học máy ứng dụng. Giảng viên hướng dẫn: Phan Hồ Viết Trường.",
        "",
        "Bản hiện trạng trước vòng huấn luyện tiếp theo; các số liệu chính là validation v3. "
        "Bản nộp theo mẫu trường được dựng bằng scripts/build_school_report_docx.py.",
        "",
        "# Mục lục",
        "",
        "1. Chương 1 Giới thiệu và đặt vấn đề",
        "2. Chương 2 Kiến trúc mô hình UFM Rec",
        "3. Chương 3 Thiết lập thực nghiệm và giao thức đánh giá",
        "4. Chương 4 Kết quả thực nghiệm tài nguyên và giới hạn",
        "5. Chương 5 Các đề xuất và hướng mở rộng",
        "6. Chương 6 Kết luận",
        "7. Tài liệu tham khảo",
        "",
    ]
    for line in selected:
        if line.strip() == "---":
            continue
        if line.startswith("$$w_k = \\frac"):
            out += [
                r"$$w_k = \frac{\mathbb{I}(k\in A)\exp(-u_k)}{\sum_{j\in A}\exp(-u_j)}$$",
                "Trong đó A là tập các nhánh khả dụng và A không rỗng.",
            ]
            continue
        if line.startswith("### CHƯƠNG ") or line.startswith("### TÀI LIỆU"):
            line = "# " + line[4:]
        elif line.startswith("### TÓM TẮT"):
            line = "# " + line[4:]
        elif line.startswith("#### "):
            line = "## " + line[5:]
        line = re.sub(
            r"(!\[[^\]]*\]\()\.\./(output/[^)]+)(\))",
            lambda m: m.group(1) + (ROOT / m.group(2)).as_posix() + m.group(3),
            line,
        )
        out.append(line)
    return "\n".join(out) + "\n"


def color_style(style, size: float, *, bold: bool = False, before: int = 0, after: int = 5):
    style.font.name = "Arial"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)
    style.paragraph_format.line_spacing = 1.14


def set_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_borders(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for side in ("top", "left", "bottom", "right"):
        edge = borders.find(qn("w:" + side))
        if edge is None:
            edge = OxmlElement("w:" + side)
            borders.append(edge)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "5")
        edge.set(qn("w:color"), "D9D9D9")


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    cached = OxmlElement("w:t")
    cached.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, cached, end):
        run._r.append(element)


def style_docx() -> None:
    doc = Document(OUTPUT)
    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(0.82)
        section.bottom_margin = Inches(0.78)
        section.left_margin = Inches(0.82)
        section.right_margin = Inches(0.82)
        section.header_distance = Inches(0.4)
        section.footer_distance = Inches(0.4)
        section.different_first_page_header_footer = True
        header = section.header.paragraphs[0]
        header.text = "UFM Rec  |  Báo cáo đồ án hiện trạng"
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in header.runs:
            run.font.name = "Arial"
            run.font.size = Pt(8)
            run.font.color.rgb = RGBColor(80, 80, 80)
        add_page_number(section.footer.paragraphs[0])
        add_page_number(section.first_page_footer.paragraphs[0])

    for name, size, bold, before, after in [
        ("Normal", 11, False, 0, 6),
        ("Title", 20, True, 0, 15),
        ("Subtitle", 12, False, 0, 9),
        ("Heading 1", 15, True, 15, 7),
        ("Heading 2", 12.5, True, 11, 5),
        ("Heading 3", 11.5, True, 9, 4),
        ("Caption", 9, False, 4, 8),
    ]:
        if name in doc.styles:
            color_style(doc.styles[name], size, bold=bold, before=before, after=after)
    for name in ("Table Contents", "Table Header"):
        if name in doc.styles:
            color_style(doc.styles[name], 9.5, bold=name == "Table Header", after=0)
    for p in doc.paragraphs:
        if p.text.strip() == "29/10/2026":
            p.add_run().add_break(WD_BREAK.PAGE)
        if p.text.strip() == "Vector tổng hợp cuối cùng:":
            p.paragraph_format.keep_with_next = True
        if p.style.name == "Normal":
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if p.style.name.startswith("Heading"):
            p.paragraph_format.keep_with_next = True
    for table in doc.tables:
        table.autofit = True
        for i, row in enumerate(table.rows):
            if i == 0:
                tr_pr = row._tr.get_or_add_trPr()
                head = OxmlElement("w:tblHeader")
                head.set(qn("w:val"), "true")
                tr_pr.append(head)
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                set_borders(cell)
                set_shading(cell, "17384A" if i == 0 else ("F2F6F8" if i % 2 == 0 else "FFFFFF"))
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(2)
                    for run in p.runs:
                        run.font.name = "Arial"
                        run.font.size = Pt(9.5)
                        if i == 0:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
    doc.core_properties.title = "Hệ khuyến nghị sản phẩm cold start đa phương thức"
    doc.core_properties.subject = "Báo cáo đồ án UFM Rec hiện trạng trước vòng huấn luyện tiếp theo"
    doc.save(OUTPUT)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prepared = BUILD / "report_source.md"
    prepared.write_text(report_markdown(), encoding="utf-8")
    pandoc = Path(os.environ.get("UFM_PANDOC", str(DEFAULT_PANDOC)))
    if not pandoc.is_file():
        raise FileNotFoundError(f"Pandoc not found: {pandoc}")
    cmd = [str(pandoc), str(prepared), "-f", "gfm+tex_math_dollars", "-t", "docx",
           "--standalone", "--resource-path", str(ROOT),
           "-o", str(OUTPUT)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    style_docx()
    print(OUTPUT)


if __name__ == "__main__":
    main()
