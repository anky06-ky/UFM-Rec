"""Build the submission report in the Van Lang University template.

Content comes from docs/final_report.md (validation v3 evidence). Pandoc turns the
Markdown into Word; python-docx restyles it to the team's template conventions
(Times New Roman 13, A4, 3 cm left margin, navy headings), prepends the template
cover with its logos and page border, and adds the acknowledgements, task table,
contents and abbreviation pages. A portable LibreOffice fills the contents and
exports the PDF. Nothing is trained.

Edit ADVISOR and TEAM when the cover details change, then run
    python scripts/build_school_report_docx.py
"""
from __future__ import annotations

import copy
import io
import json
import os
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from build_current_report_docx import add_page_number, set_shading

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/final_report.md"
# Only the first 23 blocks (the Van Lang cover and its section break) are reused from this file.
TEMPLATE = Path(os.environ.get(
    "UFM_REPORT_TEMPLATE", ROOT / "output/history/docx/Bao_Cao_Do_An_UFM_Rec_Nhom_4_20261027.docx"))
RUNTIME = Path(os.environ.get(
    "UFM_DOC_RUNTIME", r"D:\Nam3_ki1\UFM-Rec_Backups\colab_training_inputs_20261029\document_runtime"))
PANDOC = Path(os.environ.get("UFM_PANDOC", RUNTIME / "pandoc_extracted/Pandoc/pandoc.exe"))
LO_PYTHON = RUNTIME / "lo_extracted/program/python.exe"
BUILD = ROOT / "tmp/school_report"
OUTPUT_DOCX = ROOT / "output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx"
OUTPUT_PDF = ROOT / "output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf"

ADVISOR = "Phan Hồ Viết Trường"
# (họ tên, MSSV, nhiệm vụ, mức hoàn thành). Điền thông tin của Hân trước khi nộp.
TEAM = [
    ("Nguyễn Thành Phong", "2474802010304",
     "Trưởng nhóm: thiết kế kiến trúc UFM-Rec, lập trình PyTorch (Causal Transformer, cổng fusion, "
     "reliability proxy), tích hợp đặc trưng CLIP ViT-B/32, huấn luyện trên GPU FITLAB, đánh giá định "
     "lượng và tổng hợp báo cáo kỹ thuật.", "100%"),
    ("Trần An Kỳ", "2474802010460",
     "Xây dựng pipeline dữ liệu Amazon Reviews 2023 (Toys & Games: 767.045 sản phẩm, 11,57 triệu tương "
     "tác), chia 80/10/10 theo thời gian, catalog Parquet, hệ thống demo tương tác và chế độ dự phòng "
     "offline.", "100%"),
    ("Nguyễn Hoàng Hải", "2474802010099",
     "Hiện thực các mô hình đối chứng (TF-IDF, popularity), hiệu chuẩn temperature scaling (ECE, NLL), "
     "đo tài nguyên (RAM, VRAM, độ trễ) và kiểm thử tự động.", "100%"),
    ("[Họ tên đầy đủ của Hân]", "[MSSV]", "[Bổ sung nhiệm vụ]", "[...]"),
]
ACKNOWLEDGEMENTS = [
    "Nhóm xin chân thành cảm ơn Ban Giám hiệu Trường Đại học Văn Lang và Khoa Công nghệ Thông tin đã "
    "tạo điều kiện về cơ sở vật chất, môi trường học tập và hạ tầng tính toán FITLAB trong quá trình "
    "thực hiện đề tài.",
    f"Nhóm xin gửi lời cảm ơn đặc biệt đến giảng viên hướng dẫn {ADVISOR}, người đã định hướng đề tài "
    "và góp ý về phương pháp thực nghiệm, cách đánh giá và cách trình bày kết quả, giúp nhóm hoàn "
    "thiện đồ án UFM-Rec.",
    "Nhóm cũng cảm ơn quý Thầy Cô trong Khoa Công nghệ Thông tin đã truyền đạt kiến thức nền tảng về "
    "học máy, học sâu, xử lý ngôn ngữ tự nhiên và thị giác máy tính, cùng gia đình và bạn bè đã động "
    "viên nhóm trong suốt quá trình thực hiện.",
    "Do thời gian và tài nguyên tính toán có hạn, báo cáo khó tránh khỏi thiếu sót. Nhóm rất mong nhận "
    "được góp ý của quý Thầy Cô để đề tài được hoàn thiện hơn.",
]
ABBREVIATIONS = [
    ("ANN", "Approximate Nearest Neighbor – tìm kiếm láng giềng gần đúng"),
    ("API", "Application Programming Interface – giao diện lập trình ứng dụng"),
    ("BERT4Rec", "Mô hình gợi ý tuần tự dùng Transformer hai chiều"),
    ("BPR", "Bayesian Personalized Ranking"),
    ("CF", "Collaborative Filtering – lọc cộng tác"),
    ("CLIP", "Contrastive Language–Image Pre-training"),
    ("ECE", "Expected Calibration Error – sai số hiệu chuẩn kỳ vọng"),
    ("GPU", "Graphics Processing Unit"),
    ("ID", "Identifier – mã định danh sản phẩm"),
    ("LoRA", "Low-Rank Adaptation"),
    ("MF", "Matrix Factorization – phân rã ma trận"),
    ("MRR", "Mean Reciprocal Rank"),
    ("NDCG", "Normalized Discounted Cumulative Gain"),
    ("NLL", "Negative Log-Likelihood"),
    ("QPS", "Queries Per Second – số truy vấn mỗi giây"),
    ("RSS", "Resident Set Size – bộ nhớ thường trú của tiến trình"),
    ("SASRec", "Self-Attentive Sequential Recommendation"),
    ("SVD", "Singular Value Decomposition"),
    ("TF-IDF", "Term Frequency – Inverse Document Frequency"),
    ("UFM-Rec", "Uncertainty-aware Fusion Model Recommender"),
    ("UI", "User Interface – giao diện người dùng"),
    ("VRAM", "Bộ nhớ của GPU"),
]
TABLE_CAPTIONS = {
    "| Mô hình | Cold macro": "Kết quả chính trên validation v3 (40.000 ca, 100 ứng viên/ca, seed 42)",
    "| Train count |": "NDCG@10 của UFM full và semantic-only theo số tương tác train",
    "| Chỉ số trên phần audit |": "Calibration bằng temperature scaling trên phần audit của validation v3",
    "| Nhóm item |": "Baseline popularity và TF-IDF trên validation All Beauty",
    "| Nhóm | Số ca mỗi split |": "UFM full trên bộ ứng viên evaluation_v2 (kết quả lịch sử)",
}

FONT = "Times New Roman"
NAVY = RGBColor(0x0B, 0x39, 0x63)
TEXT_WIDTH = Cm(21 - 3.0 - 2.01)
MAX_FIGURE_EMU = int(Cm(15))
TABLE_MARK, FIGURE_MARK = "@@TABLE@@", "@@FIGURE@@"
TOC_PLACEHOLDER = "Mục lục được cập nhật khi xuất PDF; trong Word nhấn F9 để cập nhật."
R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
# Namespaces declared on Pandoc's document root; anything else in the cover needs mc:Ignorable.
CORE_NS = {
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "http://schemas.openxmlformats.org/drawingml/2006/main",
    "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "urn:schemas-microsoft-com:office:office", "urn:schemas-microsoft-com:vml",
    "urn:schemas-microsoft-com:office:word", "http://www.w3.org/XML/1998/namespace",
}
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
IMAGE = re.compile(r"^!\[([^\]]*)\]\(\.\./([^)]+)\)\s*$")


def plain_link(match: re.Match) -> str:
    """Repository links mean nothing on paper: keep the text and show the path."""
    text, target = match.groups()
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
        return match.group(0)
    path = re.sub(r"^(\.\./)+", "", target.split("#", 1)[0])
    return f"`{path}`" if text.startswith("`") and text.endswith("`") else f"{text} (`{path}`)"


def report_markdown() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    abstract = next(i for i, s in enumerate(lines) if s.startswith("### TÓM TẮT"))
    contents = next(i for i, s in enumerate(lines) if s.startswith("### MỤC LỤC"))
    chapters = next(i for i, s in enumerate(lines) if s.startswith("### CHƯƠNG 1"))
    selected = lines[abstract:contents] + lines[chapters:]
    out, chapter, figures, tables = [], 0, 0, 0
    for n, line in enumerate(selected):
        if line.strip() == "---":
            continue
        if line.startswith("$$w_k = \\frac"):
            out += [r"$$w_k = \frac{\mathbb{I}(k\in A)\exp(-u_k)}{\sum_{j\in A}\exp(-u_j)}$$",
                    "Trong đó A là tập các nhánh khả dụng và A không rỗng."]
            continue
        heading = re.match(r"### CHƯƠNG (\d+): (.+)", line)
        if heading:
            chapter, figures, tables = int(heading[1]), 0, 0
            line = f"# CHƯƠNG {heading[1]}. {heading[2]}"
        elif line.startswith("### TÓM TẮT"):
            line = "# TÓM TẮT ĐỀ TÀI"
        elif line.startswith("### Tài liệu tham khảo"):
            line = "# TÀI LIỆU THAM KHẢO"
        elif line.startswith("#### "):
            line = "## " + line[5:]
        image = IMAGE.match(line)
        if image:
            figures += 1
            out += [f"![]({(ROOT / image[2]).as_posix()})", "",
                    f"{FIGURE_MARK}Hình {chapter}.{figures}. {image[1]}"]
            continue
        if line.startswith("|") and n + 1 < len(selected) and re.match(r"^\|\s*:?-", selected[n + 1]):
            caption = next((c for key, c in TABLE_CAPTIONS.items() if line.startswith(key)), None)
            if caption is None:
                raise ValueError(f"Thiếu chú thích cho bảng bắt đầu bằng: {line[:60]}")
            tables += 1
            out += ["", f"{TABLE_MARK}Bảng {chapter}.{tables}. {caption}", ""]
        out.append(LINK.sub(plain_link, line))
    return "\n".join(out) + "\n"


def run_pandoc(markdown: str) -> Path:
    source = BUILD / "report_source.md"
    source.write_text(markdown, encoding="utf-8")
    body = BUILD / "body.docx"
    if not PANDOC.is_file():
        raise FileNotFoundError(f"Pandoc not found: {PANDOC} (đặt biến UFM_PANDOC)")
    subprocess.run([str(PANDOC), str(source), "-f", "gfm+tex_math_dollars", "-t", "docx",
                    "--resource-path", str(ROOT), "-o", str(body)], check=True, cwd=ROOT)
    return body


def use_times(doc) -> None:
    """Replace theme fonts in every style so the whole document uses Times New Roman."""
    styles = doc.styles.element
    defaults = styles.find(qn("w:docDefaults"))
    rpr_default = defaults.find(qn("w:rPrDefault")) if defaults is not None else None
    if rpr_default is not None:
        rpr = rpr_default.find(qn("w:rPr"))
        if rpr is not None and rpr.find(qn("w:rFonts")) is None:
            rpr.insert(0, OxmlElement("w:rFonts"))
    for fonts in styles.iter(qn("w:rFonts")):
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            fonts.attrib.pop(qn(attr), None)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fonts.set(qn(attr), FONT)


def paragraph_style(doc, name, *, size=13.0, bold=None, color=None, align=None, before=0.0, after=4.0,
                    line=1.25, indent=None, keep=None, page_break=None):
    if name not in [s.name for s in doc.styles]:
        return
    style = doc.styles[name]
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    fmt = style.paragraph_format
    if align is not None:
        fmt.alignment = align
    fmt.space_before, fmt.space_after, fmt.line_spacing = Pt(before), Pt(after), line
    if indent is not None:
        fmt.first_line_indent = indent
    if keep is not None:
        fmt.keep_with_next = keep
    if page_break is not None:
        fmt.page_break_before = page_break


def style_document(doc) -> None:
    use_times(doc)
    justify, center, left = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
    paragraph_style(doc, "Normal", align=justify)
    for name in ("Body Text", "First Paragraph"):
        paragraph_style(doc, name, align=justify, indent=Pt(18))
    paragraph_style(doc, "Compact", align=left, after=2)
    paragraph_style(doc, "Heading 1", size=16, bold=True, color=NAVY, align=center, after=12, line=1.15,
                    keep=True, page_break=True)
    paragraph_style(doc, "Heading 2", size=14, bold=True, color=NAVY, align=left, before=10, line=1.15,
                    keep=True)
    paragraph_style(doc, "Heading 3", size=13, bold=True, color=NAVY, align=left, before=8, line=1.15,
                    keep=True)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        if name in [s.name for s in doc.styles]:
            doc.styles[name].font.italic = False
    if "Verbatim Char" in [s.name for s in doc.styles]:
        doc.styles["Verbatim Char"].font.name = "Consolas"
        doc.styles["Verbatim Char"].font.size = Pt(10.5)
    for level, indent in ((1, 0), (2, 0.6)):
        name = f"toc {level}"
        style = (doc.styles[name] if name in [s.name for s in doc.styles]
                 else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH))
        style.base_style = doc.styles["Normal"]
        style.font.name, style.font.size, style.font.bold = FONT, Pt(13), level == 1
        fmt = style.paragraph_format
        fmt.alignment, fmt.left_indent, fmt.first_line_indent = left, Cm(indent), Pt(0)
        fmt.space_before, fmt.space_after, fmt.line_spacing = Pt(0), Pt(4), 1.15
        fmt.tab_stops.add_tab_stop(TEXT_WIDTH, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

    for p in doc.paragraphs:
        xml = p._p.xml
        if "<w:drawing" in xml:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.keep_with_next = True
            scale_drawings(p._p)
        elif "<m:oMathPara" in xml:
            p.paragraph_format.first_line_indent = Pt(0)
        if any(r.style is not None and r.style.name == "Verbatim Char" and len(r.text) > 30 for r in p.runs):
            p.alignment = left  # long paths/hashes cannot wrap, so justification would stretch the line
        text = p.text
        for mark in (TABLE_MARK, FIGURE_MARK):
            if text.startswith(mark):
                caption(p, text[len(mark):], keep_with_next=mark == TABLE_MARK)
    for table in doc.tables:
        style_table(table)


def scale_drawings(element) -> None:
    for extent in element.iter(qn("wp:extent")):
        cx, cy = int(extent.get("cx")), int(extent.get("cy"))
        if cx > MAX_FIGURE_EMU:
            ratio = MAX_FIGURE_EMU / cx
            for node in element.iter():
                if node.tag in (qn("wp:extent"), qn("a:ext")) and node.get("cx"):
                    node.set("cx", str(int(int(node.get("cx")) * ratio)))
                    node.set("cy", str(int(int(node.get("cy")) * ratio)))
            return


def caption(paragraph, text: str, *, keep_with_next: bool) -> None:
    for run in list(paragraph.runs):
        run._r.getparent().remove(run._r)
    label, _, rest = text.partition(". ")
    for part, bold in ((label + ". ", True), (rest, False)):
        run = paragraph.add_run(part)
        run.font.size, run.font.bold, run.font.italic = Pt(12), bold, not bold
        run.font.color.rgb = NAVY
    fmt = paragraph.paragraph_format
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt.first_line_indent, fmt.space_before, fmt.space_after = Pt(0), Pt(6), Pt(8)
    fmt.keep_with_next = keep_with_next


def cell_borders(cell, color: str = "99B4D1") -> None:
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
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:color"), color)


def style_table(table, font_size: float = 11, widths_cm=None) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    width = tbl_pr.find(qn("w:tblW"))
    if width is None:
        width = OxmlElement("w:tblW")
        tbl_pr.append(width)
    if widths_cm:  # fixed layout: LibreOffice ignores per-cell widths under an autofit 100% table
        table.autofit = False
        width.set(qn("w:type"), "dxa")
        width.set(qn("w:w"), str(round(sum(widths_cm) * 567)))
        for column, value in zip(table.columns, widths_cm):
            column.width = Cm(value)
        for row in table.rows:
            for cell, value in zip(row.cells, widths_cm):
                cell.width = Cm(value)
    else:
        width.set(qn("w:type"), "pct")
        width.set(qn("w:w"), "5000")
    for r, row in enumerate(table.rows):
        if r == 0:
            header = OxmlElement("w:tblHeader")
            header.set(qn("w:val"), "true")
            row._tr.get_or_add_trPr().append(header)
        for cell in row.cells:
            cell_borders(cell)
            if r == 0:
                set_shading(cell, "D9EAF7")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                fmt = p.paragraph_format
                fmt.first_line_indent, fmt.space_before, fmt.space_after = Pt(0), Pt(2), Pt(2)
                fmt.line_spacing = 1.0
                if p.alignment in (None, WD_ALIGN_PARAGRAPH.JUSTIFY):
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if r == 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    run.font.size = Pt(font_size)
                    if r == 0:
                        run.font.bold = True


def configure_body_section(doc) -> None:
    section = doc.sections[-1]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2.49)
    section.left_margin, section.right_margin = Cm(3.0), Cm(2.01)
    section.header_distance = section.footer_distance = Cm(1.27)
    section.footer.is_linked_to_previous = False
    footer = section.footer.paragraphs[0]
    add_page_number(footer)
    for run in footer.runs:
        run.font.name, run.font.size = FONT, Pt(12)
    set_page_numbering(section._sectPr, "decimal")


def set_page_numbering(sect_pr, fmt: str) -> None:
    numbering = sect_pr.find(qn("w:pgNumType"))
    if numbering is None:
        numbering = OxmlElement("w:pgNumType")
        later = ("cols", "formProt", "vAlign", "noEndnote", "titlePg", "textDirection", "bidi",
                 "rtlGutter", "docGrid", "printerSettings", "sectPrChange")
        anchor = next((child for child in sect_pr if child.tag in {qn("w:" + n) for n in later}), None)
        if anchor is None:
            sect_pr.append(numbering)
        else:
            anchor.addprevious(numbering)
    numbering.set(qn("w:fmt"), fmt)
    numbering.set(qn("w:start"), "1")


def section_break(doc, fmt: str):
    """Paragraph that ends a section with the body page setup and its own page numbering."""
    sect_pr = copy.deepcopy(doc.sections[-1]._sectPr)
    set_page_numbering(sect_pr, fmt)
    paragraph = OxmlElement("w:p")
    properties = OxmlElement("w:pPr")
    properties.append(sect_pr)
    paragraph.append(properties)
    return paragraph


def insert_cover(doc, before) -> None:
    template = Document(TEMPLATE)
    elements = list(template.element.body)[:23]
    if "<w:sectPr" not in elements[-1].xml or "TRƯỜNG ĐẠI HỌC VĂN LANG" not in elements[2].xml:
        raise ValueError(f"Bìa mẫu không đúng cấu trúc 23 đoạn: {TEMPLATE}")
    images, drawing_id = {}, 9000
    for element in elements:
        clone = copy.deepcopy(element)
        for extensions in clone.findall(".//" + qn("a:extLst")):  # optional Office 2010+ extras
            extensions.getparent().remove(extensions)
        for node in clone.iter():
            if node.tag.split("}")[0][1:] not in CORE_NS:
                raise ValueError(f"Bìa mẫu có phần tử Office mở rộng chưa hỗ trợ: {node.tag}")
            for attr, value in list(node.attrib.items()):
                if attr.startswith("{") and attr[1:].split("}")[0] not in CORE_NS:
                    del node.attrib[attr]  # e.g. w14:paraId; optional ids that need mc:Ignorable
                elif attr.startswith(R_NS):
                    if value not in images:
                        blob = template.part.related_parts[value].blob
                        images[value], _ = doc.part.get_or_add_image(io.BytesIO(blob))
                    node.set(attr, images[value])
            if node.tag in (qn("wp:docPr"), qn("pic:cNvPr")):
                drawing_id += 1
                node.set("id", str(drawing_id))
        before.addprevious(clone)
    # The bottom-right ornament is anchored to a paragraph, so Word and LibreOffice place it
    # differently. Pin it to the page, mirroring the top-left ornament (same page-edge offsets).
    corner = next(p._p for p in doc.paragraphs[:23] if "<wp:anchor" in p._p.xml
                  and 'relativeFrom="column"' in p._p.xml)
    top_left = next(a for a in doc.paragraphs[0]._p.iter(qn("wp:anchor"))
                    if a.find(qn("wp:positionV")).get("relativeFrom") == "page")
    anchor = corner.find(".//" + qn("wp:anchor"))
    width, height = (int(anchor.find(qn("wp:extent")).get(k)) for k in ("cx", "cy"))
    offsets = {axis: int(top_left.find(qn(f"wp:position{axis}")).find(qn("wp:posOffset")).text)
               for axis in ("H", "V")}
    page = {"H": int(Cm(21)) - offsets["H"] - width, "V": int(Cm(29.7)) - offsets["V"] - height}
    for axis in ("H", "V"):
        position = anchor.find(qn(f"wp:position{axis}"))
        position.set("relativeFrom", "page")
        position.find(qn("wp:posOffset")).text = str(page[axis])
    paragraphs = doc.paragraphs
    set_texts(paragraphs[14], f"GVHD:  {ADVISOR}")
    line_four = copy.deepcopy(paragraphs[17]._p)
    paragraphs[18]._p.addprevious(line_four)
    paragraphs[18]._p.getparent().remove(paragraphs[18]._p)
    for number, (name, student_id, *_rest) in enumerate(TEAM, start=1):
        set_texts(doc.paragraphs[14 + number], f"SVTH {number}:  ", f"{name} – MSSV: {student_id}")


def set_texts(paragraph, *texts: str) -> None:
    """Rewrite a cover line, keeping the formatting of its first non-empty runs (bold label, value)."""
    runs = [run for run in paragraph.runs if run.text]
    keep = runs[:len(texts) - 1] + runs[-1:]
    if len(keep) != len(texts):
        raise ValueError(f"Dòng bìa không đủ run để thay: {paragraph.text!r}")
    for run in paragraph.runs:
        if all(run._r is not kept._r for kept in keep):
            run._r.getparent().remove(run._r)
    for run, text in zip(keep, texts):
        run.text = text


def heading(doc, text: str, *, outline: bool = True):
    if outline:
        return doc.add_paragraph(text, style="Heading 1")
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size, run.font.bold, run.font.color.rgb = Pt(16), True, NAVY
    fmt = p.paragraph_format
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt.page_break_before, fmt.keep_with_next, fmt.space_after = True, True, Pt(12)
    fmt.first_line_indent = Pt(0)
    return p


def body_paragraph(doc, text: str, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, italic=False, indent=True):
    p = doc.add_paragraph(text)
    p.alignment = align
    fmt = p.paragraph_format
    fmt.first_line_indent = Pt(18) if indent else Pt(0)
    fmt.space_after, fmt.line_spacing = Pt(6), 1.25
    for run in p.runs:
        run.font.italic = italic
    return p


def simple_table(doc, headers, rows, widths_cm):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    for c, text in enumerate(headers):
        table.rows[0].cells[c].text = text
    for r, row in enumerate(rows, start=1):
        for c, text in enumerate(row):
            table.rows[r].cells[c].text = text
    style_table(table, font_size=11, widths_cm=widths_cm)
    return table


def toc_field(doc):
    p = doc.add_paragraph(style="toc 1")
    for kind, text in (("begin", None), ("instr", ' TOC \\o "1-2" \\h \\z \\u '), ("separate", None),
                       ("text", TOC_PLACEHOLDER), ("end", None)):
        run = p.add_run()
        if kind == "instr":
            node = OxmlElement("w:instrText")
            node.set(qn("xml:space"), "preserve")
            node.text = text
        elif kind == "text":
            node = OxmlElement("w:t")
            node.text = text
        else:
            node = OxmlElement("w:fldChar")
            node.set(qn("w:fldCharType"), kind)
        run._r.append(node)
    return p


def build_front_matter(doc, before) -> None:
    start = len(doc.element.body) - 1  # new blocks are added before the final sectPr
    heading(doc, "LỜI CẢM ƠN")
    for text in ACKNOWLEDGEMENTS:
        body_paragraph(doc, text)
    body_paragraph(doc, "TP. Hồ Chí Minh, tháng 11 năm 2026", align=WD_ALIGN_PARAGRAPH.RIGHT,
                   italic=True, indent=False)
    body_paragraph(doc, "Nhóm sinh viên thực hiện", align=WD_ALIGN_PARAGRAPH.RIGHT, indent=False)

    heading(doc, "BẢNG PHÂN CÔNG NHIỆM VỤ VÀ ĐÁNH GIÁ MỨC ĐỘ HOÀN THÀNH")
    caption(doc.add_paragraph(), "Bảng 0.1. Phân công nhiệm vụ và mức độ hoàn thành của thành viên",
            keep_with_next=True)
    simple_table(doc, ["STT", "Họ và tên", "MSSV", "Nhiệm vụ", "Hoàn thành", "Xác nhận"],
                 [(str(i), name, sid, task, done, "")
                  for i, (name, sid, task, done) in enumerate(TEAM, start=1)],
                 [1.2, 3.0, 3.2, 5.5, 1.7, 1.39])

    heading(doc, "MỤC LỤC", outline=False)
    toc_field(doc)

    heading(doc, "DANH MỤC TỪ VIẾT TẮT")
    simple_table(doc, ["Từ viết tắt", "Ý nghĩa"], ABBREVIATIONS, [3.5, 12.49])

    body = doc.element.body
    created = list(body)[start:-1]
    for element in created:
        before.addprevious(element)


def fill_toc(doc, entries: list[tuple[str, str]]) -> None:
    """Cache LibreOffice's refreshed contents inside the Word field so it shows without F9."""
    headings = [(1 if p.style.name == "Heading 1" else 2, p.text.strip()) for p in doc.paragraphs
                if p.style.name in ("Heading 1", "Heading 2") and p.text.strip()]
    if [t for _, t in headings] != [t for t, _ in entries]:
        raise ValueError("Mục lục LibreOffice không khớp danh sách đề mục; kiểm tra lại tiêu đề.")
    field = next(p for p in doc.paragraphs if TOC_PLACEHOLDER in p.text)
    runs = field._p.findall(qn("w:r"))
    begin, end = runs[:3], runs[-1]
    previous = field._p
    for i, ((level, title), (_, page)) in enumerate(zip(headings, entries)):
        p = copy.deepcopy(field._p)
        for run in p.findall(qn("w:r")):
            p.remove(run)
        p.find(qn("w:pPr")).find(qn("w:pStyle")).set(qn("w:val"), doc.styles[f"toc {level}"].style_id)
        if i == 0:
            for run in begin:
                p.append(copy.deepcopy(run))
        for kind, value in (("t", title), ("tab", None), ("t", page)):
            run = OxmlElement("w:r")
            node = OxmlElement("w:" + kind)
            if value is not None:
                node.text = value
                node.set(qn("xml:space"), "preserve")
            run.append(node)
            p.append(run)
        if i == len(headings) - 1:
            p.append(copy.deepcopy(end))
        previous.addnext(p)
        previous = p
    field._p.getparent().remove(field._p)


def read_toc(path: Path) -> list[tuple[str, str]]:
    texts = json.loads(path.read_text(encoding="utf-8"))
    if len(texts) != 1:
        raise ValueError(f"Cần đúng một mục lục, LibreOffice trả về {len(texts)}")
    entries = []
    for line in texts[0].splitlines():
        title, _, page = line.rpartition("\t")
        if title.strip():
            entries.append((title.strip(), page.strip()))
    return entries


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = Document(run_pandoc(report_markdown()))
    style_document(doc)
    configure_body_section(doc)
    body = doc.element.body
    first = body[0]
    chapter_one = next(p._p for p in doc.paragraphs if p.text.startswith("CHƯƠNG 1."))
    chapter_one.addprevious(section_break(doc, "lowerRoman"))
    build_front_matter(doc, first)
    insert_cover(doc, body[0])
    doc.core_properties.title = "UFM-Rec: Hệ khuyến nghị sản phẩm cold-start đa phương thức"
    doc.core_properties.subject = "Báo cáo đồ án môn Học máy ứng dụng"
    doc.core_properties.author = ", ".join(name for name, *_ in TEAM)
    stage = BUILD / "stage.docx"
    doc.save(stage)

    toc_json = BUILD / "toc.json"
    subprocess.run([str(LO_PYTHON), str(Path(__file__).with_name("lo_update_export.py")), str(stage),
                    str(OUTPUT_PDF), str(toc_json), str(BUILD / "lo_profile")], check=True)
    fill_toc(doc, read_toc(toc_json))
    doc.save(OUTPUT_DOCX)
    print(OUTPUT_DOCX)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
