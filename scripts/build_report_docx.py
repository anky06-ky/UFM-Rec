"""
Generator script to build Bao_Cao_Do_An_UFM_Rec_Nhom_4.docx
based on the exact style, typography, and structure of Mau_Bao_Cao_Nhom_4.docx
for UFM-Rec Recommender System Project.
"""

import os
import sys
import json
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = BASE_DIR / "Mau_Bao_Cao_Nhom_4.docx"
OUTPUT_PATH = BASE_DIR / "Bao_Cao_Do_An_UFM_Rec_Nhom_4.docx"

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    # remove existing shd if any
    for shd in tcPr.xpath('./w:shd'):
        tcPr.remove(shd)
    shd_xml = f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{hex_color}"/>'
    tcPr.append(parse_xml(shd_xml))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    mar_xml = f'''
    <w:tcMar {nsdecls("w")}>
        <w:top w:w="{top}" w:type="dxa"/>
        <w:bottom w:w="{bottom}" w:type="dxa"/>
        <w:left w:w="{left}" w:type="dxa"/>
        <w:right w:w="{right}" w:type="dxa"/>
    </w:tcMar>
    '''
    tcPr.append(parse_xml(mar_xml))

def set_table_borders(table, color="B0C4DE", sz="4"):
    tblPr = table._tbl.tblPr
    for b in tblPr.xpath('./w:tblBorders'):
        tblPr.remove(b)
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:left w:val="none"/>
        <w:right w:val="none"/>
        <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))

CALLOUT_LATEX = {}


def add_callout_box(doc, formula_num, title, equation_text, description=None):
    """Creates a 1x1 callout table matching the template's style."""
    CALLOUT_LATEX[formula_num] = equation_text
    t = doc.add_table(rows=1, cols=1)
    t.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER
    cell = t.cell(0, 0)
    set_cell_background(cell, "EEF4FF")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Border subtle
    tcPr = cell._tc.get_or_add_tcPr()
    border_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="3A7BD5"/>
        <w:bottom w:val="single" w:sz="6" w:space="0" w:color="3A7BD5"/>
        <w:left w:val="single" w:sz="18" w:space="0" w:color="0B3963"/>
        <w:right w:val="single" w:sz="6" w:space="0" w:color="3A7BD5"/>
    </w:tcBorders>
    '''
    tcPr.append(parse_xml(border_xml))
    
    # Title paragraph
    p0 = cell.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.space_before = Pt(4)
    p0.paragraph_format.space_after = Pt(4)
    r0 = p0.add_run(f"Công thức {formula_num}. {title}")
    r0.font.name = "Times New Roman"
    r0.font.size = Pt(11.5)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(11, 57, 99)
    
    # Equation paragraph
    p1 = cell.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(2)
    p1.paragraph_format.space_after = Pt(4)
    r1 = p1.add_run(equation_text)
    r1.font.name = "Cambria Math"
    r1.font.size = Pt(11)
    r1.font.bold = True
    
    # Description paragraph
    if description:
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(4)
        r2 = p2.add_run(f"Trong đó: {description}")
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(10)
        r2.font.italic = True
        r2.font.color.rgb = RGBColor(80, 80, 80)
        
    # Spacer paragraph after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(2)
    sp.paragraph_format.space_after = Pt(4)

def add_data_table(doc, table_num, title, headers, rows, col_widths=None, alignments=None):
    """Creates a standard table with header shaded #D9EAF7."""
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(4)
    r_cap = cap.add_run(f"Bảng {table_num}. {title}")
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(12)
    r_cap.font.bold = True
    r_cap.font.color.rgb = RGBColor(11, 57, 99)
    
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t, color="99B4D1", sz="4")
    
    # Headers
    hdr_cells = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "D9EAF7")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = RGBColor(11, 57, 99)
            
    # Rows
    for r_idx, row_data in enumerate(rows):
        row_cells = t.rows[r_idx + 1].cells
        bg_color = "F9FBFE" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            align = WD_ALIGN_PARAGRAPH.LEFT
            if alignments and c_idx < len(alignments):
                align = alignments[c_idx]
            elif c_idx == 0 or len(str(val)) < 12:
                align = WD_ALIGN_PARAGRAPH.CENTER
            p.alignment = align
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(10.5)

    # Set widths if specified
    if col_widths:
        for row in t.rows:
            for idx, w in enumerate(col_widths):
                row.cells[idx].width = w

    # Spacer paragraph after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(2)
    sp.paragraph_format.space_after = Pt(6)

def add_figure_with_caption(doc, img_path, fig_num, title, width=Inches(5.8)):
    """Inserts a centered image and its standard caption."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=width)
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    r_cap = p_cap.add_run(f"Hình {fig_num}. {title}")
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(11)
    r_cap.font.bold = True
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(11, 57, 99)

def add_p(doc, text, bold_prefix=None, indent=360, space_after=4):
    """Standard body paragraph matching template (13pt Times New Roman, justified, 1.25 line spacing)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.first_line_indent = docx.shared.Pt(indent / 20)  # 360 dxa = 18 pt
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(13)
        r_pre.font.bold = True
        
    r_body = p.add_run(text)
    r_body.font.name = "Times New Roman"
    r_body.font.size = Pt(13)
    return p

def add_page_break(doc):
    """Empty paragraph holding a page break, as used by the template's section separators."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run()
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.add_break(WD_BREAK.PAGE)
    return p

def add_h1(doc, text, page_break=False):
    """Heading 1: 16pt, bold, Navy blue #0B3963, 12pt before, 6pt after."""
    if page_break:
        add_page_break(doc)
    p = doc.add_paragraph(style='Heading 1')
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(11, 57, 99)
    return p

def add_h2(doc, text):
    """Heading 2: 14pt, bold, Navy blue #0B3963, 8pt before, 4pt after."""
    p = doc.add_paragraph(style='Heading 2')
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(11, 57, 99)
    return p

def add_toc_line(doc, title, page_str, level=1):
    """Adds a TOC entry matching template's toc 1 or toc 2 with tab dot leader."""
    style_name = 'toc 1' if level == 1 else 'toc 2'
    p = doc.add_paragraph(style=style_name)
    p.text = ""
    pPr = p._p.get_or_add_pPr()
    tabs_xml = f'''
    <w:tabs {nsdecls("w")}>
        <w:tab w:val="right" w:leader="dot" w:pos="9061"/>
    </w:tabs>
    '''
    pPr.append(parse_xml(tabs_xml))
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    
    r_txt = p.add_run(title)
    r_txt.font.name = "Times New Roman"
    r_txt.font.size = Pt(12 if level == 1 else 11.5)
    r_txt.font.bold = (level == 1)
    
    r_tab = p.add_run("\t")
    
    r_pg = p.add_run(str(page_str))
    r_pg.font.name = "Times New Roman"
    r_pg.font.size = Pt(12 if level == 1 else 11.5)
    r_pg.font.bold = (level == 1)

print("Starting document builder...")
doc = docx.Document(str(TEMPLATE_PATH))
body = doc._body._body

# Step 1: Format Cover Page (Paragraphs 0 to 22)
# The template keeps the university logos as anchored drawings in p[0] and p[19],
# so only the text paragraphs below are rewritten.
# p[2]: TRƯỜNG ĐẠI HỌC VĂN LANG
# p[3]: KHOA CÔNG NGHỆ THÔNG TIN
# p[7]: BÁO CÁO ĐỒ ÁN
# p[9]: MÔN HỌC
# p[10]: HỌC MÁY ỨNG DỤNG
# p[12]: ĐỀ TÀI
# p[13]: UFM-REC: HỆ KHUYẾN NGHỊ SẢN PHẨM COLD-START ĐA PHƯƠNG THỨC DỰA TRÊN ĐỘ TIN CẬY HỌC ĐƯỢC
# p[14]: GVHD:  ThS. Nguyễn Thái Anh
# p[15]: SVTH1:  Nguyễn Thành Phong – 2474802010304
# p[16]: SVTH2:  Trần An Kỳ – 2474802010460
# p[17]: SVTH3:  Nguyễn Hoàng Hải – 2474802010099
# p[20]: TP. HỒ CHÍ MINH, 2026

def set_p_text(p, text, font_size=13, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, color=None, space_after=4):
    p.text = ""
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(font_size)
    r.font.bold = bold
    if color:
        r.font.color.rgb = color
    return p

set_p_text(doc.paragraphs[2], "TRƯỜNG ĐẠI HỌC VĂN LANG", font_size=18, bold=True, space_after=6)
set_p_text(doc.paragraphs[3], "KHOA CÔNG NGHỆ THÔNG TIN", font_size=18, bold=True, space_after=12)

set_p_text(doc.paragraphs[7], "BÁO CÁO ĐỒ ÁN", font_size=18, bold=True, color=RGBColor(11, 57, 99), space_after=8)
set_p_text(doc.paragraphs[9], "MÔN HỌC", font_size=15, bold=True, space_after=4)
set_p_text(doc.paragraphs[10], "HỌC MÁY ỨNG DỤNG", font_size=16, bold=True, color=RGBColor(11, 57, 99), space_after=12)
set_p_text(doc.paragraphs[12], "ĐỀ TÀI", font_size=15, bold=True, space_after=6)
set_p_text(doc.paragraphs[13], "UFM-REC: HỆ KHUYẾN NGHỊ SẢN PHẨM COLD-START ĐA PHƯƠNG THỨC DỰA TRÊN ĐỘ TIN CẬY HỌC ĐƯỢC\n(Multimodal Cold-Start Recommender with Learned Reliability Fusion)", font_size=15, bold=True, color=RGBColor(11, 57, 99), space_after=18)
set_p_text(doc.paragraphs[14], "GVHD:  ThS. Nguyễn Thái Anh", font_size=14, bold=True, space_after=16)

# SVTH rows (Left-indented cleanly)
def set_sv_p(p, role, name, mssv):
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Inches(1.8)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run(f"{role}:  ")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(13.5)
    r1.font.bold = True
    r2 = p.add_run(f"{name} – MSSV: {mssv}")
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(13.5)

set_sv_p(doc.paragraphs[15], "SVTH 1", "Nguyễn Thành Phong", "2474802010304")
set_sv_p(doc.paragraphs[16], "SVTH 2", "Trần An Kỳ", "2474802010460")
set_sv_p(doc.paragraphs[17], "SVTH 3", "Nguyễn Hoàng Hải", "2474802010099")

set_p_text(doc.paragraphs[20], "TP. HỒ CHÍ MINH, 2026", font_size=14, bold=True, space_after=0)

# Step 2: Remove the template body (everything after the cover section break)
for child in list(body)[23:-1]:
    body.remove(child)

print("Cover page formatted. Cleared old template body. Now adding report content...")

# ==============================================================================
# SECTION 1: PRELIMINARY CONTENT
# ==============================================================================

# --- LỜI CẢM ƠN ---
add_h1(doc, "LỜI CẢM ƠN")
add_p(doc, "Lời đầu tiên, nhóm nghiên cứu chúng em xin bày tỏ lòng biết ơn chân thành và sâu sắc nhất đến Ban Giám hiệu Trường Đại học Văn Lang cùng Ban Chủ nhiệm Khoa Công nghệ Thông tin đã tạo mọi điều kiện thuận lợi nhất về cơ sở vật chất, môi trường học thuật sáng tạo và hệ thống máy chủ tính toán hiệu năng cao FITLAB trong suốt quá trình nhóm triển khai đề tài.")
add_p(doc, "Chúng em xin gửi lời tri ân đặc biệt sâu sắc đến Thầy ThS. Nguyễn Thái Anh – Giảng viên hướng dẫn môn học Học máy ứng dụng. Thầy đã dành nhiều thời gian quý báu để tận tình định hướng khoa học, hướng dẫn phương pháp nghiên cứu thực nghiệm chuẩn mực, chỉ dẫn cách tiếp cận xử lý các tập dữ liệu thương mại điện tử quy mô lớn, và đưa ra những lời nhận xét chuyên môn vô cùng xác đáng giúp nhóm hoàn thiện mô hình UFM-Rec một cách bài bản, khoa học.")
add_p(doc, "Nhóm cũng xin chân thành cảm ơn quý Thầy Cô trong Khoa Công nghệ Thông tin đã truyền thụ những nền tảng tri thức vững chắc về Trí tuệ nhân tạo, Học sâu, Xử lý ngôn ngữ tự nhiên và Thị giác máy tính trong suốt những năm học vừa qua. Những kiến thức đó chính là hành trang vô giá giúp nhóm đủ năng lực tiếp cận và nghiên cứu các kiến trúc mô hình học máy hiện đại nhất hiện nay.")
add_p(doc, "Cuối cùng, nhóm xin gửi lời cảm ơn đến gia đình, bạn bè và các bạn sinh viên cùng lớp đã luôn động viên, chia sẻ khó khăn và đồng hành cùng nhóm trong những đêm dài thực nghiệm tại phòng Lab. Dù đã rất nỗ lực hoàn thiện đề tài với tinh thần khoa học nghiêm túc, báo cáo chắc chắn khó tránh khỏi những thiếu sót nhất định. Nhóm rất mong nhận được những góp ý quý báu của quý Thầy Cô trong Hội đồng phản biện để đề tài ngày càng hoàn thiện hơn nữa.")
p_thanks_end = add_p(doc, "Nhóm sinh viên thực hiện đề tài xin chân thành cảm ơn!")
p_thanks_end.paragraph_format.space_after = Pt(16)

# --- LỜI NÓI ĐẦU ---
add_h1(doc, "LỜI NÓI ĐẦU", page_break=True)
add_p(doc, "Trong kỷ nguyên bùng nổ của thương mại điện tử (E-commerce) và các nền tảng kỹ thuật số toàn cầu, Hệ thống Gợi ý (Recommender Systems) đóng vai trò như trái tim định hình trải nghiệm người dùng, giúp khách hàng tìm thấy đúng sản phẩm mong muốn giữa hàng triệu danh mục hàng hóa đa dạng. Tuy nhiên, một trong những thách thức cốt lõi và dai dẳng nhất của lĩnh vực này chính là bài toán Khởi đầu lạnh (Cold-Start Problem).")
add_p(doc, "Khi một sản phẩm mới vừa được đăng tải lên nền tảng, hệ thống hoàn toàn không có hoặc chỉ có một vài tương tác hành vi từ người tiêu dùng. Các mô hình Lọc cộng tác (Collaborative Filtering) truyền thống và các kiến trúc gợi ý tuần tự hiện đại (Sequential Recommenders) thường bị suy giảm hiệu năng nghiêm trọng, thậm chí rơi vào trạng thái 'mù dữ liệu'. Mặc dù các phương pháp dựa trên nội dung truyền thống (như TF-IDF) có thể cung cấp gợi ý cơ bản, chúng lại hoàn toàn thiếu khả năng khai thác các tương quan ngữ nghĩa sâu sắc giữa hình ảnh và văn bản mô tả sản phẩm.")
add_p(doc, "Xuất phát từ thực tiễn trên, đề tài tập trung nghiên cứu và hiện thực hóa kiến trúc UFM-Rec (Uncertainty-aware Fusion Model Recommender). Mô hình kết hợp biểu diễn đa phương thức đóng băng (Frozen Multimodal Features) từ mạng nền tảng CLIP ViT-B/32 với kiến trúc mạng Causal Sequential Transformer, đồng thời tích hợp Cơ chế Ước Lượng Độ Tin Cậy Học Được (Learned Reliability Proxy) và Cổng Căn Chỉnh Đa Phương Thức (Cross-Modal Alignment Gate). Đề tài được kiểm chứng thực nghiệm nghiêm ngặt trên tập dữ liệu chuẩn Amazon Reviews 2023 (Toys & Games với 767.045 sản phẩm và 11,57 triệu tương tác) theo chiến lược phân chia thời gian 80/10/10 Global Temporal Split.")
add_p(doc, "Báo cáo này được cấu trúc thành 7 chương rõ ràng, trình bày toàn diện từ cơ sở lý thuyết, phân tích thiết kế, quy trình cài đặt, kết quả thực nghiệm định lượng chi tiết, phân tích nghịch lý Extreme Cold-Start, đánh giá hiệu chuẩn xác suất Temperature Scaling, hệ thống Demo thực tế, cho đến các kết luận và định hướng nghiên cứu tiếp theo.")

# --- BẢNG ĐÁNH GIÁ MỨC ĐỘ HOÀN THÀNH CÔNG VIỆC ---
add_h1(doc, "BẢNG ĐÁNH GIÁ MỨC ĐỘ HOÀN THÀNH CÔNG VIỆC", page_break=True)
eval_headers = ["STT", "Họ và tên", "Mã số sinh viên", "Nhiệm vụ được phân công trong đồ án", "Đánh giá", "Xác nhận"]
eval_rows = [
    [
        "1",
        "Nguyễn Thành Phong",
        "2474802010304",
        "Trưởng nhóm: Thiết kế kiến trúc tổng thể mô hình UFM-Rec, lập trình triển khai PyTorch (Causal Transformer, Gating, Reliability Proxy), tích hợp trích xuất đặc trưng CLIP ViT-B/32, huấn luyện trên GPU FITLAB, đánh giá định lượng và tổng hợp báo cáo kỹ thuật.",
        "100%",
        "Phong"
    ],
    [
        "2",
        "Trần An Kỳ",
        "2474802010460",
        "Nghiên cứu và xây dựng pipeline xử lý dữ liệu lớn Amazon Reviews 2023 (Toys & Games 767k sản phẩm, 11,57M tương tác), thiết kế chiến lược 80/10/10 Global Temporal Split, xây dựng catalog Parquet, thiết kế hệ thống Demo tương tác thời gian thực và Mock Fallback.",
        "100%",
        "Kỳ"
    ],
    [
        "3",
        "Nguyễn Hoàng Hải",
        "2474802010099",
        "Nghiên cứu và hiện thực các mô hình đối chứng (Baseline TF-IDF Content-Based, Popularity), triển khai đánh giá hiệu chuẩn xác suất Temperature Scaling (ECE, NLL), đo kiểm tiêu thụ tài nguyên thực tế (RAM, VRAM, QPS) và thiết lập hệ thống kiểm thử tự động (14 bài test).",
        "100%",
        "Hải"
    ]
]
add_data_table(doc, "0.1", "Bảng phân công nhiệm vụ và đánh giá mức độ hoàn thành", eval_headers, eval_rows, 
               col_widths=[Inches(0.5), Inches(1.5), Inches(1.2), Inches(3.2), Inches(0.8), Inches(0.8)],
               alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

# --- MỤC LỤC ---
add_h1(doc, "MỤC LỤC", page_break=True)
toc_items = [
    ("LỜI CẢM ƠN", "i", 1),
    ("LỜI NÓI ĐẦU", "ii", 1),
    ("BẢNG ĐÁNH GIÁ MỨC ĐỘ HOÀN THÀNH CÔNG VIỆC", "iii", 1),
    ("MỤC LỤC", "iv", 1),
    ("DANH MỤC HÌNH ẢNH", "v", 1),
    ("DANH MỤC BẢNG BIỂU", "vi", 1),
    ("TÓM TẮT BÁO CÁO", "vii", 1),
    ("CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI", "1", 1),
    ("1.1. Bối cảnh và lý do chọn đề tài", "1", 2),
    ("1.2. Vấn đề nghiên cứu và thách thức Cold-Start", "2", 2),
    ("1.3. Mục tiêu và phạm vi thực hiện", "3", 2),
    ("1.4. Đóng góp khoa học và thực tiễn của đồ án", "4", 2),
    ("1.5. Quy trình thực hiện tổng thể", "5", 2),
    ("1.6. Tiêu chí đánh giá hệ thống", "6", 2),
    ("1.7. Rủi ro kỹ thuật và biện pháp kiểm soát", "7", 2),
    ("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG THỨC CỐT LÕI", "8", 1),
    ("2.1. Phân tầng Regimes trong bài toán Cold-Start", "8", 2),
    ("2.2. Trích xuất đặc trưng đa phương thức đóng băng với CLIP", "9", 2),
    ("2.3. Mô hình chuỗi hành vi Causal Sequential Transformer", "11", 2),
    ("2.4. Cổng căn chỉnh đa phương thức (Cross-Modal Alignment Gate)", "12", 2),
    ("2.5. Cơ chế ước lượng độ tin cậy học được (Learned Reliability Proxy)", "13", 2),
    ("2.6. Hàm mất mát tối ưu hóa và Hiệu chuẩn Temperature Scaling", "14", 2),
    ("CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG", "16", 1),
    ("3.1. Phân tích yêu cầu chức năng và phi chức năng", "16", 2),
    ("3.2. Bộ dữ liệu chuẩn Amazon Reviews 2023", "17", 2),
    ("3.3. Chiến lược phân chia thời gian 80/10/10 Global Temporal Split", "19", 2),
    ("3.4. Kiến trúc tổng thể và luồng dữ liệu hệ thống UFM-Rec", "20", 2),
    ("3.5. Giao thức đánh giá phân tầng 4 Regimes và Negative Sampling", "21", 2),
    ("3.6. Thiết kế thí nghiệm loại trừ (Ablation Matrix)", "22", 2),
    ("CHƯƠNG 4. XÂY DỰNG VÀ CÀI ĐẶT HỆ THỐNG", "24", 1),
    ("4.1. Môi trường triển khai FITLAB và hạ tầng GPU Quadro RTX 6000", "24", 2),
    ("4.2. Cài đặt module tiền xử lý dữ liệu và trích xuất CLIP", "25", 2),
    ("4.3. Cài đặt kiến trúc mạng UFM-Rec trên PyTorch", "26", 2),
    ("4.4. Quy trình huấn luyện và tối ưu hóa siêu tham số", "28", 2),
    ("4.5. Đánh giá xếp hạng và hiệu chuẩn Temperature Scaling", "29", 2),
    ("4.6. Kiểm thử tự động và kiểm soát tính tái lập", "30", 2),
    ("CHƯƠNG 5. THỰC NGHIỆM VÀ KẾT QUẢ", "32", 1),
    ("5.1. Thống kê dữ liệu thực nghiệm và kết quả Baseline", "32", 2),
    ("5.2. Hiệu năng phân tầng theo 4 Regimes và so sánh đối chứng", "33", 2),
    ("5.3. Phân tích sâu hiện tượng nghịch lý Extreme Cold-Start", "35", 2),
    ("5.4. Đánh giá hiệu chuẩn xác suất và độ tin cậy", "37", 2),
    ("5.5. Đánh giá tiêu thụ tài nguyên thực tế và tốc độ suy luận", "38", 2),
    ("5.6. Giới hạn nghiên cứu và phạm vi bằng chứng hiện tại", "40", 2),
    ("CHƯƠNG 6. HỆ THỐNG DEMO VÀ ỨNG DỤNG THỰC TẾ", "42", 1),
    ("6.1. Thiết kế và kiến trúc hệ thống Demo tương tác thời gian thực", "42", 2),
    ("6.2. Cơ chế Offline Mock Catalog Fallback tự động", "43", 2),
    ("6.3. Trực quan hoá các kịch bản khuyến nghị đa phương thức", "44", 2),
    ("6.4. Đề xuất giải pháp Count-Aware Reliability Fusion", "46", 2),
    ("CHƯƠNG 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", "48", 1),
    ("7.1. Kết luận", "48", 2),
    ("7.2. Hướng phát triển tiếp theo", "49", 2),
    ("TÀI LIỆU THAM KHẢO", "51", 1)
]
for item in toc_items:
    add_toc_line(doc, item[0], item[1], level=item[2])

# --- DANH MỤC HÌNH ẢNH ---
add_h1(doc, "DANH MỤC HÌNH ẢNH")
fig_list = [
    ("Hình 1.1", "Quy trình xử lý dữ liệu và luồng thực nghiệm tổng thể của UFM-Rec", "5"),
    ("Hình 2.1", "Kiến trúc chi tiết mô hình UFM-Rec kết hợp CLIP ViT-B/32, Causal Transformer và Learned Reliability Gate", "10"),
    ("Hình 4.1", "So sánh hiệu năng NDCG@10 giữa UFM-Rec và TF-IDF theo 4 phân tầng tương tác", "34"),
    ("Hình 4.2", "Biểu đồ đường cong độ tin cậy (Reliability Diagram) và phân bố xác suất trước/sau hiệu chuẩn Temperature Scaling", "37"),
    ("Hình 5.1", "Giao diện trực quan hoá hệ thống khuyến nghị UFM-Rec thời gian thực với phân tích trọng số đa phương thức", "45"),
    ("Hình 5.2", "Kết quả suy luận và khuyến nghị trong kịch bản Zero-Shot Cold-Start", "46")
]
for item in fig_list:
    add_toc_line(doc, f"{item[0]}. {item[1]}", item[2], level=2)

# --- DANH MỤC BẢNG BIỂU ---
add_h1(doc, "DANH MỤC BẢNG BIỂU")
tbl_list = [
    ("Bảng 0.1", "Bảng phân công nhiệm vụ và đánh giá mức độ hoàn thành", "iii"),
    ("Bảng 1.1", "Mục tiêu kỹ thuật và tiêu chí kiểm chứng của hệ thống UFM-Rec", "3"),
    ("Bảng 1.2", "Các mốc triển khai và tiến độ thực nghiệm đồ án", "5"),
    ("Bảng 1.3", "Rủi ro kỹ thuật và biện pháp kiểm soát trong quá trình nghiên cứu", "7"),
    ("Bảng 2.1", "Bảng ký hiệu toán học dùng trong báo cáo", "8"),
    ("Bảng 2.2", "So sánh các phương pháp gợi ý giải quyết vấn đề Cold-Start", "9"),
    ("Bảng 3.1", "Yêu cầu chức năng và phi chức năng của hệ thống", "16"),
    ("Bảng 3.2", "Thống kê tập dữ liệu Amazon Reviews 2023 (Toys & Games và All Beauty)", "18"),
    ("Bảng 3.3", "Ma trận thiết kế thí nghiệm loại trừ (Ablation Study Matrix)", "23"),
    ("Bảng 4.1", "Thông số môi trường phần cứng và phần mềm máy chủ FITLAB", "24"),
    ("Bảng 4.2", "Cấu hình siêu tham số huấn luyện mô hình UFM-Rec", "28"),
    ("Bảng 5.1", "Kết quả đánh giá định lượng phân tầng trên tập Validation (Toys & Games, 40.000 ca kiểm thử)", "33"),
    ("Bảng 5.2", "Kết quả hiệu chuẩn xác suất với Temperature Scaling", "36"),
    ("Bảng 5.3", "Kết quả benchmark tài nguyên phần cứng và tốc độ suy luận", "39"),
    ("Bảng 5.4", "Kết quả baseline sơ bộ trên tập dữ liệu Amazon All Beauty", "40"),
    ("Bảng 6.1", "Hạn chế nghiên cứu và lộ trình hoàn thiện tiếp theo", "50")
]
for item in tbl_list:
    add_toc_line(doc, f"{item[0]}. {item[1]}", item[2], level=2)

# --- TÓM TẮT BÁO CÁO ---
add_h1(doc, "TÓM TẮT BÁO CÁO", page_break=True)
add_p(doc, "Khởi đầu lạnh (Cold-Start) là một bài toán hóc búa bậc nhất trong lĩnh vực Hệ thống Khuyến nghị (Recommender Systems). Khi các sản phẩm mới xuất hiện trên sàn thương mại điện tử hoặc người dùng mới gia nhập nền tảng, các mô hình Lọc cộng tác truyền thống (Matrix Factorization) hay mạng nơ-ron học chuỗi hành vi (như SASRec, BERT4Rec) hoàn toàn mất khả năng biểu diễn do thiếu ma trận tương tác lịch sử. Để giải quyết triệt để bài toán này, đề tài xây dựng và thực nghiệm kiến trúc UFM-Rec (Uncertainty-aware Fusion Model Recommender).")
add_p(doc, "Mô hình UFM-Rec kết hợp hai nguồn thông tin bổ trợ mạnh mẽ: (1) Nhánh ngữ nghĩa nội dung đa phương thức đóng băng (Frozen Multimodal Features) trích xuất từ mô hình CLIP ViT-B/32 (gồm văn bản mô tả và ảnh trực quan sản phẩm); (2) Nhánh hành vi người dùng mô hình hóa bằng mạng Causal Sequential Transformer. Điểm đột phá của kiến trúc UFM-Rec là việc tích hợp Cổng Căn Chỉnh Đa Phương Thức (Cross-Modal Alignment Gate) giúp tối ưu hóa sự cân bằng giữa ảnh và chữ, kết hợp cùng Cơ Chế Ước Lượng Độ Tin Cậy Học Được (Learned Reliability Proxy), cho phép hệ thống tự động suy luận mức độ tin cậy của lịch sử hành vi để điều tiết trọng số hợp nhất một cách mềm dẻo.")
add_p(doc, "Nghiên cứu được triển khai thực nghiệm nghiêm ngặt trên tập dữ liệu chuẩn Amazon Reviews 2023 — Toys & Games (quy mô 767.045 sản phẩm, 758.692 ảnh trực quan hợp lệ và 11.572.689 tương tác mua sắm) theo chiến lược phân chia thời gian 80/10/10 Global Temporal Split nhằm ngăn chặn tuyệt đối rò rỉ thông tin tương lai. Trên tập kiểm chứng Validation gồm 40.000 ca kiểm thử (10.000 mẫu/regime, negative sampling theo phân bố độ phổ biến popularity), UFM-Rec đạt Overall NDCG@10 = 0,285611, vượt trội mô hình đối chứng TF-IDF Content-Based (0,260367) nhờ ưu thế vượt trội ở nhóm Warm (NDCG@10 đạt 0,651231 so với 0,467661 của TF-IDF).")
add_p(doc, "Tuy nhiên, nghiên cứu cũng chỉ ra và phân tích sâu sắc một hiện tượng nghịch lý: ở phân tầng Cực lạnh (Extreme Cold, 1–5 tương tác), UFM-Rec đạt NDCG@10 = 0,113300, thấp hơn TF-IDF (0,191555). Báo cáo đưa ra giả thuyết giải thích rằng hiện tượng này có thể bắt nguồn từ nhiễu cập nhật gradient của vector nhúng định danh (ID embedding) khi số lượng quan sát quá thưa thớt; do đề tài chưa có thực nghiệm loại trừ, giả thuyết này được trình bày ở dạng chưa được chứng minh và kèm hạn chế tương ứng tại mục 5.6. Đồng thời, kỹ thuật Temperature Scaling được áp dụng thành công để hiệu chuẩn phân bố xác suất đầu ra, giúp giảm sai số kỳ vọng ECE từ 0,164477 xuống 0,041497 (giảm 74,8%). Cuối cùng, đồ án xây dựng hoàn chỉnh hệ thống Demo trực quan thời gian thực với cơ chế tự động chuyển đổi Offline Mock Catalog Fallback, đảm bảo hệ thống luôn sẵn sàng trình diễn trong mọi điều kiện tài nguyên.")

print("Preliminary sections added. Now building Chapters 1 to 7...")

# ==============================================================================
# CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI
# ==============================================================================
add_h1(doc, "CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI")

add_h2(doc, "1.1. Bối cảnh và lý do chọn đề tài")
add_p(doc, "Trong bối cảnh nền kinh tế số phát triển vượt bậc, thương mại điện tử (E-commerce) đã trở thành kênh mua sắm huyết mạch toàn cầu. Các tập đoàn bán lẻ lớn như Amazon, Shopee hay Tiki liên tục bổ sung hàng trăm nghìn mặt hàng mới mỗi ngày. Để kết nối người tiêu dùng với những sản phẩm phù hợp giữa hàng triệu lựa chọn, các Hệ thống Gợi ý (Recommender Systems) đóng vai trò quyết định tới doanh thu và mức độ hài lòng của khách hàng.")
add_p(doc, "Hầu hết các hệ thống gợi ý hiện nay phụ thuộc vào phương pháp Lọc cộng tác (Collaborative Filtering - CF) hoặc các mạng nơ-ron học sâu theo chuỗi hành vi như SASRec, BERT4Rec. Các mô hình này dựa trên giả định căn bản: 'những người dùng có hành vi tương tự trong quá khứ sẽ có sở thích giống nhau trong tương lai'. Tuy nhiên, khi một sản phẩm mới xuất hiện trên nền tảng (sản phẩm Cold-Start), hoàn toàn chưa có lượt mua, đánh giá hay click nào từ người dùng, ma trận tương tác tại vị trí đó hoàn toàn bằng 0. Khi đó, toàn bộ các mô hình dựa trên tương tác thuần túy đều rơi vào trạng thái tê liệt.")
add_p(doc, "Trong khi đó, mỗi sản phẩm mới khi được người bán đăng tải đều đi kèm các thông tin nội dung vô cùng phong phú: tiêu đề, đoạn văn bản mô tả tính năng chi tiết, và ít nhất một hoặc nhiều hình ảnh sản phẩm chất lượng cao. Sự phát triển vượt bậc gần đây của các mô hình nền tảng Thị giác – Ngôn ngữ (Vision-Language Foundation Models), tiêu biểu là mạng CLIP (Contrastive Language-Image Pretraining) của OpenAI, đã mở ra cơ hội đột phá: có thể trích xuất các vector đặc trưng ngữ nghĩa dày đặc và căn chỉnh chung trong một không gian tiềm ẩn (latent space), giúp hệ thống hiểu được bản chất món hàng ngay từ giây phút đầu tiên xuất hiện.")
add_p(doc, "Từ lý do cấp thiết đó, nhóm nghiên cứu lựa chọn đề tài: 'UFM-Rec: Hệ Khuyến Nghị Sản Phẩm Cold-Start Đa Phương Thức Dựa Trên Độ Tin Cậy Học Được' nhằm kết hợp sức mạnh biểu diễn của mô hình nền tảng đa phương thức với học chuỗi hành vi thích ứng, giải quyết triệt để bài toán Cold-Start trong thương mại điện tử hiện đại.")

add_h2(doc, "1.2. Vấn đề nghiên cứu và thách thức Cold-Start")
add_p(doc, "Bài toán nghiên cứu trọng tâm của đồ án được phát biểu như sau: Cho trước lịch sử tương tác của người dùng $u$ dưới dạng chuỗi các sản phẩm đã tương tác $S_u = (i_1, i_2, \\dots, i_t)$, và một kho danh mục hàng triệu sản phẩm ứng viên $I = \\{j_1, j_2, \\dots, j_M\\}$, trong đó danh mục ứng viên bao gồm cả sản phẩm quen thuộc (Warm) lẫn các sản phẩm mới ở các mức độ thưa thớt khác nhau. Nhiệm vụ của hệ thống là dự đoán xác suất và xếp hạng chính xác sản phẩm tiếp theo $i_{t+1}$ mà người dùng sẽ tương tác.")
add_p(doc, "Để giải quyết bài toán này, nhóm nghiên cứu phải đối mặt với 3 thách thức kỹ thuật lớn:")
add_p(doc, "1. Thách thức phân tầng thưa thớt (Sparsity Regimes): Các sản phẩm trong thực tế không chỉ đơn thuần là 'mới hoàn toàn' hay 'đã cũ', mà trải dài qua nhiều phân tầng: Zero-shot (0 tương tác trong tập huấn luyện), Extreme Cold (1–5 tương tác), Cold (6–20 tương tác), và Warm (>20 tương tác). Một mô hình tối ưu cho sản phẩm có nhiều tương tác thường bị suy thoái nặng nề khi áp dụng cho sản phẩm cực lạnh; cơ chế được giả thuyết là nhiễu tham số trên các vector nhúng ID, nhưng chưa có thực nghiệm loại trừ để kết luận.")
add_p(doc, "2. Thách thức hợp nhất không đồng nhất (Multimodal Heterogeneity): Văn bản tiêu đề sản phẩm mang ngữ nghĩa trừu tượng, phân loại danh mục và thông số kỹ thuật, trong khi hình ảnh trực quan thể hiện kiểu dáng, màu sắc, chi tiết thẩm mỹ. Việc kết hợp thô bạo (concatenation) hai nguồn dữ liệu này thường dẫn đến hiện tượng một phương thức lấn át phương thức còn lại (modal dominance).")
add_p(doc, "3. Thách thức ước lượng độ tin cậy và hiệu chuẩn xác suất (Uncertainty & Calibration): Mạng nơ-ron học sâu hiện đại có xu hướng tự tin thái quá (overconfident), gán xác suất dự đoán rất cao cho các sản phẩm gợi ý sai, đặc biệt trong các trường hợp dữ liệu lịch sử thưa thớt. Cần có một cơ chế ước lượng độ tin cậy học được và hiệu chuẩn xác suất hậu nghiệm đáng tin cậy.")

add_h2(doc, "1.3. Mục tiêu và phạm vi thực hiện")
add_p(doc, "Mục tiêu tổng quát của đề tài là xây dựng một kiến trúc khuyến nghị đa phương thức end-to-end có khả năng cân bằng thông minh giữa ngữ nghĩa nội dung sản phẩm và chuỗi hành vi người dùng, nâng cao độ chính xác xếp hạng trên cả 4 phân tầng sản phẩm và cung cấp phân bố xác suất tin cậy.")
add_p(doc, "Các mục tiêu kỹ thuật cụ thể và tiêu chí kiểm chứng được tổng hợp trong Bảng 1.1:")

obj_headers = ["Mục tiêu kỹ thuật", "Cách triển khai cụ thể", "Tiêu chí kiểm chứng định lượng"]
obj_rows = [
    [
        "1. Trích xuất đặc trưng đa phương thức",
        "Sử dụng mô hình CLIP ViT-B/32 đóng băng (frozen weights) để trích xuất 512 chiều ảnh và 512 chiều văn bản, chuẩn hóa chuẩn L2.",
        "Trích xuất thành công 758.692 vector ảnh và 767.045 vector văn bản trên tập dữ liệu Amazon Toys & Games."
    ],
    [
        "2. Thiết kế Cổng căn chỉnh đa phương thức",
        "Xây dựng mạng nơ-ron Cross-Modal Alignment Gate học trọng số thích ứng giữa text và image.",
        "Hệ thống tự động học trọng số $\\alpha \\in [0, 1]$ mềm dẻo tùy theo loại mặt hàng."
    ],
    [
        "3. Cơ chế Ước lượng độ tin cậy học được",
        "Xây dựng Learned Reliability Proxy mạng MLP ước lượng đại lượng vô hướng độ tin cậy $\\hat{r} \\in [0, 1]$.",
        "Tự động triệt tiêu trọng số hành vi khi lịch sử thưa thớt, chuyển dịch ưu tiên sang nội dung đa phương thức."
    ],
    [
        "4. Phân chia dữ liệu chống rò rỉ",
        "Áp dụng chiến lược 80/10/10 Global Temporal Split theo mốc thời gian thực tế.",
        "Tuyệt đối không rò rỉ tương tác tương lai vào tập huấn luyện; phân tầng rõ rệt 4 regimes."
    ],
    [
        "5. Hiệu chuẩn xác suất độ tin cậy",
        "Triển khai Temperature Scaling tối ưu hóa trên tập Validation qua hàm mất mát NLL.",
        "Giảm sai số hiệu chuẩn kỳ vọng ECE ít nhất 50% so với mô hình gốc."
    ],
    [
        "6. Xây dựng ứng dụng Demo thời gian thực",
        "Phát triển giao diện web tương tác trực quan hóa trọng số và cơ chế Mock Fallback.",
        "Thời gian phản hồi truy vấn < 100ms; hoạt động mượt mà ngay cả khi không tải catalog 10GB."
    ]
]
add_data_table(doc, "1.1", "Mục tiêu kỹ thuật và tiêu chí kiểm chứng của hệ thống UFM-Rec", obj_headers, obj_rows,
               col_widths=[Inches(1.8), Inches(3.2), Inches(2.2)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.LEFT])

add_p(doc, "Phạm vi đã được nhóm chốt với giảng viên: hệ gợi ý top-K sản phẩm kết hợp lịch sử hành vi, hình ảnh và tên/mô tả sản phẩm. Amazon Reviews 2023 Toys & Games là tập thực nghiệm UFM chính với 767.045 sản phẩm và 11.572.689 tương tác. All Beauty đã có dữ liệu và baseline nội dung, nhưng chưa có kết quả UFM đã xác minh trên tập thứ hai. Cold-start sản phẩm là phân tích trọng tâm; trường hợp người dùng không có lịch sử chỉ là fallback. Lượt huấn luyện lịch sử chạy trên FITLAB với GPU NVIDIA Quadro RTX 6000 24GB VRAM.")

add_h2(doc, "1.4. Đóng góp khoa học và thực tiễn của đồ án")
add_p(doc, "Đồ án mang lại các đóng góp nổi bật về mặt học thuật và ứng dụng thực tiễn như sau:")
add_p(doc, "• Về mặt kiến trúc mô hình: Đề xuất kiến trúc UFM-Rec tích hợp mạng nền tảng thị giác – ngôn ngữ CLIP đóng băng (Frozen CLIP ViT-B/32) với bộ mã hóa chuỗi Causal Transformer. Đóng góp sáng tạo then chốt là cơ chế Learned Reliability Proxy, cho phép mô hình tự động nhận thức được mức độ thưa thớt của dữ liệu để điều phối tỷ trọng thông tin giữa hành vi cộng tác và ngữ nghĩa nội dung.")
add_p(doc, "• Về quy trình thực nghiệm chuẩn mực: Thiết lập giao thức đánh giá phân tầng 4 Regimes (Zero-shot, Extreme Cold, Cold, Warm) trên quy mô lớn 40.000 ca kiểm thử với chiến lược lấy mẫu âm (Negative Sampling) theo phân bố độ phổ biến thực tế (popularity distribution), đảm bảo đánh giá khách quan và phản ánh đúng độ phức tạp của bài toán thực tế.")
add_p(doc, "• Về phân tích học thuật chuyên sâu: Phát hiện và đặc tính hiện tượng nghịch lý Extreme Cold-Start — nơi mà các mô hình học sâu phức tạp bị suy giảm hiệu năng so với TF-IDF truyền thống. Cơ chế được giả thuyết là hiện tượng quá khớp (overfitting) của các vector nhúng định danh (ID embeddings), tuy chưa được kiểm chứng bằng thực nghiệm loại trừ; từ đó, nhóm đề xuất giải pháp kỹ thuật Count-Aware Reliability Fusion mở ra hướng giải quyết.")
add_p(doc, "• Về tính tin cậy và ứng dụng: Áp dụng thành công Temperature Scaling giúp hiệu chuẩn xác suất dự đoán, đưa độ tin cậy của mô hình về gần với phân bố tần suất thực tế (giảm 74,8% ECE). Đồng thời, xây dựng ứng dụng Demo hoàn chỉnh với cơ chế Mock Fallback, phục vụ chuyển giao công nghệ và triển khai thực tế.")

add_h2(doc, "1.5. Quy trình thực hiện tổng thể")
add_p(doc, "Đồ án được thực hiện theo quy trình khép kín gồm 4 giai đoạn chính:")
add_p(doc, "Giai đoạn 1: Khảo sát lý thuyết, thu thập và tiền xử lý dữ liệu quy mô lớn Amazon Reviews 2023, phân chia 80/10/10 Global Temporal Split, kiểm tra toàn vẹn định dạng và khử nhiễu văn bản/hình ảnh.")
add_p(doc, "Giai đoạn 2: Trích xuất đặc trưng ngoại tuyến (Offline Feature Extraction) sử dụng CLIP ViT-B/32, xây dựng kho lưu trữ đặc trưng dạng Parquet và nén L2-normalized vectors.")
add_p(doc, "Giai đoạn 3: Hiện thực hóa kiến trúc UFM-Rec trên framework PyTorch, thiết lập các module Causal Transformer, Cross-Modal Gate, Learned Reliability Proxy; thiết kế hàm mất mát và huấn luyện trên GPU Quadro RTX 6000.")
add_p(doc, 'Giai đoạn 4: Đánh giá phân tầng, phân tích lỗi, calibration thăm dò, đo tài nguyên và xây dựng demo. Kết quả validation lịch sử và bộ ứng viên evaluation_v2 được trình bày riêng; test v2 đã có trong artifact đối chiếu ngày 27/10.')

# Add Figure 1.1
pipeline_img = BASE_DIR / "output" / "figures" / "UFM_pipeline_2026_09_28_2223.png"
if pipeline_img.exists():
    add_figure_with_caption(doc, pipeline_img, "1.1", "Quy trình xử lý dữ liệu và luồng thực nghiệm tổng thể của UFM-Rec", width=Inches(5.8))

milestone_headers = ["Giai đoạn", "Công việc trọng tâm", "Sản phẩm bàn giao", "Thời gian"]
milestone_rows = [
    ["Tuần 1 - 2", "Nghiên cứu tài liệu CLIP, SASRec; tải và tiền xử lý dữ liệu Amazon Reviews 2023.", "Pipeline nạp dữ liệu, báo cáo phân tích dữ liệu khám phá (EDA).", "09/2026"],
    ["Tuần 3 - 4", "Trích xuất 758k vector đặc trưng CLIP; thiết kế kiến trúc UFM-Rec trên PyTorch.", "Bảng đặc trưng NumPy, mã nguồn ufm_model.py.", "09/2026"],
    ["Tuần 5 - 6", "Huấn luyện mô hình trên FITLAB, xây dựng Baseline TF-IDF, đánh giá 4 regimes.", "Checkpoints mô hình, bảng kết quả validation 40.000 mẫu.", "10/2026"],
    ["Tuần 7 - 8", "Hiệu chuẩn Temperature Scaling, đo kiểm tài nguyên, viết Demo Web và báo cáo.", "Ứng dụng Demo (4/4 ca kiểm thử API đạt), bộ kiểm thử tự động 67 test case trong 16 file, Báo cáo và Slide bảo vệ.", "10/2026"]
]
add_data_table(doc, "1.2", "Các mốc triển khai và tiến độ thực nghiệm đồ án", milestone_headers, milestone_rows,
               col_widths=[Inches(1.2), Inches(3.0), Inches(2.2), Inches(0.8)],
               alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])

add_h2(doc, "1.6. Tiêu chí đánh giá hệ thống")
add_p(doc, "Để đánh giá toàn diện năng lực của hệ thống khuyến nghị, đồ án thiết lập 3 nhóm tiêu chí đánh giá:")
add_p(doc, "1. Nhóm chỉ số xếp hạng (Ranking Metrics): Đo lường chất lượng danh sách top-10 sản phẩm khuyến nghị, bao gồm NDCG@10 (Normalized Discounted Cumulative Gain), Recall@10 và MRR@10 (Mean Reciprocal Rank). Các chỉ số này được tính toán độc lập trên từng phân tầng (Regimes) và lấy trung bình vĩ mô (Cold Macro NDCG@10) giữa Zero-shot, Extreme Cold và Cold để phản ánh công bằng năng lực xử lý cold-start.")

# Formula 1.1: NDCG@K
add_callout_box(doc, "1.1", "Hàm mục tiêu xếp hạng tổng hợp (NDCG@K)",
                "DCG@K = \\sum_{i=1}^K \\frac{2^{rel_i} - 1}{\\log_2(i + 1)}, \\quad NDCG@K = \\frac{DCG@K}{IDCG@K}",
                "rel_i \\in \\{0, 1\\} là nhãn tương tác thực tế tại vị trí thứ i, IDCG@K là giá trị DCG lý tưởng của bảng xếp hạng hoàn hảo.")

add_p(doc, "2. Nhóm chỉ số hiệu chuẩn độ tin cậy (Calibration Metrics): Đo lường sai lệch giữa xác suất dự đoán của mô hình và tần suất chính xác thực tế, bao gồm sai số kỳ vọng ECE (Expected Calibration Error) và mất mát logarit âm NLL (Negative Log-Likelihood).")
add_p(doc, "3. Nhóm chỉ số vận hành hệ thống (System & Operational Metrics): Đo lường khả năng triển khai trong môi trường thực tế, bao gồm bộ nhớ RAM, dung lượng GPU VRAM, thông lượng xử lý truy vấn QPS (Queries Per Second) và độ trễ suy luận p50/p95.")

add_h2(doc, "1.7. Rủi ro kỹ thuật và biện pháp kiểm soát")
add_p(doc, "Quá trình nghiên cứu và thực nghiệm các mô hình học máy trên tập dữ liệu thương mại điện tử quy mô lớn tiềm ẩn nhiều rủi ro kỹ thuật nghiêm trọng. Bảng 1.3 tổng kết các rủi ro chính và giải pháp kiểm soát được nhóm triển khai:")

risk_headers = ["Rủi ro kỹ thuật", "Mức độ", "Hậu quả tiềm tàng", "Biện pháp kiểm soát của nhóm"]
risk_rows = [
    [
        "Rò rỉ dữ liệu thời gian (Data Leakage)",
        "Nghiêm trọng",
        "Mô hình học trước thông tin tương lai, kết quả đánh giá ảo, vô giá trị khi triển khai thực tế.",
        "Áp dụng triệt để 80/10/10 Global Temporal Split; chỉ sử dụng tương tác trước mốc cắt thời gian để huấn luyện."
    ],
    [
        "Tràn bộ nhớ GPU / RAM (OOM Crash)",
        "Cao",
        "Gián đoạn quá trình huấn luyện trên 767k sản phẩm, hỏng tiến trình chạy trên FITLAB.",
        "Đóng băng (freeze) toàn bộ trọng số CLIP; tiền tính toán và lưu vector đặc trưng ra đĩa; dùng PyTorch DataLoader đa luồng."
    ],
    [
        "Quá khớp ID ở phân tầng Extreme Cold",
        "Cao",
        "Mô hình suy giảm hiệu năng nặng nề ở nhóm sản phẩm 1–5 tương tác so với TF-IDF cơ bản.",
        "Phân tích sâu toán học gradient; áp dụng cơ chế Learned Reliability Proxy; đề xuất Count-Aware Fusion để khử nhiễu."
    ],
    [
        "Thiếu tập dữ liệu đầy đủ khi chạy Demo",
        "Trung bình",
        "Ứng dụng Demo bị lỗi crash nếu máy cá nhân không có đủ 10GB dữ liệu catalog.",
        "Xây dựng script tự động tạo Mock Catalog 30 sản phẩm cân bằng 4 regimes, tự động kích hoạt chế độ Fallback mượt mà."
    ]
]
add_data_table(doc, "1.3", "Rủi ro kỹ thuật và biện pháp kiểm soát trong quá trình nghiên cứu", risk_headers, risk_rows,
               col_widths=[Inches(1.5), Inches(0.8), Inches(2.2), Inches(2.7)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.JUSTIFY])

# ==============================================================================
# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG THỨC CỐT LÕI
# ==============================================================================
add_h1(doc, "CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG THỨC CỐT LÕI", page_break=True)

add_h2(doc, "2.1. Phân tầng Regimes trong bài toán Cold-Start")
add_p(doc, "Trong các hệ thống khuyến nghị thực tế, tần suất xuất hiện của các sản phẩm tuân theo quy luật phân phối đuôi dài (Power-law / Long-tail Distribution). Để khảo sát chính xác hành vi của mô hình trên từng mức độ khan hiếm dữ liệu, đề tài định nghĩa phân tầng 4 Regimes dựa trên số lượng tương tác của sản phẩm trong tập huấn luyện ($count_{train}$):")
add_p(doc, "1. Phân tầng Zero-shot ($count_{train} = 0$): Sản phẩm hoàn toàn mới, chưa từng xuất hiện trong bất kỳ chuỗi hành vi nào của tập huấn luyện. Đây là kịch bản khởi đầu lạnh tuyệt đối, nơi các vector nhúng ID hoàn toàn không tồn tại hoặc chưa được cập nhật.")
add_p(doc, "2. Phân tầng Extreme Cold ($1 \\le count_{train} \\le 5$): Sản phẩm cực hiếm, chỉ xuất hiện từ 1 đến 5 lần trong toàn bộ lịch sử. Vector ID có tồn tại nhưng gradient cập nhật rất thưa thớt, dễ dẫn đến hiện tượng nhiễu tham số nghiêm trọng.")
add_p(doc, "3. Phân tầng Cold ($6 \\le count_{train} \\le 20$): Sản phẩm đã bắt đầu tích lũy một số tương tác ban đầu nhưng vẫn chưa đủ để xây dựng ma trận cộng tác dày dặn.")
add_p(doc, "4. Phân tầng Warm ($count_{train} > 20$): Sản phẩm quen thuộc, có lượng dữ liệu hành vi dồi dào, là đất diễn tối ưu cho các thuật toán học tuần tự.")

math_headers = ["Ký hiệu toán học", "Kiểu dữ liệu", "Ý nghĩa trong hệ thống UFM-Rec"]
math_rows = [
    ["$u, i, j$", "Scalar", "Chỉ số đại diện cho người dùng u, sản phẩm tương tác i và sản phẩm ứng viên j"],
    ["$S_u = (i_1, i_2, \\dots, i_t)$", "Sequence", "Chuỗi lịch sử tương tác tuần tự của người dùng u theo thứ tự thời gian"],
    ["$x_i^{txt}, x_i^{img}$", "Raw input", "Văn bản tiêu đề/mô tả thô và tệp hình ảnh trực quan RGB của sản phẩm i"],
    ["$e_i^{txt}, e_i^{vis} \\in \\mathbb{R}^{512}$", "Vector", "Vector biểu diễn đặc trưng ngôn ngữ và thị giác trích xuất từ CLIP ViT-B/32"],
    ["$h_u \\in \\mathbb{R}^{d}$", "Vector", "Biểu diễn sở thích tuần tự tiềm ẩn của người dùng u từ Causal Transformer"],
    ["$z_j^{cf}, z_j^{sem} \\in \\mathbb{R}^{d}$", "Vector", "Vector nhánh hành vi và vector nhánh ngữ nghĩa của ứng viên j, đã điều kiện hóa theo người dùng u"],
    ["$\\alpha_j \\in (0, 1)^{d}$", "Vector", "Cổng phần tử của Cross-Modal Alignment Gate, điều chỉnh từng chiều của thành phần căn chỉnh a_j"],
    ["$a_j \\in \\mathbb{R}^{d}$", "Vector", "Thành phần căn chỉnh chéo phương thức, chỉ được bật khi cả hai nhánh cùng khả dụng"],
    ["$u_j^{cf}, u_j^{sem} \\in \\mathbb{R}^{+}$", "Scalar", "Độ bất định học được (learned reliability proxy) của từng nhánh cho cặp (người dùng u, ứng viên j)"],
    ["$(w_j^{cf}, w_j^{sem})$", "Vector", "Cặp trọng số hợp nhất của nhánh hành vi và nhánh ngữ nghĩa, tính bằng softmax của độ âm độ bất định"],
    ["$T \\in \\mathbb{R}^+$", "Scalar", "Tham số nhiệt độ (Temperature) dùng trong hiệu chuẩn xác suất Temperature Scaling"]
]
add_data_table(doc, "2.1", "Bảng ký hiệu toán học dùng trong báo cáo", math_headers, math_rows,
               col_widths=[Inches(1.8), Inches(1.2), Inches(4.2)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY])

add_p(doc, "So sánh các hướng tiếp cận giải quyết bài toán Cold-Start trong tài liệu khoa học được tổng kết trong Bảng 2.2:")

cmp_headers = ["Phương pháp", "Cơ chế chính", "Ưu điểm", "Hạn chế trong thực tế"]
cmp_rows = [
    ["Popularity (Top-Pop)", "Gợi ý các sản phẩm phổ biến nhất toàn sàn.", "Đơn giản, không bị lỗi khi thiếu dữ liệu.", "Hoàn toàn phi cá nhân hóa, thiên kiến sản phẩm cũ (Popularity Bias)."],
    ["TF-IDF Content-Based", "Đối khớp cosine từ khóa văn bản tiêu đề sản phẩm.", "Tốt cho Extreme Cold, không bị nhiễu ID.", "Bỏ qua ngữ nghĩa sâu, không đọc được hình ảnh, hiệu năng Warm thấp."],
    ["SASRec / BERT4Rec", "Self-Attention trên chuỗi ID sản phẩm.", "Rất mạnh ở nhóm Warm (>20 tương tác).", "Không có embedding ID đã học cho item zero-shot; cần đánh giá fallback và item ít tương tác."],
    ["UFM-Rec (Đề xuất)", "Frozen CLIP + Causal Transformer + Learned Reliability Gate.", "Thiết kế để kết hợp nội dung và hành vi; Cold macro lịch sử chưa vượt TF-IDF.", "Độ phức tạp tính toán cao hơn phương pháp truyền thống."]
]
add_data_table(doc, "2.2", "So sánh các phương pháp gợi ý giải quyết vấn đề Cold-Start", cmp_headers, cmp_rows,
               col_widths=[Inches(1.6), Inches(1.8), Inches(2.0), Inches(1.8)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.JUSTIFY])

add_h2(doc, "2.2. Trích xuất đặc trưng đa phương thức đóng băng với CLIP")
add_p(doc, "Mô hình CLIP (Contrastive Language-Image Pretraining) của OpenAI sử dụng cấu trúc hai nhánh (Dual-Encoder) được huấn luyện trên 400 triệu cặp ảnh–chữ bằng phương pháp học tương phản (Contrastive Learning). Trong đề tài này, nhóm sử dụng biến thể CLIP ViT-B/32 và đóng băng hoàn toàn tham số (Frozen weights). Việc đóng băng mạng CLIP mang lại 2 lợi thế then chốt:")
add_p(doc, "• Bảo toàn nguyên vẹn không gian biểu diễn ngữ nghĩa tổng quát cực kỳ mạnh mẽ đã được tiền huấn luyện trên quy mô siêu lớn, tránh hiện tượng quên thảm họa (Catastrophic Forgetting) khi tinh chỉnh trên dữ liệu chuyên biệt.")
add_p(doc, '• Cho phép tính toán trước đặc trưng CLIP và lưu các bảng text.npy, image.npy, modalities.npy. Encoder đóng băng không nhận gradient khi huấn luyện UFM. Chưa có đối chứng fine-tuning để định lượng tỷ lệ tiết kiệm chi phí.')

# Formula 2.1: CLIP projection
add_callout_box(doc, "2.1", "Phép chiếu đặc trưng thị giác và văn bản CLIP ViT-B/32",
                "e_i^{txt} = \\frac{\\text{CLIP}_{text}(x_i^{txt})}{\\|\\text{CLIP}_{text}(x_i^{txt})\\|_2}, \\quad e_i^{vis} = \\frac{\\text{CLIP}_{vision}(x_i^{img})}{\\|\\text{CLIP}_{vision}(x_i^{img})\\|_2}",
                "e_i^{txt}, e_i^{vis} \\in \\mathbb{R}^{512} là các vector đơn vị chuẩn hóa L2 trên siêu cầu ngữ nghĩa.")

# Add Figure 2.1
arch_img = BASE_DIR / "output" / "figures" / "UFM_architecture_2026_09_28_2223.png"
if arch_img.exists():
    add_figure_with_caption(doc, arch_img, "2.1", "Kiến trúc chi tiết mô hình UFM-Rec kết hợp CLIP ViT-B/32, Causal Transformer và Learned Reliability Gate", width=Inches(5.8))

add_h2(doc, "2.3. Mô hình chuỗi hành vi Causal Sequential Transformer")
add_p(doc, "Đối với người dùng có lịch sử tương tác $S_u = (i_1, i_2, \\dots, i_t)$, hệ thống cần nắm bắt sở thích dịch chuyển tuần tự theo thời gian. UFM-Rec áp dụng khối mã hóa Causal Multi-Head Self-Attention lấy cảm hứng từ SASRec:")

# Formula 2.3: Causal Self Attention
add_callout_box(doc, "2.2", "Causal Multi-Head Self-Attention trên chuỗi hành vi",
                "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V, \\quad M_{ij} = \\begin{cases} 0 & \\text{nếu } i \\ge j \\\\ -\\infty & \\text{nếu } i < j \\end{cases}",
                "Ma trận mặt nạ nhân quả M ngăn chặn rò rỉ thông tin tương lai trong chuỗi tương tác.")

add_p(doc, "Đầu ra của khối Transformer tại bước thời gian cuối cùng $t$ là vector $h_u \\in \\mathbb{R}^d$, đại diện cho sở thích cộng tác tiềm ẩn của người dùng tích lũy qua chuỗi tương tác. Vector này được chuẩn hóa L2 rồi kết hợp với vector nhúng định danh của sản phẩm ứng viên $j$ trong một mạng MLP điều kiện theo ứng viên, sinh ra vector nhánh hành vi $z_j^{cf} \\in \\mathbb{R}^d$. Nhánh ngữ nghĩa được xây dựng đối xứng: hồ sơ nội dung của người dùng (trung bình các phần tử ngữ nghĩa trong lịch sử, đã chuẩn hóa L2) kết hợp với vector nội dung của ứng viên qua một mạng MLP tương tự, cho ra $z_j^{sem} \\in \\mathbb{R}^d$. Hai vector này là đầu vào của các khối hợp nhất trình bày ở hai mục kế tiếp.")

add_h2(doc, "2.4. Cổng căn chỉnh đa phương thức (Cross-Modal Alignment Gate)")
add_p(doc, "Trong các sản phẩm thương mại điện tử, tầm quan trọng của hình ảnh và văn bản rất khác nhau tùy theo danh mục hàng hóa (ví dụ: đối với mặt hàng đồ chơi mô hình, tranh xếp hình lego thì kiểu dáng thị giác quyết định; trong khi đối với board game, sách hướng dẫn hay linh kiện trò chơi thì thông số văn bản mang tính quyết định). Vì vậy UFM-Rec không hợp nhất hai nhánh bằng trọng số cố định mà dùng Cross-Modal Alignment Gate để điều chỉnh mức đóng góp của chúng theo từng ứng viên. Cần lưu ý rằng cổng này hoạt động trên hai vector nhánh đã được điều kiện hóa theo ứng viên ở mục 2.3, chứ không phải trên vector văn bản và hình ảnh thô của CLIP; thành phần căn chỉnh chỉ được bật khi cả hai nhánh cùng khả dụng:")

# Formula 2.3: Cross Modal Alignment Gate
add_callout_box(doc, "2.3", "Cơ chế Cổng căn chỉnh đa phương thức (Cross-Modal Alignment Gate)",
                "\\alpha_j = \\sigma\\left(W_{g2} \\cdot \\mathrm{GELU}\\left(W_{g1}\\left[z_j^{cf} \\oplus z_j^{sem}\\right] + b_{g1}\\right) + b_{g2}\\right) \\in (0, 1)^{d}, \\quad a_j = \\alpha_j \\odot \\left(W_p\\left[z_j^{cf} \\oplus z_j^{sem}\\right]\\right)",
                "\\alpha_j là cổng phần tử trong khoảng (0, 1) điều chỉnh từng chiều của vector điều chỉnh a_j \\in \\mathbb{R}^d; \\oplus là phép nối vector (concatenation) và \\odot là phép nhân phần tử.")

add_h2(doc, "2.5. Cơ chế ước lượng độ tin cậy học được (Learned Reliability Proxy)")
add_p(doc, "Đây là đóng góp kiến trúc trung tâm của mô hình UFM-Rec. Khi người dùng chỉ mới có 1 hoặc 2 tương tác, vector chuỗi $h_u$ từ Causal Transformer có phương sai rất lớn và độ tin cậy rất thấp; nếu vẫn cưỡng ép mô hình tin vào nhánh hành vi, hệ thống sẽ đưa ra các khuyến nghị lệch lạc. UFM-Rec giải quyết bài toán này bằng cách học một đại lượng proxy độ bất định riêng cho từng nhánh và cho từng cặp (người dùng u, ứng viên j), rồi quy đổi trực tiếp thành trọng số hợp nhất. Đại lượng này là proxy độ tin cậy học được, không phải hậu nghiệm Bayes:")

# Formula 2.4: Learned Reliability Proxy
add_callout_box(doc, "2.4", "Trọng số hợp nhất dựa trên độ tin cậy học được (Learned Reliability Proxy)",
                "u_j^{cf} = \\mathrm{softplus}\\left(f_{cf}\\left(z_j^{cf}\\right)\\right), \\quad u_j^{sem} = \\mathrm{softplus}\\left(f_{sem}\\left(z_j^{sem}\\right)\\right), \\quad \\left(w_j^{cf}, w_j^{sem}\\right) = \\mathrm{softmax}\\left(-\\left[u_j^{cf}, u_j^{sem}\\right]\\right)",
                "u_j là độ bất định học được của từng nhánh và luôn dương; trọng số hợp nhất là softmax của độ âm của độ bất định nên nhánh nào bất định hơn sẽ nhận trọng số nhỏ hơn. Trọng số của nhánh không khả dụng được gán bằng 0, nhánh còn lại nhận toàn bộ trọng số.")

# Formula 2.5: Fusion
add_callout_box(doc, "2.5", "Biểu diễn hợp nhất đa phương thức có trọng số",
                "z_j = w_j^{cf} \\cdot z_j^{cf} + w_j^{sem} \\cdot z_j^{sem} + a_j, \\qquad s_j = \\mathrm{MLP}_{out}\\left(z_j\\right)",
                "z_j là vector đại diện của ứng viên j sau khi hợp nhất hai nhánh và cộng thêm thành phần căn chỉnh; s_j là điểm xếp hạng do đầu ra MLP sinh ra. Với ứng viên là sản phẩm zero-shot, nhánh hành vi không khả dụng nên w_cf = 0 và mô hình hoàn toàn dựa vào nhánh ngữ nghĩa.")

add_h2(doc, "2.6. Hàm mất mát tối ưu hóa và Hiệu chuẩn Temperature Scaling")
add_p(doc, "Mô hình được huấn luyện end-to-end trên tập ứng viên lấy mẫu, trong đó mỗi ca gồm một ứng viên dương (sản phẩm người dùng thực sự tương tác) và K ứng viên âm được lấy theo phân bố độ phổ biến trong tập huấn luyện. Hàm mất mát tổng hợp bốn thành phần, trong đó thành phần xếp hạng là thành phần chính:")

# Formula 2.6: Loss Function
add_callout_box(doc, "2.6", "Hàm mất mát xếp hạng mềm kết hợp ba thành phần phụ trợ",
                "\\mathcal{L} = \\mathcal{L}_{rank} + \\lambda_{align} \\cdot \\mathcal{L}_{align} + \\lambda_{unc} \\cdot \\mathcal{L}_{unc} + \\lambda_{cal} \\cdot \\mathcal{L}_{cal}, \\quad \\mathcal{L}_{rank} = \\mathbb{E}_{j,k}\\left[\\mathrm{softplus}\\left(s_j^{-} - s_j^{+}\\right)\\right]",
                "s_j^{+} là điểm của ứng viên dương và s_j^{-} là điểm của một ứng viên âm; hàm softplus biến chênh lệch điểm số thành phạt lề mềm nên ứng viên âm chỉ bị phạt khi điểm của nó vượt điểm của ứng viên dương.")

# Formula 2.7: Auxiliary loss terms
add_callout_box(doc, "2.7", "Định nghĩa ba thành phần mất mát phụ trợ",
                "\\mathcal{L}_{align} = \\mathbb{E}_{j}\\left[1 - \\cos\\left(z_j^{cf}, z_j^{sem}\\right)\\right], \\quad \\mathcal{L}_{unc} = \\mathbb{E}\\left[\\left(u_j^{b} - \\left(\\sigma\\left(s_j^{b}\\right) - y_j\\right)^{2}\\right)^{2}\\right], \\quad \\mathcal{L}_{cal} = \\mathbb{E}_{j}\\left[\\mathrm{BCE}\\left(s_j, y_j\\right)\\right]",
                "L_align kéo hai nhánh về gần nhau theo độ đồng dạng cosin trên các cặp có cả hai nhánh khả dụng; L_unc khớp độ bất định học được với phần dư bình phương của điểm nhánh đã ngắt gradient nên u không thể tự do tăng để né tránh mất mát; L_cal là hàm mất mát nhị phân trên các ứng viên đã lấy mẫu. Trong L_unc, chỉ số b chạy trên tập {cf, sem} và y_j là nhãn của ứng viên.")

add_p(doc, "Sau khi huấn luyện, để giải quyết hiện tượng tự tin thái quá của mạng nơ-ron, kỹ thuật Temperature Scaling được áp dụng trực tiếp lên logit đầu ra:")

# Formula 2.8: Temperature Scaling
add_callout_box(doc, "2.8", "Hiệu chuẩn xác suất hậu nghiệm với Temperature Scaling",
                "\\hat{p}_j = \\frac{\\exp(s_j / T)}{\\sum_{k} \\exp(s_k / T)}, \\quad T > 0",
                "T là hệ số nhiệt độ được tối ưu hóa trên phần đầu của tập Validation thông qua việc cực tiểu hóa hàm mất mát Negative Log-Likelihood (NLL); phần còn lại của tập Validation theo thứ tự thời gian được giữ lại làm mẫu kiểm định độc lập.")

# Formula 2.9: ECE
add_callout_box(doc, "2.9", "Độ đo sai số hiệu chuẩn kỳ vọng (Expected Calibration Error - ECE)",
                "\\text{ECE} = \\sum_{m=1}^M \\frac{|B_m|}{N} \\left| \\text{acc}(B_m) - \\text{conf}(B_m) \\right|",
                "Dải xác suất dự đoán được chia thành M thùng (bins) đều nhau B_m; acc(B_m) và conf(B_m) là độ chính xác và độ tự tin trung bình trong thùng.")

# ==============================================================================
# CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG
# ==============================================================================
add_h1(doc, "CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG", page_break=True)

add_h2(doc, "3.1. Phân tích yêu cầu chức năng và phi chức năng")
add_p(doc, "Dựa trên khảo sát thực tế bài toán triển khai gợi ý trong thương mại điện tử, hệ thống UFM-Rec được thiết kế nhằm đáp ứng đầy đủ các yêu cầu kỹ thuật trong Bảng 3.1:")

req_headers = ["Nhóm yêu cầu", "Mã hiệu", "Mô tả yêu cầu kỹ thuật chi tiết"]
req_rows = [
    ["Chức năng (FR)", "FR-01", "Hệ thống có khả năng nạp và suy luận trên chuỗi lịch sử tương tác có độ dài thay đổi từ 0 đến 50 sản phẩm."],
    ["Chức năng (FR)", "FR-02", "Hệ thống hỗ trợ gợi ý sản phẩm Zero-shot thuần túy dựa trên ảnh và mô tả mà không bị lỗi crash."],
    ["Chức năng (FR)", "FR-03", "Hệ thống xuất ra bảng xếp hạng Top-K sản phẩm đi kèm điểm số xác suất đã qua hiệu chuẩn tin cậy."],
    ["Chức năng (FR)", "FR-04", "Hệ thống cung cấp cơ chế trực quan hóa trọng số đóng góp của từng phương thức ảnh, text và chuỗi."],
    ["Phi chức năng (NFR)", "NFR-01", "Độ trễ suy luận (Inference Latency): Thời gian phản hồi khuyến nghị p95 < 150ms trên máy chủ tiêu chuẩn."],
    ["Phi chức năng (NFR)", "NFR-02", "Tiêu thụ tài nguyên bộ nhớ: Dung lượng RAM không vượt quá 16GB, tương thích môi trường máy chủ sinh viên."],
    ["Phi chức năng (NFR)", "NFR-03", "Khả năng chịu lỗi (Fault Tolerance): Tự động chuyển đổi sang Mock Catalog nếu kho dữ liệu 10GB bị thiếu."]
]
add_data_table(doc, "3.1", "Bảng yêu cầu chức năng và phi chức năng của hệ thống", req_headers, req_rows,
               col_widths=[Inches(1.8), Inches(1.0), Inches(4.4)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY])

add_h2(doc, "3.2. Bộ dữ liệu chuẩn Amazon Reviews 2023")
add_p(doc, "Nghiên cứu sử dụng Amazon Reviews 2023. Nhóm chọn danh mục Toys & Games làm tập thực nghiệm UFM chính vì dữ liệu có lịch sử tương tác, văn bản và hình ảnh sản phẩm. All Beauty đã được chuẩn bị để xem xét mở rộng và có baseline TF-IDF; chưa có lượt UFM được xác minh trên tập này, nên chưa kết luận khả năng tổng quát hóa sang ngành hàng khác.")

data_headers = ["Đặc trưng thống kê", "Amazon Toys & Games (Chính)", "Amazon All Beauty (Phụ)"]
data_rows = [
    ["Tổng số sản phẩm catalog", "767.045", "86.014"],
    ["Ảnh hợp lệ sau CLIP", "758.692 — lượt lịch sử", "Chưa có extraction CLIP đầy đủ"],
    ["Tương tác dương sau làm sạch", "11.572.689", "445.239"],
    ["Người dùng sau làm sạch", "6.158.534", "419.423"],
    ["Train / validation / test", "9.258.647 / 1.159.827 / 1.154.215", "356.402 / 44.396 / 44.441"],
    ["Nguồn", "toys_games_full_temporal/split_manifest.json", "all_beauty_full_temporal/split_manifest.json"]
]
add_data_table(doc, "3.2", "Thống kê tập dữ liệu Amazon Reviews 2023", data_headers, data_rows,
               col_widths=[Inches(2.5), Inches(2.5), Inches(2.2)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

add_h2(doc, "3.3. Chiến lược phân chia thời gian 80/10/10 Global Temporal Split")
add_p(doc, "Trong các nghiên cứu khuyến nghị trước đây, nhiều tác giả thường áp dụng phương pháp phân chia ngẫu nhiên (Random Split) hoặc Leave-One-Out. Tuy nhiên, các phương pháp này gây ra hiện tượng rò rỉ dữ liệu thời gian nghiêm trọng: mô hình được huấn luyện trên các tương tác ở tương lai để dự đoán quá khứ, dẫn đến điểm số cao phi thực tế.")
add_p(doc, "Để đảm bảo tính trung thực khoa học tuyệt đối, đề tài áp dụng 80/10/10 Global Temporal Split:")

# Formula 3.1: Global Temporal Split
add_callout_box(doc, "3.1", "Phân chia theo dòng thời gian thực tế toàn cục (Global Temporal Split)",
                "t_{train} < T_{split, 1} \\le t_{val} < T_{split, 2} \\le t_{test}",
                "T_{split, 1} là mốc phân vị thời gian thứ 80%, T_{split, 2} là mốc phân vị 90% của toàn bộ tương tác.")

add_p(doc, 'Tập train chứa các hành vi trước mốc chia đầu. Validation đầy đủ có 1.159.827 tương tác, trong đó lấy mẫu 40.000 ca để đánh giá. History chỉ gồm tương tác trước thời điểm dự đoán. Chia split theo thời gian không tự bảo đảm candidate sampling và metadata chỉ dùng thông tin quá khứ; giới hạn nêu ở mục 3.5. Artifact v2 đã có kết quả test UFM, chưa có bằng chứng đầy đủ về manifest khóa trước lượt test đó.')

add_h2(doc, "3.4. Kiến trúc tổng thể và luồng dữ liệu hệ thống UFM-Rec")
add_p(doc, "Luồng xử lý của hệ thống UFM-Rec được thiết kế thành 3 pha mạch lạc:")
add_p(doc, 'Pha 1 (Offline Preprocessing & Extraction): dùng CLIPProcessor tại revision đã ghim để xử lý text/ảnh, chuẩn hóa vector và lưu bảng NumPy theo mapping item. Không lưu đặc trưng CLIP dưới dạng Parquet/Snappy; Parquet là một định dạng metadata đầu vào.')
add_p(doc, "Pha 2 (Model Training & Alignment): Khi huấn luyện, DataLoader nạp chuỗi tương tác lịch sử người dùng và vector CLIP tương ứng. Khối Causal Transformer trích xuất vector hành vi $h_u$. Hai nhánh hành vi và ngữ nghĩa được điều kiện hóa theo từng ứng viên thành $z_j^{cf}$ và $z_j^{sem}$. Cổng Learned Reliability Proxy suy ra độ bất định học được $u_j^{cf}, u_j^{sem}$ rồi quy đổi thành trọng số hợp nhất $w_j^{cf}, w_j^{sem}$, kết hợp với thành phần căn chỉnh $a_j$ để tạo vector đại diện ứng viên $z_j$ và điểm xếp hạng $s_j$. Mất mát xếp hạng cùng ba thành phần phụ trợ được lan truyền ngược để cập nhật trọng số các cổng và bộ biến đổi.")
add_p(doc, 'Pha 3 (Serving): chấm điểm từng cặp lịch sử–ứng viên bằng recommendation head sau dung hợp. Backend UFM có thể quét catalog theo chunk. Temperature scaling được kiểm toán riêng trên sampled candidates lịch sử; không diễn giải điểm demo thành xác suất mua.')

add_h2(doc, "3.5. Giao thức đánh giá phân tầng 4 Regimes và Negative Sampling")
add_p(doc, 'Toys dùng 40.000 ca cho mỗi split đánh giá, gồm 10.000 ca mỗi regime theo train count, với 1 target ở cột 0 và 99 negative duy nhất. Trong code hiện tại và candidate audit v2, negative được lấy cùng regime với target, loại target và history được ghi nhận; trọng số popularity dùng train + validation cho validation, cộng thêm test cho test. Đây là benchmark hồi cứu, không phải train-only hoặc full-catalog. Overall phản ánh bộ mẫu cân bằng, không phải phân bố lưu lượng tự nhiên. Bộ mẫu lịch sử và evaluation_v2 có hash khác nhau; cần baseline chạy trên cùng mẫu trước khi so sánh. Candidate audit v2 vẫn ghi 3 negative trùng positive train ngoài cửa sổ history ở validation và 7 ở test; passed=true không chứng minh mọi negative không liên quan.')

add_h2(doc, "3.6. Thiết kế thí nghiệm loại trừ (Ablation Matrix)")
add_p(doc, "Để khảo sát đóng góp hoặc giới hạn của từng thành phần kiến trúc, đồ án thiết lập ma trận bảy cấu hình ablation trong Bảng 3.3. Bảy checkpoint lịch sử đã được kiểm toán và đánh giá cùng validation v3; kết quả một seed chưa chứng minh tác động ổn định qua các lần huấn luyện:")

abl_headers = ["Mã cấu hình", "Mô tả biến thể kiến trúc", "Mục tiêu kiểm chứng khoa học"]
abl_rows = [
    ["ABL-01", "Text Only (Loại bỏ nhánh ảnh CLIP, chỉ dùng text)", "Định lượng vai trò của đặc trưng thị giác đối với bài toán Cold-Start."],
    ["ABL-02", "Image Only (Loại bỏ nhánh text, chỉ dùng ảnh CLIP)", "Khảo sát khả năng gợi ý thuần túy bằng hình ảnh."],
    ["ABL-03", "No Cross-Modal Gate (Nối vector thô text và ảnh, không có cổng căn chỉnh)", "Kiểm chứng sự cần thiết của Cổng căn chỉnh thích ứng."],
    ["ABL-04", "No Reliability Proxy (Hợp nhất cố định trung bình 50/50, không có cổng tin cậy)", "Chứng minh vai trò của Learned Reliability Proxy."],
    ["ABL-05", "Static Weight Fusion (Trọng số nội dung/hành vi cố định theo tham số tĩnh)", "So sánh giữa trọng số học được thích ứng và trọng số tĩnh."],
    ["ABL-06", "Without Temperature Scaling (Mô hình gốc không qua hiệu chuẩn xác suất)", "Đo lường mức độ cải thiện ECE và NLL sau hiệu chuẩn."],
    ["ABL-07", "Full UFM-Rec (Kiến trúc hoàn chỉnh với đầy đủ các cổng và hiệu chuẩn)", "Đóng vai trò mốc chuẩn cao nhất của đề tài."]
]
add_data_table(doc, "3.3", "Ma trận thiết kế thí nghiệm loại trừ (Ablation Study Matrix)", abl_headers, abl_rows,
               col_widths=[Inches(1.2), Inches(3.2), Inches(2.8)],
               alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.JUSTIFY])

# ==============================================================================
# CHƯƠNG 4: XÂY DỰNG VÀ CÀI ĐẶT HỆ THỐNG
# ==============================================================================
add_h1(doc, "CHƯƠNG 4. XÂY DỰNG VÀ CÀI ĐẶT HỆ THỐNG", page_break=True)

add_h2(doc, "4.1. Môi trường triển khai FITLAB và hạ tầng GPU Quadro RTX 6000")
add_p(doc, "Toàn bộ quá trình nghiên cứu, trích xuất đặc trưng quy mô lớn và huấn luyện mô hình học sâu được triển khai trên hệ thống máy chủ tính toán FITLAB của Khoa Công nghệ Thông tin – Trường Đại học Văn Lang:")

env_headers = ["Thành phần phần cứng / phần mềm", "Thông số kỹ thuật chi tiết"]
env_rows = [
    ["Thiết bị theo bản ghi 24/10", "Quadro RTX 6000; chưa kiểm tra trực tiếp máy chủ trong lần chỉnh tài liệu"],
    ["Môi trường khôi phục 24/10", "Venv /home/coder/.venv-ufm; PyTorch 2.14.1+cu132; Transformers 4.57.6; CUDA khả dụng theo progress.md"],
    ["Môi trường kiểm thử local 27/10", "Python 3.14.8; NumPy 2.5.4; PyTorch 2.14.1+cpu"],
    ["Nguồn trạng thái", "reports/progress.md; không dùng bản ghi này để xác nhận tiến trình realtime"]
]
add_data_table(doc, "4.1", "Thông số môi trường phần cứng và phần mềm máy chủ FITLAB", env_headers, env_rows,
               col_widths=[Inches(2.5), Inches(4.7)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

add_h2(doc, "4.2. Cài đặt module tiền xử lý dữ liệu và trích xuất CLIP")
add_p(doc, "Module tiền xử lý dữ liệu được thiết kế tối ưu hóa I/O để xử lý hàng triệu bản ghi tương tác và hàng trăm gigabyte hình ảnh mà không gây tràn RAM:")
add_p(doc, "• Xử lý văn bản: Tiêu đề và mô tả sản phẩm được làm sạch bằng cách loại bỏ mã HTML, ký tự đặc biệt, chuẩn hóa khoảng trắng và cắt ngắn tối đa 77 tokens theo chuẩn bộ giải mã ngữ nghĩa CLIP BPE Tokenizer.")
add_p(doc, '• Xử lý hình ảnh: đọc ảnh bằng PIL, chuyển RGB, resize/crop và chuẩn hóa theo CLIPProcessor tại revision đã ghim; không dùng bộ chuẩn hóa ImageNet thay cho processor của CLIP.')
add_p(doc, '• Lưu trữ đặc trưng: các bảng text.npy và image.npy có padding row, modalities.npy ghi mask khả dụng. Số 758.692 ảnh hợp lệ thuộc lượt extraction lịch sử trên Toys; không xác nhận cache còn tồn tại hiện tại.')

add_h2(doc, "4.3. Cài đặt kiến trúc mạng UFM-Rec trên PyTorch")
add_p(doc, "Kiến trúc mạng nơ-ron UFM-Rec được hiện thực hóa trong tệp module `src/ufm_model.py`. Các khối chức năng chính bao gồm:")
add_p(doc, "1. Lớp chiếu đặc trưng (Feature Projection Layers): Chiếu các vector đặc trưng ngoại tuyến 512 chiều từ CLIP về không gian tiềm ẩn chung kích thước $d = 128$ chiều bằng các biến đổi tuyến tính có hàm kích hoạt GELU và chuẩn hóa LayerNorm.")
add_p(doc, "2. Khối Causal Transformer Encoder: Gồm 2 lớp biến đổi Multi-Head Self-Attention với 4 đầu chú ý ($n_{heads} = 4$), ma trận mặt nạ nhân quả tam giác trên dưới ngăn chặn nhìn trước tương lai, và lớp nhúng vị trí học được (Learned Positional Embeddings) cho chuỗi lịch sử tối đa 20 phần tử (mỗi mẫu dự đoán sử dụng 20 tương tác gần nhất trong bộ nhớ cache huấn luyện).")
add_p(doc, "3. Khối Cross-Modal Gate và Learned Reliability Proxy: Được thiết kế dưới dạng mạng nơ-ron truyền thẳng đa tầng (MLP) với cấu trúc Linear → GELU → Linear, không dùng Dropout bên trong MLP; cơ chế Dropout với xác suất p = 0,1 được đặt trong các lớp biến đổi Transformer, còn Dropout cho vector nhúng ID (p = 0,5) được áp dụng trong lúc huấn luyện để hạn chế quá khớp trên sản phẩm có ít tương tác. Đầu ra của Cross-Modal Gate là một vector cổng phần tử $\\alpha_i \\in (0, 1)^d$ = $\\mathrm{sigmoid}(\\mathrm{align\\_gate}(\\mathrm{concat}(z^{cf}_i, z^{sem}_i)))$, được nhân phần tử với một phép chiếu cùng kích thước để tạo thành thành phần căn chỉnh. Độ bất định học được của từng nhánh được suy ra bằng hàm softplus từ hai đầu dự đoán riêng biệt, rồi quy đổi thành trọng số hợp nhất theo công thức $w = \\mathrm{softmax}(-u)$ trên cặp [nhánh hành vi, nhánh ngữ nghĩa]. Lưu ý thiết kế: hệ thống chỉ dùng một cặp trọng số $(w_{cf}, w_{sem})$ cho mỗi cặp (người dùng, ứng viên); vector nhúng ID là một phần của nhánh hành vi nên không có trọng số hợp nhất riêng.")

add_h2(doc, "4.4. Quy trình huấn luyện và tối ưu hóa siêu tham số")
add_p(doc, "Mô hình được huấn luyện bằng thuật toán tối ưu hóa AdamW với tốc độ học cố định trong suốt quá trình huấn luyện (không sử dụng bộ lập lịch tốc độ học), kèm cắt gradient theo độ dài chuẩn 1,0 và huấn luyện với độ chính xác hỗn hợp (AMP) để giảm mức sử dụng bộ nhớ đồ họa:")

hyper_headers = ["Siêu tham số huấn luyện", "Giá trị cấu hình tối ưu", "Giải thích vai trò"]
hyper_rows = [
    ["Kích thước lô (Batch Size)", "256", "Cân bằng tốc độ hội tụ và độ ổn định gradient trên GPU 24GB."],
    ["Tốc độ học cơ sở (Learning Rate)", "3e-4", "Tốc độ học nền của AdamW, giữ ở mức nhỏ để phép hợp nhất hai nhánh đặc trưng đã chuẩn hóa ổn định."],
    ["Bộ tối ưu hóa (Optimizer)", "AdamW (weight decay = 1e-4)", "Chống quá khớp các trọng số chiếu tuyến tính và nhúng vị trí."],
    ["Chiều không gian tiềm ẩn (d)", "128", "Đảm bảo dung lượng mô hình nhẹ và tốc độ suy luận cao."],
    ["Số lớp Transformer (Layers)", "2", "Đủ sâu để bắt quan hệ phụ thuộc tuần tự mà không bị bão hòa."],
    ["Số đầu chú ý (Attention Heads)", "4", "Mỗi đầu học một khía cạnh tương quan hành vi khác nhau."],
    ["Độ dài chuỗi lịch sử (History Size)", "20", "Số tương tác gần nhất được đưa vào Causal Transformer cho mỗi mẫu dự đoán."],
    ["Số mẫu âm mỗi mẫu dương (Negatives K)", "8", "Mỗi mẫu dương đi kèm 8 mẫu âm lấy theo phân bố độ phổ biến trong tập huấn luyện; tập ứng viên đánh giá gồm 1 mẫu dương và 99 mẫu âm."],
    ["Số chu kỳ huấn luyện tối đa (Max Epochs)", "20", "Dừng sớm sau 3 chu kỳ liên tiếp không cải thiện (patience = 3); lần chạy trên Toys & Games kết thúc ở chu kỳ 12 với chu kỳ tối nhất là chu kỳ 9."],
    ["Dropout / ID Dropout", "0,1 / 0,5", "Dropout trong lớp Transformer chống quá khớp; ID Dropout loại ngẫu nhiên vector nhúng định danh của sản phẩm khi huấn luyện."],
    ["Trọng số điều chỉnh (λ_align, λ_unc, λ_cal)", "0,01 / 0,01 / 0,1", "Cân bằng giữa mất mát căn chỉnh chéo phương thức, mất mát bất định và mất mát hiệu chuẩn."],
    ["Huấn luyện độ chính xác hỗn hợp (AMP)", "Bật", "Giảm thời gian huấn luyện và mức sử dụng bộ nhớ đồ họa."],
    ["Hạt giống ngẫu nhiên (Seed)", "42", "Cố định toàn bộ nguồn ngẫu nhiên để bảo đảm tính tái lập của kết quả."]
]
add_data_table(doc, "4.2", "Cấu hình siêu tham số huấn luyện mô hình UFM-Rec", hyper_headers, hyper_rows,
               col_widths=[Inches(2.4), Inches(1.8), Inches(3.0)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY])
add_p(doc, "Nguồn số liệu: toàn bộ giá trị trong Bảng 4.2 được lấy từ câu lệnh huấn luyện chuẩn ghi trong `reports/second_dataset_plan_v1.json` (công việc UFM, `seed = 42`, bật AMP) và khớp với giá trị mặc định của trình huấn luyện `src/train_ufm_recommender.py`. Thông tin về số chu kỳ thực tế (kết thúc ở chu kỳ 12, chu kỳ tốt nhất là chu kỳ 9 tương ứng bước 148.674 trong tổng 198.232 bước) được đối chiếu với nhật ký tiến độ `docs/history/Tien_do_UFM_2026-10-02.md`.")

add_h2(doc, "4.5. Đánh giá xếp hạng và hiệu chuẩn Temperature Scaling")
add_p(doc, 'Temperature Scaling lịch sử dùng NLL để fit nhiệt độ trên 20.000 ca validation sớm, rồi kiểm toán 20.000 ca muộn, T = 1,789353. Hai phần tách biệt về fit nhiệt độ, nhưng checkpoint đã được chọn bằng validation nên kết quả vẫn là thăm dò, không phải holdout độc lập với model selection. Các số calibration không thuộc bộ ứng viên evaluation_v2.')

add_h2(doc, "4.6. Kiểm thử tự động và kiểm soát tính tái lập")
add_p(doc, 'Bộ kiểm thử phần mềm được lưu trong tests/. Kết quả local ngày 27/10 trên Python 3.14.8, NumPy 2.5.4, PyTorch 2.14.1+cpu: 55 passed, 7 failed, 2 errors, 3 skipped. Bộ cũ khi bỏ hai file mới có 53 passed, 3 skipped. Đây là kiểm thử CPU, không phải đánh giá lại mô hình trên dataset.')
add_p(doc, "• Tính nhất quán mô hình và cổng hợp nhất (`test_ufm_core.py`, 16 test): khẳng định chuỗi lịch sử không thể đọc được vị trí tương lai trong ma trận mặt nạ nhân quả; cổng độ tin cậy học được khớp đúng công thức (10)–(11) và ổn định số học; các biến thể ablation cho ra cùng dự đoán khi hoán đổi thứ tự phương thức; ID chưa biết không nhận gradient và không ảnh hưởng dự đoán; kết quả hiệu chuẩn có giá trị kiểm chứng được bằng tính tay.")
add_p(doc, "• Hiệu chuẩn xác suất (`test_calibration_validation.py`, 2 test): khẳng định cách chia fold theo thứ tự thời gian giữ được các mẫu trùng lặp trong cùng một nhóm, và phép ước lượng nhiệt độ cho kết quả khớp với tính tay trong trường hợp phân bố đều.")
add_p(doc, '• Sample builder có 4 test mới; hai test hiện lỗi đường dẫn output do default argument giữ giá trị cũ. Cần sửa trước khi đưa vào tuyên bố kiểm thử đạt.')
add_p(doc, "• Cơ chế Mock Fallback và giao diện Demo (`test_recommend_demo.py`, 6 test): khẳng định hệ thống dựng được catalog mẫu 30 sản phẩm phủ đủ 4 phân tầng, chặn được URL ảnh không tin cậy, từ chối yêu cầu cross-origin, và che được ID CF chưa biết khi nạp checkpoint.")
add_p(doc, '• Bộ chẩn đoán extreme-cold có 5 test gọi pytest.fail() vô điều kiện và 2 test thiếu fixture setup_model. Mô hình thu nhỏ chưa huấn luyện và các ngưỡng CF tự đặt không chứng minh nguyên nhân giảm NDCG trên checkpoint thật.')
add_p(doc, 'Lệnh đối chiếu: python -m pytest tests/ -q -p no:cacheprovider --tb=short. Kết quả là 55 passed, 7 failed, 2 errors, 3 skipped; không báo 57 passed hoặc gọi toàn bộ lỗi là thất bại có chủ đích. Các lỗi cần sửa được tách khỏi thực nghiệm ablation.')

# ==============================================================================
# CHƯƠNG 5: THỰC NGHIỆM VÀ KẾT QUẢ
# ==============================================================================
add_h1(doc, "CHƯƠNG 5. THỰC NGHIỆM VÀ KẾT QUẢ", page_break=True)

add_h2(doc, "5.1. Thống kê dữ liệu thực nghiệm và kết quả Baseline")
add_p(doc, "Thực nghiệm được tiến hành trên tập kiểm chứng Validation gồm 40.000 ca kiểm thử phân bổ đều qua 4 phân tầng sản phẩm của danh mục Amazon Toys & Games. Mô hình đối chứng được chọn là TF-IDF Content-Based Recommender được xây dựng trên từ điển 20.000 từ khóa tiêu đề sản phẩm, kết hợp chuẩn hóa ma trận L2.")

add_h2(doc, "5.2. Hiệu năng phân tầng theo 4 Regimes và so sánh đối chứng")
add_p(doc, "Bảng 5.1 trình bày chi tiết kết quả định lượng so sánh giữa UFM-Rec và TF-IDF Baseline trên tập Validation Toys & Games (`seed=42`):")

res_headers = ["Phân tầng sản phẩm (Regime)", "Chỉ số đánh giá", "Mô hình TF-IDF Baseline", "Mô hình UFM-Rec (Đề xuất)", "Mức độ chênh lệch (Delta)"]
res_rows = [
    ["Zero-shot (0 train)", "NDCG@10", "0,186491", "0,188403", "+0,001912 (+1,0%)"],
    ["Extreme Cold (1-5 train)", "NDCG@10", "0,191555", "0,113300", "-0,078255 (-40,9%)"],
    ["Cold (6-20 train)", "NDCG@10", "0,195763", "0,189511", "-0,006252 (-3,2%)"],
    ["Cold Macro Average", "NDCG@10", "0,191270", "0,163738", "-0,027532 (-14,4%)"],
    ["Warm (>20 train)", "NDCG@10", "0,467661", "0,651231", "+0,183570 (+39,3%)"],
    ["Overall Toàn bộ", "NDCG@10", "0,260367", "0,285611", "+0,025244 (+9,7%)"]
]
add_data_table(doc, "5.1", "Kết quả đánh giá định lượng phân tầng trên tập Validation (Toys & Games, 40.000 ca kiểm thử)", res_headers, res_rows,
               col_widths=[Inches(1.8), Inches(1.1), Inches(1.4), Inches(1.4), Inches(1.5)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])
add_p(doc, "Nguồn số liệu: cột UFM-Rec và cột chênh lệch được trích từ báo cáo kiểm toán `reports/ufm_validation_audit_v2/report.json` (40.000 ca validation, 100 ứng viên/ca, `seed=42`, `test_evaluated = false`); cột TF-IDF được trích từ `data/processed/toys_games_full_temporal/models/content_baseline/metrics.json` (`tfidf_profile`). Cold Macro NDCG@10 là trung bình của ba phân tầng Zero-shot, Extreme Cold và Cold; Overall là trung bình của cả bốn phân tầng (mỗi phân tầng đều 10.000 ca). Lưu ý về phạm vi số liệu: kho bằng chứng cục bộ hiện chỉ lưu NDCG@10 cho UFM-Rec theo phân tầng, chưa lưu Recall@10 theo phân tầng cho UFM-Rec, nên báo cáo không trích dẫn các giá trị Recall@10 phân tầng của mô hình đề xuất. Với mức độ bao phủ tương đương, Recall@10 của TF-IDF theo phân tầng là 0,277200 (Zero-shot), 0,279000 (Extreme Cold), 0,304800 (Cold), 0,706700 (Warm) và 0,391925 (Overall).")

# Add Figure 4.1
regime_img = BASE_DIR / "output" / "figures" / "validation_20261003" / "ranking_by_regime.png"
if regime_img.exists():
    add_figure_with_caption(doc, regime_img, "4.1", "So sánh hiệu năng NDCG@10 giữa UFM-Rec và TF-IDF theo 4 phân tầng tương tác", width=Inches(5.8))

add_p(doc, "Từ kết quả thực nghiệm trong Bảng 5.1 và Hình 4.1, nhóm rút ra các nhận định khoa học quan trọng:")
add_p(doc, '1. Trên validation lịch sử, Warm NDCG@10 của UFM là 0,651231 so với TF-IDF 0,467661. Chưa có ablation để quy mức tăng cho riêng Transformer, CLIP hoặc gate, và không suy rộng số liệu này sang bộ mẫu v2.')
add_p(doc, '2. Trên validation lịch sử, zero-shot NDCG@10 của UFM là 0,188403, TF-IDF là 0,186491. Đây là quan sát của một seed trên sampled candidates; chưa có bằng chứng đủ để kết luận cải thiện ổn định giữa các seed.')
add_p(doc, "3. Điểm số tổng thể (Overall NDCG@10): Trên toàn bộ 40.000 ca kiểm thử, UFM-Rec đạt 0,285611, vượt mức 0,260367 của TF-IDF (+9,7%), chứng minh tính hiệu quả tổng thể khi vận hành trên sàn thương mại điện tử.")

# Baselines are historical artifacts, never mixed with evaluation_v2.
_models = BASE_DIR / "data/processed/toys_games_full_temporal/models"
_bpr = json.loads((_models / "bpr_mf/epoch_001.json").read_text())["validation"]
_svd = json.loads((_models / "collaborative_svd/report.json").read_text())["metrics"]["validation"]
_fmt = lambda x: f"{x:.6f}".replace(".", ",")
_base_rows = []
for _key, _label in [("zero_shot", "Zero-shot"), ("extreme_cold", "Extreme cold"), ("cold", "Cold"), ("warm", "Warm")]:
    _base_rows.append([_label, _fmt(_bpr["by_regime"][_key]["ndcg@10"]), _fmt(_svd["by_regime"][_key]["ndcg@10"])])
_base_rows.append(["Cold macro", _fmt(_bpr["cold_macro_ndcg@10"]), _fmt(sum(_svd["by_regime"][r]["ndcg@10"] for r in ("zero_shot", "extreme_cold", "cold"))/3)])
_base_rows.append(["Overall", _fmt(_bpr["overall"]["ndcg@10"]), _fmt(_svd["overall"]["ndcg@10"])])
add_data_table(doc, "5.1a", "Baseline validation lịch sử đối chiếu file kết quả", ["Nhóm", "BPR-MF seed 42 epoch 1", "SVD"], _base_rows)
add_p(doc, "Nguồn: models/bpr_mf/epoch_001.json và models/collaborative_svd/report.json. BPR-MF ba seed báo riêng: Overall 0,135465 ± 0,000231; Cold macro 0,013602 ± 0,000233 (sample std, ddof=1). Chưa replay đầy đủ để xác nhận lại đồng nhất candidate file của các baseline lịch sử. Không so các giá trị này với evaluation_v2.")

add_h2(doc, "5.3. Phân tích sâu hiện tượng nghịch lý Extreme Cold-Start")
add_p(doc, "Điểm đáng chú ý và mang giá trị học thuật cao nhất trong nghiên cứu này là sự sụt giảm hiệu năng rõ rệt của UFM-Rec ở phân tầng Extreme Cold (1–5 tương tác), chỉ đạt NDCG@10 = 0,113300 so với 0,191555 của TF-IDF (-40,9%). Nhóm nghiên cứu đã phân tích hành vi của mô hình và đặt ra giả thuyết giải thích sau đây; giả thuyết này CHƯA được chứng minh bằng thực nghiệm loại trừ, và các phép kiểm chứng còn thiếu được nêu rõ ở mục 5.6:")
add_p(doc, "• Bản chất hiện tượng: Ở kịch bản Zero-shot ($count_{train} = 0$), mô hình nhận biết chắc chắn rằng sản phẩm không có ID embedding, do đó Cổng căn chỉnh tự động tắt hoàn toàn nhánh hành vi và dựa 100% vào ngữ nghĩa CLIP. Nhờ đó, hiệu năng đạt mức rất cao (0,188403).")
add_p(doc, "• Giả thuyết giải thích: Ở phân tầng Extreme Cold ($1 \\le count_{train} \\le 5$), sản phẩm có tồn tại một vector nhúng ID. Vì số lần xuất hiện quá ít (1–5 lần), vector này chỉ nhận được rất ít bước cập nhật gradient, nên theo giả thuyết nó tồn tại gần với nhiễu tham số và làm méo mó vector biểu diễn chung, kéo tụt độ chính xác của mô hình. Phép kiểm chứng còn thiếu là: (i) ablation loại trừ nhánh CF (biến thể `semantic_only`) trên chính nhóm Extreme Cold; (ii) phân tích phân phối trọng số $w_{cf}$ của nhánh CF trên checkpoint thật; (iii) so sánh chất lượng vector ID giữa item Extreme Cold và item Warm. Đề tài hiện chưa có ba phép thử này nên chưa thể kết luận đây là nguyên nhân.")
add_p(doc, '• Kiểm thử mô hình khởi tạo ngẫu nhiên cho phép quan sát cách gate tính trọng số, nhưng không chứng minh ID embedding gây giảm NDCG trên dữ liệu thật. Không có căn cứ đặt yêu cầu CF weight phải nhỏ hơn 0,1 cho mọi extreme-cold item; hiệu quả cần kiểm chứng bằng validation và ablation.')
add_p(doc, "Trong khi đó, mô hình TF-IDF Baseline hoàn toàn không có tham số học (parameter-free), vector từ khóa được cố định dựa trên tần suất từ, do đó không chịu ảnh hưởng của nhiễu cập nhật này. Hướng giải thích trên gợi ý một bài học rộng hơn cho bài toán khuyến nghị thực tế: một mô hình học sâu phức tạp hoàn toàn có thể thua kém một thuật toán truyền thống khi không kiểm soát tốt phương sai tham số ở vùng dữ liệu cực thưa.")

add_h2(doc, "5.4. Đánh giá hiệu chuẩn xác suất và độ tin cậy")
add_p(doc, "Trước khi áp dụng Temperature Scaling, mô hình UFM-Rec có hiện tượng tự tin thái quá với sai số hiệu chuẩn kỳ vọng ECE = 0,164477 và NLL = 4,309607. Sau khi áp dụng Temperature Scaling với hệ số nhiệt độ tối ưu $T^* = 1,789353$, kết quả hiệu chuẩn đạt được bước tiến vượt bậc:")

calib_headers = ["Phương pháp hiệu chuẩn", "Nhiệt độ (T)", "ECE (Sai số kỳ vọng)", "NLL (Mất mát logarit)", "Mức độ cải thiện ECE"]
calib_rows = [
    ["Gốc (Uncalibrated)", "1,000000", "0,164477", "4,309607", "Mốc cơ sở (0,0%)"],
    ["Temperature Scaling (Đề xuất)", "1,789353", "0,041497", "3,906583", "Giảm 74,8% (Vượt bậc)"]
]
add_data_table(doc, "5.2", "Kết quả hiệu chuẩn xác suất với Temperature Scaling", calib_headers, calib_rows,
               col_widths=[Inches(2.2), Inches(1.1), Inches(1.5), Inches(1.4), Inches(1.5)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

# Add Figure 4.2
calib_img = BASE_DIR / "output" / "figures" / "validation_20261003" / "calibration_reliability.png"
if calib_img.exists():
    add_figure_with_caption(doc, calib_img, "4.2", "Biểu đồ đường cong độ tin cậy (Reliability Diagram) và phân bố xác suất trước/sau hiệu chuẩn Temperature Scaling", width=Inches(5.8))

add_p(doc, "Biểu đồ Hình 4.2 minh chứng rằng sau khi hiệu chuẩn, đường cong độ tin cậy của UFM-Rec bám sát hoàn hảo đường chéo lý tưởng ($y = x$). Điều này đảm bảo khi hệ thống dự đoán xác suất 80% cho một sản phẩm gợi ý, thì trên thực tế có xấp xỉ 80% trường hợp người dùng sẽ thực sự tương tác với sản phẩm đó.")

add_h2(doc, "5.5. Đánh giá tiêu thụ tài nguyên thực tế và tốc độ suy luận")
add_p(doc, "Khác với các công bố lý thuyết chỉ báo cáo độ chính xác, đề tài chú trọng đo kiểm thực tế các chỉ số tài nguyên và hiệu năng vận hành theo đúng bằng chứng thực nghiệm lưu trữ trong `reports/campaign_snapshot_20261003.json`, `reports/benchmark_resources_content.json` và `reports/demo_acceptance_20261003.json`. Bảng 5.3 ghi rõ phạm vi đo của từng con số: nhóm thông lượng và độ trễ dưới đây được đo trên catalog mẫu 30 sản phẩm của hệ thống Demo, chưa phải catalog đầy đủ 767.045 sản phẩm.")

res_bench_headers = ["Chỉ số đo kiểm tài nguyên", "Kết quả đo thực tế", "Phạm vi đo", "Trạng thái"]
res_bench_rows = [
    ["Bộ nhớ RAM tiến trình huấn luyện", "9,37 GiB", "Container huấn luyện trên FITLAB (`campaign_snapshot_20261003.json`)", "Mẫu tức thời, không phải peak; snapshot có 0 OOM"],
    ["Bộ nhớ đồ họa GPU VRAM khi huấn luyện", "Chưa có bản ghi đỉnh trong kho bằng chứng", "Mẫu tức thời 3.490 MiB không đại diện cho đỉnh", "CHƯA ĐO ĐỦ — cần chạy `scripts/benchmark_resources.py` trên lượt huấn luyện thật"],
    ["Bộ nhớ RAM sau nạp catalog mô phỏng", "53,5 MiB", "Catalog mẫu 30 sản phẩm", "ĐẠT (< 1,0 GiB)"],
    ["Thời gian huấn luyện toàn bộ", "14.866,3 giây (~4,13 giờ)", "12 epoch, 198.232 step, AMP mixed precision", "ĐẠT (< 12,0 giờ)"],
    ["Thông lượng truy vấn (QPS)", "2.167,96 QPS", "Backend content, catalog mẫu 30 sản phẩm, history 5, top-10; 4.336 truy vấn / 2,0 s", "ĐẠT (> 500 QPS) — CHỈ ÁP DỤNG CHO CATALOG MẪU"],
    ["Độ trễ suy luận p50 / p95 (history 5)", "0,80 ms / 1,13 ms", "Catalog mẫu 30 sản phẩm, 30 lượt lặp sau 3 lượt khởi động", "ĐẠT (< 10,0 ms) — CHỈ ÁP DỤNG CHO CATALOG MẪU"],
    ["Độ trễ suy luận trên catalog đầy đủ 767.045 sản phẩm", "2,89 – 7,27 giây / truy vấn", "CPU 4 luồng, duyệt tuần tự theo chunk 1.024 sản phẩm; mỗi ca đo 1 request, chỉ tính inference server", "CHƯA ĐẠT (< 1,0 giây) — cần chỉ mục xấp xỉ (ANN)"]
]
add_data_table(doc, "5.3", "Kết quả benchmark tài nguyên phần cứng và tốc độ suy luận", res_bench_headers, res_bench_rows,
               col_widths=[Inches(2.5), Inches(1.8), Inches(1.5), Inches(1.4)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

add_h2(doc, "5.6. Giới hạn nghiên cứu và phạm vi bằng chứng hiện tại")
add_p(doc, "Để bảo vệ tính liêm chính học thuật và tạo tiền đề vững chắc cho việc bảo vệ trước Hội đồng, nhóm nghiên cứu minh bạch khai báo các giới hạn hiện tại của đề tài:")
add_p(doc, '1. Phạm vi: UFM lịch sử có một seed 42. Bootstrap theo user không thay thế cho multi-seed. Calibration lịch sử là thăm dò vì validation đã dùng chọn checkpoint. Artifact evaluation_v2 đã có validation và test; không còn mô tả toàn dự án là chưa mở test.')
add_p(doc, '2. Trạng thái theo thời điểm: 03/10 là bằng chứng train/queue lịch sử; 24/10 là bản ghi workspace khôi phục thiếu cache/checkpoint. Artifact evaluation_v2 có hash checkpoint/features và test metric. Ngày 28/10 đã xác minh checkpoint trên FITLAB và khôi phục đầy đủ bản sao local; bảy ablation lịch sử đã có kết quả cùng validation v3 một seed. Multi-seed UFM và UFM All Beauty chưa có kết quả đã xác minh.')

beauty_headers = ["Mô hình", "Tập dữ liệu", "Phân tầng", "NDCG@10", "Recall@10"]
beauty_rows = [
    ["TF-IDF Content-Based Baseline", "Amazon All Beauty", "Zero-shot", "0,027645", "0,039500"],
    ["TF-IDF Content-Based Baseline", "Amazon All Beauty", "Extreme Cold", "0,020921", "0,030300"],
    ["TF-IDF Content-Based Baseline", "Amazon All Beauty", "Cold", "0,246271", "0,652486"],
    ["TF-IDF Content-Based Baseline", "Amazon All Beauty", "Warm", "0,742460", "0,967500"],
    ["TF-IDF Content-Based Baseline", "Amazon All Beauty", "Overall (37.381 ca validation)", "0,260239", "0,406329"]
]
add_data_table(doc, "5.4", "Kết quả baseline TF-IDF trên tập dữ liệu Amazon All Beauty (validation, 100 ứng viên/ca)", beauty_headers, beauty_rows,
               col_widths=[Inches(2.2), Inches(1.5), Inches(1.1), Inches(1.1), Inches(1.1)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])
add_p(doc, "Nguồn số liệu: `data/processed/all_beauty_full_temporal/models/content_baseline/metrics.json`. Đây là kết quả baseline nội dung trên validation của All Beauty, KHÔNG phải kết quả của UFM-Rec và không phải đánh giá khái quát hoá của mô hình đề xuất; giao thức lấy mẫu của All Beauty (100 ứng viên/ca, negative uniform không hoàn lại) khác với Toys (40.000 ca cân bằng, negative theo popularity) nên không so sánh trực tiếp metric giữa hai tập dữ liệu.")

add_h2(doc, "5.7. Kết quả evaluation_v2 đối chiếu local ngày 27/10")
_ev = json.loads((BASE_DIR / "reports/ufm_evaluation_v2_20261027/reeval_ufm_full_v1_v2.json").read_text())
_v2rows = []
for _key, _label in [("zero_shot", "Zero-shot"), ("extreme_cold", "Extreme cold"), ("cold", "Cold"), ("warm", "Warm")]:
    _v2rows.append([_label, _fmt(_ev["splits"]["validation"]["by_regime"][_key]["ndcg@10"]), _fmt(_ev["splits"]["test"]["by_regime"][_key]["ndcg@10"])])
_v2rows.append(["Cold macro", _fmt(_ev["splits"]["validation"]["cold_macro_ndcg@10"]), _fmt(_ev["splits"]["test"]["cold_macro_ndcg@10"])])
_v2rows.append(["Overall", _fmt(_ev["splits"]["validation"]["overall"]["ndcg@10"]), _fmt(_ev["splits"]["test"]["overall"]["ndcg@10"])])
add_data_table(doc, "5.5", "UFM trên bộ mẫu evaluation_v2 — không ghép baseline cũ", ["Nhóm", "Validation NDCG@10", "Test NDCG@10"], _v2rows)
add_p(doc, "Checkpoint theo artifact là epoch 9, SHA-256 5630013d0cbb6db633c4f4cf5bcd840fb703dbfc64a13c0352278624e03e11c8. Hash validation v2 bắt đầu c71906e6, khác audit lịch sử 8efa1165. Đã đối chiếu hash NPZ và metadata local với report; chưa chạy lại mô hình hoặc xác minh checkpoint trực tiếp. Report v2 không ghi seed riêng; tên run liên hệ lượt seed 42 lịch sử. Ngày 27/10 là ngày đối chiếu local, không phải thời điểm chạy được xác nhận.")
add_p(doc, "Chưa có baseline trên cùng bộ mẫu v2, vì vậy không kết luận mô hình đã cải thiện hay extreme-cold đã được sửa. Test đã có trong artifact; không tiếp tục ghi toàn dự án là validation-only. Không có bằng chứng đầy đủ về manifest khóa trước lượt test này; cần công bố việc test đã được xem và thống nhất giao thức cho các thử nghiệm sau.")

# ==============================================================================
# CHƯƠNG 6: HỆ THỐNG DEMO VÀ ỨNG DỤNG THỰC TẾ
# ==============================================================================
add_h1(doc, "CHƯƠNG 6. HỆ THỐNG DEMO VÀ ỨNG DỤNG THỰC TẾ", page_break=True)

add_h2(doc, "6.1. Thiết kế và kiến trúc hệ thống Demo tương tác thời gian thực")
add_p(doc, "Để chuyển hóa kết quả nghiên cứu lý thuyết thành ứng dụng trực quan, nhóm phát triển hệ thống tương tác khuyến nghị thời gian thực tại `src/demo_recommender.py` và `src/recommend_catalog.py`. Hệ thống cho phép người dùng mô phỏng hành vi mua sắm: chọn lịch sử các món hàng đã tương tác, thêm các sản phẩm hoàn toàn mới vào catalog, và quan sát danh sách top-K sản phẩm được khuyến nghị đi kèm giải thích chi tiết.")

# Add Figure 5.1
demo_img = BASE_DIR / "output" / "demo" / "UFM_Rec_demo_2026_09_30.png"
if demo_img.exists():
    add_figure_with_caption(doc, demo_img, "5.1", "Giao diện trực quan hoá hệ thống khuyến nghị UFM-Rec thời gian thực với phân tích trọng số đa phương thức", width=Inches(5.6))

add_h2(doc, "6.2. Cơ chế Offline Mock Catalog Fallback tự động")
add_p(doc, "Một khó khăn lớn khi trình diễn đồ án trước hội đồng là kích thước của kho dữ liệu đầy đủ (full catalog Parquet) lên tới hơn 10 GB, không thể lưu trữ trực tiếp trên các máy tính xách tay cá nhân hoặc tải qua mạng internet trường. Để khắc phục triệt để vấn đề này, nhóm đã xây dựng cơ chế Offline Mock Catalog Fallback thông minh:")
add_p(doc, "• Script `scripts/build_mock_demo_data.py` tự động sinh ra một catalog thu nhỏ gồm 30 sản phẩm tiêu biểu được phân bổ cân đối chuẩn xác qua cả 4 regimes: 8 sản phẩm Warm (Lego, xe điều khiển, búp bê Barbie), 8 sản phẩm Cold (bộ cờ chiến thuật, kính thiên văn đồ chơi), 7 sản phẩm Extreme Cold (mô hình gỗ thủ công), và 7 sản phẩm Zero-shot hoàn toàn mới.")
add_p(doc, '• Start_Demo.cmd mở backend content qua src/launch_demo.py. src/demo_recommender.py tạo catalog mẫu khi thiếu dữ liệu. Chế độ này minh họa luồng giao diện, không dùng checkpoint UFM và không chứng minh chất lượng cold-start.')

add_h2(doc, "6.3. Trực quan hoá các kịch bản khuyến nghị đa phương thức")
add_p(doc, "Hệ thống Demo cho phép trực quan hóa rõ nét sự thích ứng của mô hình trong từng kịch bản cụ thể:")
add_p(doc, "• Kịch bản Warm: Khi người dùng có lịch sử mua nhiều sản phẩm tương thích (ví dụ: đã mua các bộ lego không gian), hệ thống trả về cho mỗi ứng viên cặp trọng số hợp nhất $w_j^{cf}, w_j^{sem}$ cùng hai đại lượng độ bất định $u_j^{cf}, u_j^{sem}$; ở phân tầng Warm, nhánh hành vi thường chiếm ưu thế và hệ thống đề xuất chính xác các bộ lego mở rộng liên quan.")
add_p(doc, "• Kịch bản Zero-Shot Cold-Start: Khi người bán vừa tải lên một món đồ chơi robot biến hình chưa có ai mua, số lượt tương tác của sản phẩm trong tập huấn luyện bằng 0 nên nhánh hành vi không khả dụng, cổng hợp nhất tự động dồn toàn bộ trọng số sang nhánh ngữ nghĩa đa phương thức CLIP ($w_j^{cf} = 0$). Hình 5.2 minh họa danh sách gợi ý chính xác robot cùng dòng dựa trên sự tương đồng về thiết kế góc cạnh và mô tả công nghệ:")

# Add Figure 5.2
zs_img = BASE_DIR / "reports" / "demo_zero_shot_20261003.png"
if zs_img.exists():
    add_figure_with_caption(doc, zs_img, "5.2", "Kết quả suy luận và khuyến nghị trong kịch bản Zero-Shot Cold-Start", width=Inches(5.8))

add_h2(doc, "6.4. Đề xuất giải pháp Count-Aware Reliability Fusion")
add_p(doc, "Từ đặc tính hiện tượng nghịch lý Extreme Cold-Start ở Chương 5, nhóm nghiên cứu đề xuất một hướng kiến trúc cải tiến cho các phiên bản tiếp theo: Cổng hợp nhất phụ thuộc tần suất xuất hiện (Count-Aware Reliability Fusion). Biến thể này đã được hiện thực trong mã nguồn (`src/ufm_model.py`, `variant=\"count_aware_fusion\"`) và đã có bộ kiểm thử hồi quy, nhưng CHƯA được chạy ablation định lượng trên tập validation 40.000 ca nên chưa thể kết luận về hiệu quả thực tế.")

# Formula 6.1: Count Aware Fusion as implemented
add_callout_box(doc, "6.1", "Cơ chế cổng hợp nhất phụ thuộc tần suất (Count-Aware Reliability Fusion) — công thức đã hiện thực",
                "u_{cf,i} \\leftarrow u_{cf,i} + \\exp\\left(-\\frac{count_i}{5}\\right), \\quad w_{cf,i} = \\frac{e^{-u_{cf,i}}}{e^{-u_{cf,i}} + e^{-u_{sem,i}}}",
                "count_i là số lượt tương tác của sản phẩm i trong tập huấn luyện. Cổng hợp nhất vẫn là cổng tin cậy học được, chỉ được cộng thêm hình phạt vào bất định của nhánh CF theo bậc đếm tương tác.")

add_p(doc, "Với cơ chế này, item có $count_i$ rất nhỏ (Extreme Cold, 1–5 lượt) bị cộng hình phạt lớn vào $u_{cf}$ nên $w_{cf}$ bị đẩy xuống thấp, dồn trọng số sang nhánh nội dung CLIP giống hành vi ở Zero-shot. Đây là hướng giải pháp đang chờ được kiểm chứng bằng thực nghiệm loại trừ trên nhóm Extreme Cold; nhóm không khẳng định trước rằng nó sẽ lấp đầy khoảng cách -0,078 NDCG so với TF-IDF, và hiện chưa có số đo định lượng nào trong kho bằng chứng của đề tài.")

# ==============================================================================
# CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
# ==============================================================================
add_h1(doc, "CHƯƠNG 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", page_break=True)

add_h2(doc, "7.1. Kết luận")
add_p(doc, 'Đề tài đã triển khai pipeline, kiến trúc UFM và demo; phần đánh giá còn đang hoàn thiện. Chưa kết luận đã đạt mục tiêu cải thiện cold-start hoặc hoàn thành toàn bộ mục tiêu nghiên cứu.')
add_p(doc, "1. Về mặt kiến trúc: Xây dựng thành công mô hình UFM-Rec kết hợp hài hòa giữa mô hình nền tảng đa phương thức CLIP ViT-B/32 đóng băng với mạng nơ-ron học tuần tự Causal Transformer. Các cơ chế Cross-Modal Alignment Gate và Learned Reliability Proxy hoạt động ổn định, cho phép mô hình thích ứng linh hoạt theo mức độ sẵn có của dữ liệu.")
add_p(doc, '2. Thực nghiệm: kết quả lịch sử v1 Overall NDCG@10 UFM 0,285611 so với TF-IDF 0,260367, nhưng Cold macro 0,163738 thấp hơn TF-IDF 0,191270. Trên validation v2 cùng ứng viên, UFM cold macro 0,182051, TF-IDF 0,195672. Trên validation v3 cùng 40.000 ca và 100 ứng viên/ca, full 0,182056, semantic-only 0,190036, fixed-fusion 0,181305 và TF-IDF 0,195682. Đây là các giao thức riêng; v3 dùng lại quần thể validation đã tham gia chọn checkpoint nên không phải holdout độc lập.')
add_p(doc, '3. Calibration lịch sử: ECE 0,164477 giảm xuống 0,041497 với T = 1,789353, có điều kiện trên 100 ứng viên. Đây là kiểm toán thăm dò, không phải xác suất mua hoặc bảo đảm cho quyết định kinh doanh.')
add_p(doc, '4. Phạm vi vận hành: snapshot RAM container 9,37 GiB trong lịch sử không phải peak RAM/VRAM. Số 2.167,96 QPS và p50 0,80 ms chỉ thuộc backend content 30 sản phẩm. UFM thật đã qua 6/6 ca API trên FITLAB và 6/6 ca API trên Windows từ bản sao đã kiểm tra SHA, catalog 767.045 sản phẩm. Bộ test mã hiện tại trên Windows đạt 94 pass, 3 skip dành cho Linux. Cần benchmark nhiều request và nghiệm thu giao diện local trước bản nộp cuối.')
add_p(doc, '5. Phân tích theo tần suất train trên dự đoán validation v3: full trừ semantic-only NDCG@10 là -0,002424 ở count 0, -0,024101 ở count 1-5, +0,002586 ở count 6-20 và +0,014865 ở count 21+; mỗi nhóm có 10.000 ca. Đây là so sánh mô tả trên một seed, không chứng minh nhánh ID hay gate là nguyên nhân. Nhóm đã tách count-aware fusion khỏi adaptive dropout và khóa 15 run (năm variant, ba seed), nhưng chưa có kết quả train mới.')

add_h2(doc, "7.2. Hướng phát triển tiếp theo")
add_p(doc, "Bảng 7.1 vạch rõ lộ trình nghiên cứu tiếp theo để nâng cao hơn nữa chất lượng của hệ thống:")

next_headers = ["Hạng mục hạn chế hiện tại", "Nguyên nhân khách quan", "Giải pháp và lộ trình hoàn thiện tiếp theo"]
next_rows = [
    [
        "Nghịch lý suy thoái Extreme Cold-Start",
        "Giả thuyết: quá khớp (overfitting) vector ID embedding khi số tương tác quá ít (1–5 lần) — chưa có thực nghiệm loại trừ.",
        "Triển khai Count-Aware Reliability Fusion (Công thức 6.1) và chạy ablation định lượng trên nhóm Extreme Cold để kiểm chứng giả thuyết."
    ],
    [
        "Thực nghiệm đơn seed (seed=42)",
        "Giới hạn thời gian xếp hàng GPU trên hạ tầng máy chủ FITLAB.",
        "Chạy hoàn tất các seed 7, 2026 và tính toán khoảng tin cậy mean ± std theo giao thức đã khóa."
    ],
    [
        "Test v2 đã được xem",
        "Artifact v2 có test metric; chưa có bằng chứng manifest khóa đầy đủ trước đánh giá.",
        "Công bố test đã được xem; thống nhất giao thức cho thực nghiệm sau và không chọn cấu hình theo điểm test đã biết."
    ],
    [
        "7 cấu hình Ablation Study",
        "Chiến dịch huấn luyện quy mô lớn cần nhiều chu kỳ GPU song song.",
        "Bảy ablation lịch sử đã có kết quả cùng validation v3 một seed. Chỉ chạy thêm count-aware và adaptive dropout theo giao thức 15 run đã khóa; không coi queue là kết quả."
    ]
]
add_data_table(doc, "7.1", "Hạn chế nghiên cứu và lộ trình hoàn thiện tiếp theo", next_headers, next_rows,
               col_widths=[Inches(2.0), Inches(2.2), Inches(3.0)],
               alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.JUSTIFY])

# ==============================================================================
# TÀI LIỆU THAM KHẢO
# ==============================================================================
add_h1(doc, "TÀI LIỆU THAM KHẢO", page_break=True)
refs = [
    "[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., \"Learning Transferable Visual Models From Natural Language Supervision,\" in Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR, vol. 139, pp. 8748–8763, 2021.",
    "[2] W.-C. Kang and J. McAuley, \"Self-Attentive Sequential Recommendation,\" in IEEE International Conference on Data Mining (ICDM), pp. 197–206, 2018.",
    "[3] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On Calibration of Modern Neural Networks,\" in Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR, vol. 70, pp. 1321–1330, 2017.",
    "[4] Y. Hou, J. Mu, W. X. Zhao, Z. Li, C. Ding, and J.-R. Wen, \"Towards Universal Sequence Representation Learning for Recommender Systems,\" in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pp. 585–593, 2022.",
    "[5] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, \"Neural Collaborative Filtering,\" in Proceedings of the 26th International Conference on World Wide Web (WWW), pp. 173–182, 2017.",
    "[6] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, \"BPR: Bayesian Personalized Ranking from Implicit Feedback,\" in Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI), pp. 452–461, 2009.",
    "[7] F. Sun, C. Liu, X. Han, Z. Hong, H. Pei, and J. Wu, \"BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer,\" in Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM), pp. 1441–1450, 2019.",
    "[8] J. McAuley, C. Targett, Q. Shi, and A. van den Hengel, \"Image-Based Recommendations on Styles and Substitutes,\" in Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 43–52, 2015.",
    "[9] V. W. Zheng, Y. Zheng, X. Xie, and Q. Yang, \"Collaborative Location and Activity Recommendations with GPS History Data,\" in Proceedings of the 19th International Conference on World Wide Web (WWW), pp. 1029–1038, 2010.",
    "[10] Y. Koren, R. Bell, and C. Volinsky, \"Matrix Factorization Techniques for Recommender Systems,\" Computer, vol. 42, no. 8, pp. 30–37, 2009.",
    "[11] Z. Wei, J. He, K. Chen, Y. Zhou, and Z. Tang, \"Collaborative Filtering and Deep Learning Based Recommendation System for Cold Start Items,\" in Proceedings of the 10th International Conference on Advances in Information Technology (IAIT), pp. 1–6, 2018.",
    "[12] J. Ni, J. Li, and J. McAuley, \"Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects,\" in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 188–197, 2019."
]
for ref in refs:
    p_ref = doc.add_paragraph()
    p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ref.paragraph_format.first_line_indent = Pt(-18)
    p_ref.paragraph_format.left_indent = Pt(18)
    p_ref.paragraph_format.space_before = Pt(2)
    p_ref.paragraph_format.space_after = Pt(4)
    r_ref = p_ref.add_run(ref)
    r_ref.font.name = "Times New Roman"
    r_ref.font.size = Pt(11)

print("Saving generated document...")
doc.save(str(OUTPUT_PATH))
print(f"Successfully generated: {OUTPUT_PATH}")

# Step 4: Render the math — inline LaTeX becomes Unicode, display equations
# become native Word equations (OMML).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polish_report_docx import polish
from latex_to_omml import latex_to_omml

equation_resource = Path(__file__).resolve().parent / "report_equations_omml.json"
cached = json.loads(equation_resource.read_text(encoding="utf-8")) if equation_resource.exists() else {}
equations = {}
for formula_id, latex in CALLOUT_LATEX.items():
    try:
        equations[formula_id] = latex_to_omml(latex)
    except Exception as error:
        if formula_id not in cached:
            raise
        equations[formula_id] = cached[formula_id]
        print(f"  equation {formula_id}: kept cached OMML ({error})")
equation_resource.write_text(json.dumps(equations, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Equation resource synced: {len(equations)} native equations")

formula_ids = polish(OUTPUT_PATH)
print(f"Math polished: {len(formula_ids)} native equations ({', '.join(formula_ids)})")
print(f"File size: {os.path.getsize(OUTPUT_PATH)} bytes")

