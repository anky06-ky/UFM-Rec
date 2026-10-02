"""Build proposal PDF/HTML and vector figures from docs/proposal.md.

Requires reportlab; defaults to Windows Arial or Linux DejaVu Sans fonts.
Run from any directory: python scripts/build_proposal.py
"""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon
from reportlab.graphics import renderSVG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
W = A4[0] - 100
INK = colors.HexColor('#182e35')
GREEN = colors.HexColor('#177566')
BLUE = colors.HexColor('#477da8')
MUTED = colors.HexColor('#586b73')
PALE = colors.HexColor('#eff5f3')


def fonts():
    for normal, bold in [(Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')),
                         (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                          Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))]:
        if normal.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont('Body', str(normal)))
            pdfmetrics.registerFont(TTFont('Bold', str(bold)))
            pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
            return
    raise FileNotFoundError('Install Arial or DejaVu Sans fonts.')


def label(d, x, y, text, size=9, color=INK, anchor='start', bold=False):
    d.add(String(x, y, text, fontName='Bold' if bold else 'Body', fontSize=size,
                 fillColor=color, textAnchor=anchor))


def box(d, x, y, w, h, lines, fill=PALE):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=colors.HexColor('#bdcfca')))
    for i, text in enumerate(lines):
        label(d, x+w/2, y+h/2+(len(lines)-1)*6-i*12, text, 8.5, anchor='middle')


def arrow(d, x, y, xx, yy):
    d.add(Line(x, y, xx, yy, strokeColor=MUTED, strokeWidth=1))
    if yy < y:
        points = [xx, yy, xx-3, yy+6, xx+3, yy+6]
    else:
        points = [xx, yy, xx-6, yy-3, xx-6, yy+3]
    d.add(Polygon(points, fillColor=MUTED, strokeColor=None))


def figures():
    result = {}
    d = Drawing(W, 155)
    nodes = [(0, ['Review + metadata', 'Nguồn Amazon']), (170, ['Lọc + khử trùng', '11.572.689 tương tác']),
             (340, ['Split theo thời gian', '80 / 10 / 10'])]
    for x, lines in nodes: box(d, x, 104, 150, 42, lines)
    arrow(d, 150, 125, 170, 125); arrow(d, 320, 125, 340, 125)
    for x, lines in [(0, ['TRAIN', 'Fit / học tham số']), (170, ['VALIDATION', 'Chọn config / checkpoint']),
                     (340, ['TEST', 'Đánh giá cấu hình đã khóa'])]:
        box(d, x, 28, 150, 42, lines)
    arrow(d, 415, 104, 415, 70)
    d.add(Line(75, 85, 415, 85, strokeColor=MUTED))
    arrow(d, 75, 85, 75, 70); arrow(d, 245, 85, 245, 70)
    label(d, 0, 8, 'Counts, vocabulary và gradient chỉ từ train; history phải trước thời điểm dự đoán.', 8)
    result['pipeline'] = d

    d = Drawing(W, 264)
    box(d, 0, 210, 220, 43, ['Text / ảnh sản phẩm', 'CLIP đóng băng: 512 + 512'])
    box(d, 275, 210, 220, 43, ['Lịch sử ID + vị trí', 'Embedding -> Transformer causal'])
    box(d, 0, 146, 220, 43, ['Adapter -> semantic pair m', 'Mean history + candidate + masks'])
    box(d, 275, 146, 220, 43, ['Collaborative pair c', 'Last history + candidate + counts'])
    arrow(d, 110, 210, 110, 189); arrow(d, 385, 210, 385, 189)
    box(d, 55, 83, 385, 43, ['Softplus heads: u_sem, u_cf -> masked softmax(-u)',
                               'Cross-alignment A chỉ khi cả hai nhánh có tín hiệu'])
    arrow(d, 110, 146, 110, 126); arrow(d, 385, 146, 385, 126)
    box(d, 55, 22, 385, 42, ['f = w_cf*c + w_sem*m + A -> MLP_rec -> score',
                              'Fallback train-popularity nếu cả hai nhánh thiếu'])
    arrow(d, 247, 83, 247, 64)
    label(d, 0, 3, 'Học: adapter, ID/Transformer, pair MLP, heads/gates. CLIP luôn đóng băng.', 8)
    result['model'] = d

    d = Drawing(W, 215)
    left, width = 104, 315
    for tick in range(8):
        x = left + tick/10/.7*width
        d.add(Line(x, 38, x, 193, strokeColor=colors.HexColor('#dfe7e5')))
        label(d, x, 24, f'{tick/10:.1f}', 8, anchor='middle')
    for i, (name, a, b) in enumerate([('Overall', .260367, .285611), ('Cold macro', .191270, .163738), ('Warm', .467661, .651231)]):
        y = 166-i*53
        label(d, 93, y+2, name, 9, anchor='end')
        for j, (val, col) in enumerate([(a, BLUE), (b, GREEN)]):
            yy = y+12-j*19
            d.add(Rect(left, yy, val/.7*width, 13, fillColor=col, strokeColor=None))
            label(d, left+val/.7*width+4, yy+3, f'{val:.6f}', 8)
    label(d, 110, 5, 'TF-IDF', 9, BLUE, bold=True)
    label(d, 190, 5, 'UFM (1 seed)', 9, GREEN, bold=True)
    label(d, 320, 5, 'NDCG@10; trục bắt đầu từ 0', 8)
    result['results'] = d

    d = Drawing(W, 160)
    for x, lines in [(0, ['Trình duyệt', 'Search / history / Top-K']), (170, ['Proxy đã xác thực', 'Host + Origin + base path']),
                     (340, ['Backend loopback', 'TF-IDF hoặc UFM'])]:
        box(d, x, 105, 150, 43, lines)
    arrow(d, 150, 127, 170, 127); arrow(d, 320, 127, 340, 127)
    box(d, 240, 32, 250, 45, ['Catalog + features + checkpoint', 'Đọc storage; kiểm tra marker và hash'])
    arrow(d, 415, 105, 415, 77)
    box(d, 0, 32, 205, 45, ['Kết quả Top-K / latency / JSON', 'Loại đã xem; lọc regime'])
    d.add(Line(240, 54, 205, 54, strokeColor=MUTED))
    label(d, 0, 8, 'UFM chỉ nghiệm thu sau inference thật qua proxy; status thành công chưa đủ.', 8)
    result['demo'] = d

    d = Drawing(W, 180)
    names = ['G0 Phục hồi', 'G1 Ablation', 'G2 Baseline', 'G3 Seeds / audit', 'G4 Test khóa', 'G5 Bàn giao']
    for i, name in enumerate(names):
        y = 143-i*23
        label(d, 0, y+4, name, 9)
        for j in range(6):
            d.add(Rect(135+j*57, y, 51, 16, fillColor=PALE, strokeColor=None))
        d.add(Rect(135+i*57, y, 51, 16, fillColor=GREEN, strokeColor=None))
    for j in range(6): label(d, 160+j*57, 168, f'G{j}', 8, anchor='middle')
    label(d, 0, 4, 'Thứ tự phụ thuộc đề xuất; không biểu diễn ngày hoặc thời lượng thực tế.', 8)
    result['plan'] = d
    return result


def inline(text, pdf=True):
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'\1' if pdf else r'<code>\1</code>', text)
    tag = 'link' if pdf else 'a'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: f'<{tag} href="{m[2]}">{m[1]}</{tag}>', text)


def main():
    fonts()
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    (OUT/'pdf').mkdir(parents=True, exist_ok=True)
    drawings = figures()
    for name, drawing in drawings.items():
        renderSVG.drawToFile(drawing, str(OUT/'figures'/f'proposal_{name}.svg'))
    body = ParagraphStyle('body', fontName='Body', fontSize=9.2, leading=13.3, textColor=INK, spaceAfter=7)
    small = ParagraphStyle('small', parent=body, fontSize=8.1, leading=11, spaceAfter=4)
    styles = {1: ParagraphStyle('title', parent=body, fontName='Bold', fontSize=28, leading=34, spaceAfter=18),
              2: ParagraphStyle('section', parent=body, fontName='Bold', fontSize=18, leading=23, spaceAfter=16),
              3: ParagraphStyle('sub', parent=body, fontName='Bold', fontSize=11, leading=15, spaceBefore=8, spaceAfter=8)}
    source = (ROOT/'docs/proposal.md').read_text(encoding='utf-8')
    story, web = [], []
    for number, page in enumerate(source.split('<!-- page -->')):
        if number: story.append(PageBreak())
        web.append('<section>')
        for block in re.split(r'\n\s*\n', page.strip()):
            if block.startswith('#'):
                for line in block.splitlines():
                    match = re.match(r'^(#{1,3}) (.*)', line)
                    if match:
                        level, value = len(match[1]), match[2]
                        story.append(Paragraph(inline(value), styles[level]))
                        web.append(f'<h{level}>{inline(value, False)}</h{level}>')
            elif block.startswith('|'):
                rows = [[cell.strip() for cell in row.strip().strip('|').split('|')]
                        for row in block.splitlines() if not re.match(r'^\|\s*---', row)]
                widths = ([W*.4, W*.6] if len(rows[0]) == 2 else [W*.25, W*.35, W*.4])
                table = Table([[Paragraph(inline(cell), small) for cell in row] for row in rows], colWidths=widths, repeatRows=1)
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebe6')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, PALE]),('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                story.extend([table, Spacer(1,10)])
                web.append('<table>'+''.join('<tr>'+''.join(f'<td>{inline(c,False)}</td>' for c in row)+'</tr>' for row in rows)+'</table>')
            elif block.startswith('!['):
                m = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', block)
                name = Path(m[2]).stem.removeprefix('proposal_')
                story.extend([drawings[name], Paragraph(inline(m[1]), small), Spacer(1,5)])
                web.append(f'<figure><img src="{m[2]}" alt="{html.escape(m[1])}"><figcaption>{m[1]}</figcaption></figure>')
            else:
                lines = block.splitlines() if block.startswith('- ') or re.match(r'^1\. ', block) else [block.replace('\n',' ')]
                for line in lines:
                    story.append(Paragraph(inline(line), body))
                    web.append('<p>'+inline(line, False)+'</p>')
        web.append('</section>')
    def footer(canvas, doc):
        canvas.setTitle('UFM-Rec - Proposal 2.0')
        canvas.setAuthor('UFM-Rec project')
        canvas.setFont('Body', 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(50, 27, 'UFM-Rec | Proposal 2.0 | 02/10/2026')
        canvas.drawRightString(A4[0]-50, 27, str(doc.page))
    SimpleDocTemplate(str(OUT/'pdf/proposal.pdf'), pagesize=A4, leftMargin=50, rightMargin=50,
                      topMargin=43, bottomMargin=45).build(story, onFirstPage=footer, onLaterPages=footer)
    css = 'body{font:16px/1.65 Arial,sans-serif;color:#182e35;background:#eaf1ee;margin:0}section{max-width:850px;margin:30px auto;padding:45px;background:white}h1{font-size:42px;color:black}h2{font-size:27px}h3{font-size:19px}table{width:100%;border-collapse:collapse;font-size:14px}td{padding:10px;vertical-align:top;border-bottom:1px solid #dae5df}tr:first-child{background:#dcebe6;font-weight:bold}tr:nth-child(even){background:#eff5f3}img{width:100%}figure{margin:20px 0}figcaption{font-size:13px;color:#586b73}a{color:#177566}@media print{body{background:white}section{margin:0;padding:0;break-after:page}}@media(max-width:600px){section{padding:20px;margin:10px}table{font-size:12px}}'
    (ROOT/'docs/proposal.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UFM-Rec Proposal 2.0</title><style>'+css+'</style><body>'+''.join(web)+'</body></html>', encoding='utf-8')
    print('Created output/pdf/proposal.pdf, docs/proposal.html and 5 vector figures.')


if __name__ == '__main__':
    main()
