"""Build the 30 Sep progress snapshot from experiment artifacts (no training)."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import statistics
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.graphics.shapes import Drawing,Line,String,Circle,PolyLine
from reportlab.graphics import renderSVG

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed/toys_games_full_temporal'
NAME = 'Bao_cao_tien_trinh_UFM_Rec_2026_09_30'
REPORTS = ROOT/'reports'
OUTPUT = ROOT/'output/pdf'/f'{NAME}.pdf'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


selected = []
reference = None
for folder in ('bpr_mf','bpr_mf_seed7','bpr_mf_seed2026'):
    path = DATA/'models'/folder
    config,complete = read(path/'config.json'),read(path/'completed.json')
    assert complete['epochs']==5 and not complete['smoke_run'] and not complete['test_evaluated']
    comparable = {k:v for k,v in config.items() if k!='seed'}
    if reference is None: reference=comparable
    assert comparable==reference,'Seed runs do not share the same protocol.'
    with np.load(path/'best.npz') as checkpoint: epoch=int(checkpoint['epoch'])
    record = read(path/f'epoch_{epoch:03d}.json')['validation']
    selected.append(dict(seed=config['seed'],epoch=epoch,run=str(path.relative_to(ROOT)),
                         overall_ndcg=record['overall']['ndcg@10'],
                         cold_macro_ndcg=record['cold_macro_ndcg@10'],
                         warm_ndcg=record['by_regime']['warm']['ndcg@10']))
summary = dict(seeds=[r['seed'] for r in selected],runs=selected,split='validation',
    selection='Best cold macro NDCG@10 for each seed, fixed five-epoch budget',
    sample_standard_deviation_ddof=1,test_evaluated=False)
for name in ('overall_ndcg','cold_macro_ndcg','warm_ndcg'):
    values=[r[name] for r in selected]
    summary[name]=dict(mean=statistics.mean(values),std=statistics.stdev(values))
(REPORTS/'BPR_MF_multiseed_2026_09_30.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

cal = read(DATA/'models/content_calibration_v1/report.json')
demo = read(REPORTS/'demo_acceptance_2026_09_30.json')
remote = read(REPORTS/'FITLAB_snapshot_2026_09_30.json')
before,after = cal['before']['overall'],cal['after']['overall']
test_bytes = (ROOT/'tmp/tests_20260930.log').read_bytes()
tests = test_bytes.decode('utf-16' if test_bytes.startswith(b'\xff\xfe') else 'utf-8')
assert 'Ran 36 tests' in tests and tests.rstrip().endswith('OK')
assert len(demo['checks'])==4 and all(r['passed'] for r in demo['checks'])
export = read(ROOT/'output/demo/ufm-rec-demo.json')
assert export['request']['regime']=='zero_shot' and len(export['results'])==5
assert all(r['train_count']==0 for r in export['results'])
assert export['request']['history'][0] not in [r['asin'] for r in export['results']]

pdfmetrics.registerFont(TTFont('UFMArial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('UFMArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFontFamily('UFMArial',normal='UFMArial',bold='UFMArialBold')
GREEN=colors.HexColor('#235c49'); MUTED=colors.HexColor('#64766e')


def reliability_plot():
    drawing=Drawing(500,235)
    x0,y0,w,h=52,36,280,175
    for tick in range(6):
        v=tick/5
        drawing.add(Line(x0,y0+v*h,x0+w,y0+v*h,strokeColor=colors.HexColor('#e2e7df'),strokeWidth=.5))
        drawing.add(String(x0-8,y0+v*h-3,f'{v:.1f}',fontName='UFMArial',fontSize=8,textAnchor='end',fillColor=MUTED))
        drawing.add(String(x0+v*w,y0-14,f'{v:.1f}',fontName='UFMArial',fontSize=8,textAnchor='middle',fillColor=MUTED))
    drawing.add(Line(x0,y0,x0+w,y0+h,strokeColor=colors.HexColor('#a0aea3'),strokeDashArray=[3,3]))
    for name,label,color in [('before','Trước calibration',colors.HexColor('#c38c28')),
                             ('after','Sau calibration',GREEN)]:
        bins=[b for b in cal[name]['overall']['bins'] if b['n']]
        points=[(x0+b['confidence']*w,y0+b['accuracy']*h) for b in bins]
        drawing.add(PolyLine([v for p in points for v in p],strokeColor=color,strokeWidth=1.5))
        for x,y in points:drawing.add(Circle(x,y,2.7,fillColor=color,strokeColor=color))
        yy=185 if name=='before' else 160
        drawing.add(Line(357,yy,380,yy,strokeColor=color,strokeWidth=2))
        drawing.add(String(387,yy-3,label,fontName='UFMArial',fontSize=9,fillColor=color))
    drawing.add(String(x0+w/2,5,'Độ tin cậy top-1',fontName='UFMArial',fontSize=9,textAnchor='middle'))
    drawing.add(String(x0,221,'Độ chính xác top-1',fontName='UFMArial',fontSize=9,fillColor=MUTED))
    drawing.add(String(357,125,'15 bins bằng độ rộng',fontName='UFMArial',fontSize=8,fillColor=MUTED))
    drawing.add(String(357,110,'Chỉ vẽ bin có mẫu',fontName='UFMArial',fontSize=8,fillColor=MUTED))
    drawing.add(String(357,95,'20.000 mẫu audit',fontName='UFMArial',fontSize=8,fillColor=MUTED))
    return drawing


figpath=ROOT/'output/figures/calibration_validation_20260930.svg'
figpath.parent.mkdir(parents=True,exist_ok=True)
renderSVG.drawToFile(reliability_plot(),str(figpath))

rows='\n'.join(f"| {r['seed']} / {r['epoch']} | {r['overall_ndcg']:.6f} | {r['cold_macro_ndcg']:.6f} | {r['warm_ndcg']:.6f} |" for r in selected)
latencies='\n'.join(f"| {r['case']} | PASS, Top {len(r['response']['results'])} | {r['response']['elapsed_ms']:.1f} ms |" for r in demo['checks'])
percentage=remote['latest_feature_log']['rows']/remote['latest_feature_log']['expected_rows']*100
md=f'''# BÁO CÁO TIẾN TRÌNH UFM-REC
## Snapshot ngày 30/09/2026

Hệ khuyến nghị sản phẩm cold-start trên Amazon Reviews 2023 - Toys and Games.
Mốc FITLAB: 11:07 ngày 30/09/2026 (UTC+07). Kiểm tra demo local: 11:10 cùng ngày.

### 1. Kết luận hiện tại

Đã có pipeline dữ liệu đầy đủ, các baseline thực nghiệm và demo local hoạt động trên 767.045 sản phẩm. Đợt này hoàn tất thêm BPR-MF với 3 seed, calibration TF-IDF theo thời gian và giao diện demo có lọc cold-start. Toàn bộ 36 kiểm thử local PASS.

UFM đầy đủ chưa train xong. CLIP đang trích đặc trưng; queue UFM và 7 ablation chờ điều kiện đầu vào/GPU. Chưa có metric GPU UFM, chưa đánh giá test cho BPR-MF hoặc chiến dịch UFM mới. Không thể kết luận dự án đã hoàn tất proposal.

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Temporal data + TF-IDF | Hoàn thành | 11.572.689 tương tác; catalog 767.045 |
| Popularity / content / SVD / RRF | Có kết quả test trước đây | Cùng 40.000 sampled cases |
| BPR-MF | Hoàn thành 3 seed | Mỗi seed 5 epoch, chọn bằng validation |
| Calibration TF-IDF | Hoàn thành bước thăm dò | 20.000 fit + 20.000 audit theo thời gian |
| CLIP text + image | Đang chạy | 522.880/767.045 dòng, {percentage:.2f}% |
| UFM + 7 ablation | Đã có code/queue; đang chờ | Chưa có production checkpoint |
| Demo local | Hoạt động với TF-IDF thật | 4 API scenarios và luồng UI đã kiểm tra |

### 2. Dữ liệu và phạm vi đánh giá

- 16.260.406 review gốc; giữ 11.572.689 tương tác verified, rating từ 4 trở lên.
- Temporal 80/10/10: train 9.258.647; validation 1.159.827; test 1.154.215.
- 584.846 sản phẩm có tương tác train; 182.199 sản phẩm zero-shot.
- Validation/test sample: 40.000 trường hợp mỗi split, 10.000 mỗi regime, 1 positive + 99 negative cố định.
- Phần trăm CLIP chỉ phản ánh tiến độ extraction, không phải phần trăm hoàn thành toàn dự án.

<!-- pagebreak -->
## Kết quả baseline đã kiểm chứng

### 3. Test của các baseline cổ điển (kết quả đã có)

| Mô hình | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Popularity | 0.2669 | 0.1691 | 0.1393 |
| TF-IDF content | 0.4013 | 0.2658 | 0.2243 |
| TruncatedSVD | 0.2193 | 0.1310 | 0.1043 |
| Hybrid RRF | 0.4231 | 0.2657 | 0.2172 |

RRF đạt Recall@10 cao nhất trong nhóm trên; TF-IDF có NDCG@10 cao hơn rất ít. Chênh lệch chưa được kiểm chứng thống kê. Các giá trị test này không so trực tiếp với bảng validation bên dưới.

### 4. BPR-MF: kết quả mới trên validation

ID-only implicit BPR, 32 factors, learning rate 0,05, regularization 0,0001, batch 4.096, 5 epoch/seed. Graph train giữ 1.622.880 người dùng có ít nhất 2 sản phẩm khác nhau và 5.832.475 cặp user-item. Negative chỉ lấy từ item train và loại toàn bộ positive đã biết của user.

Checkpoint mỗi seed được chọn theo NDCG@10 macro của zero-shot, extreme cold và cold; warm/overall được báo đồng thời. Các lượt có cùng hash source, code và cấu hình ngoài seed.

| Seed / epoch chọn | Overall NDCG@10 | Cold macro NDCG@10 | Warm NDCG@10 |
|---|---:|---:|---:|
{rows}

Trung bình ± độ lệch chuẩn mẫu (3 seed): overall {summary['overall_ndcg']['mean']:.6f} ± {summary['overall_ndcg']['std']:.6f}; cold macro {summary['cold_macro_ndcg']['mean']:.6f} ± {summary['cold_macro_ndcg']['std']:.6f}; warm {summary['warm_ndcg']['mean']:.6f} ± {summary['warm_ndcg']['std']:.6f}.

Độ lệch chuẩn seed không phải khoảng tin cậy trên người dùng. BPR-MF không có nội dung để biểu diễn item zero-shot; Recall@10 nhóm này bằng 0 trong cả ba seed. Kết quả cho thấy baseline thiên về warm, chưa phải bằng chứng UFM cải thiện cold-start.

Đối chứng TF-IDF + Transformer GPU cũ đã hoàn tất 8 epoch, best epoch 5: validation overall NDCG@10 0,277327; known-user 0,380043. Đây là kiến trúc đối chứng, không phải checkpoint UFM đầy đủ.

<!-- pagebreak -->
## Calibration theo thời gian

### 5. Thiết lập và kết quả mới

Tái tạo score TF-IDF của 40.000 validation cases; toàn bộ target ranks khớp artifacts baseline. Chia theo timestamp: 20.000 mẫu sớm để fit temperature, 20.000 mẫu muộn để audit; các timestamp trùng không bị tách qua ranh giới. Temperature chọn bằng NLL trên fold fit, trong [0,01; 100], thu được T = {cal['temperature']:.6f}.

| Metric audit | Trước (T = 1) | Sau temperature scaling |
|---|---:|---:|
| ECE top-1, 15 bins | {before['ece']:.6f} | {after['ece']:.6f} |
| NLL | {before['nll']:.6f} | {after['nll']:.6f} |
| Brier multiclass | {before['brier_multiclass']:.6f} | {after['brier_multiclass']:.6f} |

![Reliability diagram](../output/figures/calibration_validation_20260930.svg)

Temperature dương không đổi thứ tự ranking; nó đổi phân phối softmax trên 100 candidates. Các trường hợp đồng hạng top-1 dùng xác suất đúng kỳ vọng với tie-break đều để không thiên vị cột positive.

### Giới hạn cần giữ khi viết báo cáo cuối kỳ

- Đây là calibration thăm dò của baseline TF-IDF, chưa phải kiểm chứng uncertainty của UFM.
- Baseline đã được đánh giá trên full validation trước đó; audit này không phải holdout mới cho việc chọn mô hình.
- Fold thời gian có thể chung người dùng. Chưa có bootstrap interval hay nhiều seed UFM.
- ECE/NLL/Brier chỉ có ý nghĩa trong candidate protocol đã lấy mẫu, không phải xác suất mua trong toàn catalog.
- Phương pháp tham khảo: Guo et al. (2017), On Calibration of Modern Neural Networks, PMLR 70:1321-1330; proceedings.mlr.press/v70/guo17a.html.

<!-- pagebreak -->
## Demo có thể trình bày ngay

### 6. Giao diện và luồng sử dụng

Mở Start_Demo.cmd tại thư mục dự án hoặc truy cập http://127.0.0.1:8765 khi server đang chạy. Demo dùng TF-IDF trên dữ liệu thật; backend và giới hạn được ghi ngay trên giao diện.

![Demo TF-IDF, Top 5 zero-shot từ lịch sử LEGO](../output/demo/UFM_Rec_demo_2026_09_30.png)

Tìm LEGO → thêm sản phẩm → chọn Top K → tìm gợi ý → lọc zero-shot → xuất JSON. Có xóa lịch sử, chống thêm trùng, đánh dấu kết quả cũ khi thay lựa chọn, fallback khi ảnh lỗi và popularity khi lịch sử rỗng. File JSON tải qua UI đã được kiểm tra: đúng 5 kết quả zero-shot, không chứa ASIN lịch sử.

| Kịch bản API trên catalog thật | Kiểm tra | Thời gian server |
|---|---|---|
{latencies}

Mỗi kịch bản đo một lần; chưa phải benchmark throughput, p95 hoặc so sánh phần cứng. Backend UFM có đường dẫn features thay thế với kiểm tra hash; cần checkpoint full hoàn tất trước khi phục vụ. Hướng dẫn và kịch bản 3 phút: docs/DEMO_GUIDE.md.

<!-- pagebreak -->
## Phần còn lại và điều kiện hoàn tất

### 7. Trạng thái FITLAB

CLIP đã phục hồi sau SIGKILL tại 497.856 dòng ngày 29/09; snapshot mới ghi 522.880 dòng và 520.317 ảnh thành công. Cursor queue đọc ở chu kỳ trước là 522.752, nên lệch 128 dòng so với log mới hơn là bình thường. GPU utilization 100%, free 6.528 MiB; PID extraction 443209.

UFM đang waiting_for_complete_features_and_idle_gpu; ablation đang waiting_for_full_ufm_and_idle_gpu. Không restart job đang chạy. Queue tự tiếp tục trong cửa sổ đã cấu hình khi đủ features và GPU rảnh; không bảo đảm tự phục hồi nếu container bị tạo lại hoặc tiến trình bị kill.

| Mốc tiếp theo | Việc cần thực hiện | Điều kiện nghiệm thu |
|---|---|---|
| 1. Full CLIP | Chờ extraction, audit coverage/hash | complete.json đủ 767.045 item, vector hữu hạn |
| 2. UFM full | GPU smoke/resume rồi train/validation | Production completed.json, best checkpoint và metric |
| 3. Ablation | 7 biến thể train độc lập | Cùng budget/seed/protocol, bảng cold/warm và gate |
| 4. Baseline còn thiếu | SASRec, BERT4Rec, concat hybrid | Tái hiện đúng phương pháp, cùng dữ liệu/candidates |
| 5. Kiểm chứng mở rộng | MovieLens-1M; nhiều seed UFM; calibration UFM | Tách rõ sanity check không ảnh và thí nghiệm đa phương thức |
| 6. Hiệu năng + test | VRAM 16 GiB, throughput; khóa config trước test | Bằng chứng đo thực tế, test một đợt có kiểm soát |
| 7. Bàn giao cuối | Demo UFM checkpoint thật, báo cáo và slide | Kết quả thực nghiệm đầy đủ, kết luận có nguồn |

Chưa thể đánh dấu các mốc 2-7 hoàn thành. SASRec/BERT4Rec, concat hybrid, dataset thứ hai và slide bảo vệ cuối cùng chưa được làm xong trong snapshot này. Các bảng mới không thay thế những hạng mục đó.

### 8. Bằng chứng và tái lập

- BPR: models/bpr_mf, bpr_mf_seed7, bpr_mf_seed2026 dưới data/processed/toys_games_full_temporal; tổng hợp reports/BPR_MF_multiseed_2026_09_30.json.
- Calibration: models/content_calibration_v1/report.json và scores_validation.npz; chạy src/calibrate_validation.py với output mới.
- FITLAB: reports/FITLAB_snapshot_2026_09_30.json; báo cáo phục hồi cùng ngày; các log/queue trên server.
- Demo: reports/demo_acceptance_2026_09_30.json; output/demo/ufm-rec-demo.json; ảnh chụp giao diện; Start_Demo.cmd.
- Kiểm thử: python -m unittest discover -s tests -v; 36 tests PASS (lần cuối 8,959 giây). Smoke/unit tests không phải bằng chứng chất lượng recommendation.
- Mã train UFM và fingerprints của chiến dịch đang chạy được giữ nguyên trong đợt này. Thay đổi local chưa commit/push.
'''
(REPORTS/f'{NAME}.md').write_text(md,encoding='utf-8')

styles={
 'title':ParagraphStyle('title',fontName='UFMArialBold',fontSize=23,leading=28,spaceAfter=12,textColor=GREEN),
 'section':ParagraphStyle('section',fontName='UFMArialBold',fontSize=16,leading=21,spaceAfter=13,textColor=GREEN),
 'heading':ParagraphStyle('heading',fontName='UFMArialBold',fontSize=11,leading=15,spaceBefore=8,spaceAfter=6,keepWithNext=True),
 'body':ParagraphStyle('body',fontName='UFMArial',fontSize=9.3,leading=13.1,spaceAfter=8),
 'bullet':ParagraphStyle('bullet',fontName='UFMArial',fontSize=9.1,leading=12.5,spaceAfter=5,leftIndent=10,firstLineIndent=-10),
 'cell':ParagraphStyle('cell',fontName='UFMArial',fontSize=8.2,leading=11.4),
 'th':ParagraphStyle('th',fontName='UFMArialBold',fontSize=8.2,leading=11.4,textColor=colors.white),
}
WIDTH=A4[0]-88
def para(text,style='body'):return Paragraph(escape(text),styles[style])
def table(rows):
    if len(rows[0])==4: ratios=[.37,.21,.21,.21]
    elif rows[0][0]=='Mốc tiếp theo':ratios=[.22,.35,.43]
    else:ratios=[.32,.34,.34]
    t=Table([[para(c,'th' if i==0 else 'cell') for c in row] for i,row in enumerate(rows)],
            colWidths=[WIDTH*r for r in ratios],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),GREEN),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4ed'),colors.white]),
        ('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#d9e1d5')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return t

story=[];lines=md.splitlines();i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line=='<!-- pagebreak -->':story.append(PageBreak())
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            row=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in row):rows.append(row)
            i+=1
        story.extend([table(rows),Spacer(1,10)]);continue
    elif line.startswith('!['):
        match=re.fullmatch(r'!\[(.+)\]\((.+)\)',line)
        path=(REPORTS/match[2]).resolve()
        if path.suffix=='.svg':story.append(reliability_plot())
        else:
            w,h=ImageReader(str(path)).getSize();scale=min(WIDTH/w,325/h)
            story.append(Image(str(path),width=w*scale,height=h*scale))
        story.append(Spacer(1,8))
    elif line.startswith('# '):story.append(para(line[2:],'title'))
    elif line.startswith('## '):story.append(para(line[3:],'section'))
    elif line.startswith('### '):story.append(para(line[4:],'heading'))
    elif line.startswith('- '):story.append(para('• '+line[2:],'bullet'))
    else:story.append(para(line))
    i+=1

def footer(canvas,doc):
    canvas.setFont('UFMArial',8);canvas.setFillColor(MUTED)
    canvas.drawString(44,25,'UFM Rec | Tiến trình nghiên cứu | 30/09/2026')
    canvas.drawRightString(A4[0]-44,25,str(doc.page))

OUTPUT.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=36,bottomMargin=43,
    title='Báo cáo tiến trình UFM Rec - 30/09/2026',author='UFM Rec').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUTPUT)
print(json.dumps(summary,ensure_ascii=False))
