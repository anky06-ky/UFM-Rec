# UFM-Rec: phạm vi đầy đủ và mốc triển khai

Cập nhật 28/09/2026. Người dùng đã chọn **giữ đầy đủ proposal**, không thu gọn
thành TF-IDF + Transformer. Hạn nộp chưa được cung cấp. Quyết định của người
dùng chưa đồng nghĩa giảng viên đã duyệt thay đổi triển khai; không tự gửi thông
báo hoặc xin duyệt thay người dùng.

## 1. Phần đã làm trong đợt này

- Đọc đủ 6 trang `Proposal.pdf`, đối chiếu với pipeline hiện có.
- Giữ nguyên `src/train_gpu_recommender.py`, dữ liệu và checkpoint cũ. Lượt
  `content_transformer_v1` là mô hình đối chứng, không đổi tên thành UFM-Rec.
- Thêm `src/ufm_model.py`: adapter nhận embedding văn bản/ảnh từ encoder đóng
  băng; nhánh ID + Transformer causal; hai uncertainty head Softplus; reliability
  `exp(-u)`; UGAF có cross-alignment; recommendation head; BPR, alignment,
  uncertainty regularization và calibration loss.
- Thêm wrapper CLIP thật, ghim model commit và đóng băng encoder. Thêm CLI trích
  đặc trưng có checkpoint theo chunk, không tự train hoặc đánh giá test.
- Kiểm kê toàn bộ catalog thật bằng `src/prepare_ufm_catalog.py`, giữ đúng
  `row_index` của baseline và chỉ dùng `train_count`, không đưa `rating_number`,
  `average_rating`, review text hoặc count validation/test vào đặc trưng.
- Thêm 13 kiểm thử CPU trong `tests/test_ufm_core.py`, đã chạy PASS. Kiểm thử dùng dữ liệu tổng
  hợp, không phải kết quả thực nghiệm recommendation.
- Chạy CLIP thật trên 4 sản phẩm đầu catalog bằng CPU: 4/4 văn bản và 4/4 ảnh tải,
  giải mã, tạo được vector 512 chiều; padding/norm đúng, UFM forward/BPR/backward
  với features thật hữu hạn. Kiểm tra encoder không nhận gradient và luôn eval
  ngay cả khi caller gọi train. Cache kiểm tra: `runs/foundation_cpu_smoke_4_v2`.
  Không sử dụng loss của ví dụ kỹ thuật làm metric đồ án.
- Đã triển khai các module mới lên FITLAB; 17 kiểm thử CPU (13 core + 4 queue)
  PASS trên Python 3.12.3 / PyTorch 2.12.1+cu130. Catalog local đã nhập lên FITLAB
  sau khi đối chiếu hash mapping, text và split. Job queue được xác nhận đang chờ
  baseline/GPU lúc 12:43 ngày 28/09; chưa chạy smoke GPU hay extraction full.
  Chi tiết và cách phục hồi: `docs/FITLAB_FOUNDATION_20260928.md`.

**Cập nhật sau snapshot 12:43:** đối chứng hoàn tất 8 epoch, best ở epoch 5;
CLIP GPU smoke 128/128 văn bản và ảnh PASS, extraction toàn catalog đang chạy.
Đã triển khai trainer UFM với graph negative train-only, checkpoint/RNG/cursor
và loss tích lũy khi resume. 25 kiểm thử CPU PASS trên local và FITLAB (9,692 giây
trên FITLAB). Queue train nền đã khởi động, wrapper PID 117817; lúc xác nhận
đang chuẩn bị graph CPU, chưa có optimizer step UFM GPU.
Xem [hướng dẫn train mới](FITLAB_UFM_TRAINING_20260928.md).

**Chưa hoàn thành:** cache Foundation Model cho toàn bộ catalog,
GPU smoke/resume và các lượt train/validation UFM, baseline SASRec/BERT4Rec
chuẩn, bộ dữ liệu thứ hai, đánh giá test UFM, demo và tài liệu bảo vệ.

## 2. Kiểm kê dữ liệu đa phương thức thật

Nguồn: `data/processed/toys_games_full_temporal/foundation_catalog_v1/catalog_report.json`.
Đã quét 890.874 bản ghi metadata, khớp đầy đủ 767.045 sản phẩm trong mapping.

| Nhóm sản phẩm | Tổng | Có văn bản | Có URL ảnh | Có cả hai |
|---|---:|---:|---:|---:|
| Zero-shot | 182.199 | 182.199 | 182.191 | 182.191 |
| Extreme cold | 388.352 | 388.352 | 388.333 | 388.333 |
| Cold | 120.002 | 120.001 | 119.995 | 119.994 |
| Warm | 76.492 | 76.492 | 76.488 | 76.488 |
| Tổng | 767.045 | 767.044 | 767.007 | 767.006 |

Số **sản phẩm** ở đây khác số **tương tác**. Có URL không bảo đảm ảnh còn tồn
tại, tải được hoặc giải mã được; phải báo độ phủ ảnh thành công sau extraction.
Không dùng ảnh review làm ảnh sản phẩm. Metadata là snapshot, chưa chứng minh
văn bản/ảnh đã tồn tại đúng tại thời điểm tương tác lịch sử.

## 3. Ánh xạ proposal sang code

| Thành phần proposal | Triển khai đầu tiên | Điều kiện nghiệm thu |
|---|---|---|
| Semantic Foundation Branch, eq. 4-5 | CLIP ViT-B/32 văn bản + ảnh đóng băng, adapter nhỏ trainable | Features thật, hữu hạn, đúng mapping, encoder không nhận gradient |
| Collaborative Sequential Branch, eq. 6-7 | ID train-only + positional embedding + causal Transformer; candidate-conditioned pair vector | Không đọc tương lai/padding; không dùng ID chưa train; so sánh với baseline tuần tự chuẩn |
| Dual uncertainty, eq. 8-11 | Hai MLP + Softplus; chuẩn hóa reliability bằng softmax(-u) để tránh underflow | Uncertainty hữu hạn; gate giảm khi uncertainty nhánh đó tăng; đo reliability thực nghiệm |
| UGAF, eq. 12-14 | Gated cross-alignment + weighted CF/semantic + MLP scoring | Ablation và phân tích alpha theo regime; không chỉ chứng minh bằng smoke test |
| Objective, eq. 15 | BPR + cosine alignment + uncertainty residual regularization + sampled BCE | Gradient train-only; báo từng loss; chọn lambda bằng validation |
| Calibration | ECE top-1, NLL, Brier trong candidate set; temperature scaling validation-only ở bước sau | Reliability diagram trước/sau calibration và cùng candidate protocol |

CLIP là lựa chọn thực nghiệm ban đầu, **không có tuyên bố là model tốt nhất**.
Hai encoder CLIP dùng chung không gian 512 chiều; văn bản giới hạn 77 token nên
cần kiểm tra ảnh hưởng cắt mô tả. Có thể so sánh encoder văn bản dài hơn sau khi
có số liệu validation, không đổi model theo kết quả test.

Adapter nhận `[text; image; modality_masks]` và chỉ các module nhỏ cùng nhánh
tuần tự được cập nhật. Đây là phương án encoder đóng băng + adapter của proposal;
không được viết trong báo cáo rằng đã fine-tune LoRA bên trong CLIP. Muốn kiểm
chứng H5 phải thêm đối chứng frozen-feature-only / adapter / chi phí full tuning
hoặc công bố rõ mức độ kiểm chứng còn thiếu.

Độ bất định hiện là **learned reliability proxy**, không phải posterior Bayes
hay tách riêng epistemic/aleatoric uncertainty. Proposal chưa định nghĩa chi tiết
`L_unc`; lựa chọn đầu tiên là khớp residual bình phương của từng nhánh trên nhãn
sampled, với residual được detach. Cần ablation và reliability plot để chứng minh
proxy có ý nghĩa, không suy diễn chỉ từ head Softplus.

## 4. Hợp đồng dữ liệu và các kiểm thử

- ID zero là padding; item thứ `row_index` có ID `row_index + 1`.
- Bảng text/image: `[767046,512]`, padding row zero, float16 trên đĩa; modality
  mask boolean `[767046,2]`. Khi huấn luyện chuyển riêng batch sang float32.
- Train-only counts: int64 `[767046]`, padding row 0. Counts full train dùng để
  phân regime; không dùng count tương lai trong validation/test.
- Histories right-padding, max 20 item; dữ liệu lịch sử lấy từ pipeline temporal
  hiện có. Causal mask là lớp bảo vệ thêm, không thay kiểm tra timestamp upstream.
- Candidate-first-positive cho loss/đánh giá. Negative sampler mới phải loại
  target trùng, item lịch sử và positive đã biết, không đọc nhãn test để lọc.
- Cold ID không được lookup weight chưa học; nhánh CF không khả dụng thì alpha=0.
- Thiếu một modality không loại sản phẩm: zero-mask modality đó. Nếu cả CF và
  semantic không có tín hiệu, dùng popularity **train-only**, giống đối chứng.

Các kiểm thử đã qua: gate đúng công thức và ổn định với uncertainty 1000;
unknown-ID/padding không ảnh hưởng và không nhận gradient; lịch sử causal;
missing modality/empty history; gradient adapter/head/alignment hữu hạn;
ablation chạy được và no-uncertainty không train uncertainty head; ECE/NLL/Brier
khớp ví dụ tính tay, xử lý tie không thiên vị cột positive; URL an toàn; catalog không
ghi đè và giữ thứ tự. Kiểm thử đó không chứng minh mô hình đã học tốt trên Amazon.
Đã mô phỏng ngắt sau một chunk extraction và xác nhận resume tạo bảng/features
hash và status counts giống lượt không ngắt. Đây là kiểm thử bằng encoder giả lập,
không phải bằng chứng đã chạy extraction full qua một lần container FITLAB tắt.

```bash
python -m unittest discover -s tests -p test_ufm_core.py -v
```

## 5. Chạy extraction an toàn

Mã mới đã triển khai lên FITLAB, giữ nguyên mã/checkpoint của baseline đang chạy.
Không cài lại PyTorch CUDA, không khởi chạy hai job GPU dài cùng lúc. Dependency
UFM nằm trong môi trường riêng `/tmp/ufm_venv_20260928_cdcp1xhw`, đọc PyTorch có
sẵn qua `.pth`; không sửa package của môi trường baseline. Thư mục `/tmp` mất
nếu container được tạo lại; cách dựng lại nằm trong tài liệu deployment.

Catalog đã hoàn thành trên local và nhập lên FITLAB. Lệnh dưới chỉ dành cho nơi
có metadata gốc và output chưa tồn tại; **không chạy lại trên catalog đã nhập**:

```bash
python src/prepare_ufm_catalog.py
```

Cài dependency bổ sung trong môi trường UFM riêng, không chạy pip vào venv của
lượt baseline. Lệnh dưới đã thực hiện; không cần cài lại:

```bash
/tmp/ufm_venv_20260928_cdcp1xhw/bin/python -m pip install -r requirements-ufm.txt
```

Queue đã bật sẽ tự chạy smoke khi baseline hoàn thành và GPU rảnh qua hai lần
kiểm tra. **Không chạy thêm các lệnh manual bên dưới trong khi queue còn sống.**
Nếu phục hồi manual sau khi xác nhận queue đã dừng, chạy smoke trước:

```bash
/tmp/ufm_venv_20260928_cdcp1xhw/bin/python -u src/extract_foundation_features.py \
  --device cuda --limit 128 --batch-size 32 --output runs/foundation_smoke_128_v1
```

Sau smoke, kiểm tra số ảnh thành công, time/throughput, VRAM và fingerprint rồi
mới trích toàn bộ catalog vào folder **khác**:

```bash
/tmp/ufm_venv_20260928_cdcp1xhw/bin/python -u src/extract_foundation_features.py \
  --device cuda --limit 0 --batch-size 64 \
  --output data/processed/toys_games_full_temporal/foundation_clip_b32_v1
```

Thêm `--resume` cùng cấu hình nếu bị ngắt sau khi có `progress.json`. Marker
`complete.json` chỉ được ghi sau khi đủ số dòng và có hash các bảng. Cache smoke
không đủ item và **không được dùng** cho full training. Chunk chưa commit sẽ được
chạy lại; log lỗi có thể lặp dòng chẩn đoán, cursor và tổng trong progress mới là
trạng thái chính thức. Muốn thay policy tải ảnh/lần thử/revision, dùng folder mới.

Khoảng 1,46 GiB cho hai bảng float16 toàn catalog, chưa gồm model weights,
runtime, checkpoint và cache/backup. Vì lưu embedding và không lưu toàn bộ ảnh,
đợt đầu chỉ tải ảnh trong RAM rồi giải phóng. Chưa đo thời gian extraction full;
không lấy thời gian 4/128 sản phẩm nhân thẳng để hứa thời gian hoàn thành.

## 6. Kế hoạch giữ đầy đủ các phần còn lại

1. Đối chứng GPU đã hoàn tất 8 epoch, best epoch 5. Tiếp tục đối chiếu validation;
   không dùng checkpoint
   TF-IDF làm trọng số pretrained cho CLIP/UFM.
2. Smoke Foundation Model trên FITLAB, trích full features, kiểm tra manifest/mapping
   và ảnh hỏng theo regime. Dữ liệu phải giữ nguyên split 80/10/10.
3. Trainer UFM đã viết và kiểm thử CPU, gồm BPR sampling train-only,
   checkpoint model/optimizer/scaler/RNG/cursor và fingerprint, loss mean đúng
   khi resume giữa epoch. Queue đang chuẩn bị CPU graph/chờ features;
   cần GPU smoke forward/backward/validation/resume trước full run.
4. Chốt tiêu chí validation **trước** chạy nhiều biến thể: NDCG@10 macro của ba
   nhóm cold-start là metric chính, warm/overall và nhóm có/không lịch sử báo
   đồng thời. Xác định các fold calibration-validation theo thời gian nếu tuning
   temperature; không dùng test.
5. Train từ đầu từng ablation với cùng data, candidate set, budget và seeds:
   full; no uncertainty (learned gate, bỏ L_unc); fixed fusion; no cross-align;
   semantic-only; collaborative-only; text-only; image-only. Không chỉ tắt module
   ở checkpoint full rồi gọi đó là ablation training. Lambda/alpha phải validation-only.
6. Bổ sung các baseline còn thiếu trong proposal: MF/BPR, SASRec, BERT4Rec và
   fixed/concat hybrid sử dụng cùng pretrained features. Custom Transformer trong
   lượt cũ không được tự coi là tái hiện chuẩn SASRec/BERT4Rec.
7. Dùng MovieLens-1M hoặc tập nhỏ tương tự làm nhóm dữ liệu thứ hai, giữ chia theo
   thời gian. Nếu không có ảnh, công bố đó là sanity check text/sequential và không
   tuyên bố đã kiểm chứng đa phương thức trên cả hai bộ.
8. Khóa config, selection và temperature bằng validation, sau đó đánh giá test
   một đợt có kiểm soát cho các config đã chốt. Báo cold breakdown, ECE/NLL/Brier,
   nhiều seed/độ biến thiên và paired uncertainty interval phù hợp người dùng.
   1 positive + 99 negative chỉ là sampled ranking/calibration, không phải xác
   suất mua thật hoặc kết quả xếp hạng toàn catalog.
9. Đo params trainable/frozen, peak VRAM, training/inference throughput, latency.
   Chạy thành công trên RTX 6000 24 GiB không tự chứng minh chạy được GPU 16 GiB;
   phải đo/cấu hình giới hạn phù hợp hoặc nêu rõ chưa kiểm chứng.
10. Demo nghiên cứu: chọn lịch sử, Top-K, tên/ảnh sản phẩm, train-frequency regime,
    alpha CF/semantic và cảnh báo độ tin cậy. Cuối cùng biểu đồ, báo cáo cuối kỳ,
    slide và kịch bản bảo vệ. Không nhúng thông tin đăng nhập vào demo.

## 7. Nguồn phương pháp

- Kiến trúc nghiên cứu: `docs/Proposal.pdf`, phương trình 4-15; các ngưỡng kỳ vọng
  3%-8% chưa phải kết quả đạt được.
- [CLIP paper, Radford et al.](https://arxiv.org/abs/2103.00020)
- [CLIP official model card](https://huggingface.co/openai/clip-vit-base-patch32)
- [SASRec official implementation](https://github.com/kang205/SASRec)
- [BPR, Rendle et al.](https://arxiv.org/abs/1205.2618)
- [Calibration, Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)

CLIP được dùng cho nghiên cứu trên văn bản sản phẩm tiếng Anh; model card không
phải cam kết chất lượng ở domain này. Demo đồ án không đồng nghĩa sẵn sàng triển
khai thương mại. So sánh, calibration và kết luận novelty cần dữ liệu thực nghiệm.
