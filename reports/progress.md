# Tiến độ UFM-Rec

## Mốc 24/10/2026 — khôi phục và kiểm tra workspace FITLAB-02

Bản cập nhật dựa trên `status.py` lúc **12:28:37 UTC+07**. `DA_AI` được dựng lại từ
archive 24/09, sau đó đồng bộ source từ commit `faad9ed4ec0b27a837a6736f30bbaab61659e387`.
Archive gốc khoảng **1,3 GiB** vẫn ở thư mục cha; dữ liệu và `runs/` hiện có không bị ghi đè.

### 6 chặng chuẩn bị workspace đã hoàn tất

1. **Khôi phục dữ liệu FITLAB:** bộ temporal Toys có 11.572.689 tương tác
   (9.258.647 train, 1.159.827 validation, 1.154.215 test), mapping/content **767.045 item**
   và TF-IDF. `items.csv.gz` 7,2 MiB và `products_text.csv.gz` 283 MiB còn trên FITLAB.
2. **Đồng bộ source hiện tại:** checkout FITLAB khớp commit GitHub/local
   `faad9ed`. Mã nguồn đã đồng bộ vào `DA_AI`, loại trừ `data/`, `runs/`, `.git/`,
   `.venv/`, `tmp/`, `archive/`, `reports/` và `output/`.
3. **Kiểm kê job mới:** `status.py` lúc 12:28:37 báo CLIP **0/767.045, images OK 0**,
   mọi heartbeat/queue “chưa có trạng thái”, `FULL CHECKPOINT: not complete`.
   Không có trainer/queue UFM, cache CLIP hay checkpoint; không có gì để resume.
4. **Cài môi trường riêng:** `/home/coder/.venv-ufm` (khoảng **5,5 GiB**) có PyTorch
   **2.14.1+cu132**, Transformers **4.57.6**, Pillow và tqdm; `torch.cuda.is_available()`
   là `True`, thiết bị **Quadro RTX 6000**. Filesystem FITLAB còn khoảng **13 GiB**;
   venv mặc định không bị sửa.
5. **Sửa đường dựng catalog:** `src/prepare_ufm_catalog.py` hiện đọc JSONL gzip hoặc
   các Parquet shard có thứ tự, xử lý schema ảnh struct/list và ghi hash từng shard.
   Đã thêm `requirements-catalog-parquet.txt` và downloader có resume/checksum.
6. **Khôi phục metadata nguồn:** tải đủ 5 shard Toys từ
   [McAuley-Lab trên Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
   tại revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`; tổng **972.667.016 byte**.
   Mỗi file đã khớp kích thước và SHA-256; manifest local ở
   `data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/download_manifest.json`.

### Bước chuyển dữ liệu và trạng thái mô hình

- **Chưa chuyển shard lên FITLAB, chưa tạo catalog và chưa chạy CLIP/training.** FITLAB
  không phân giải được Hugging Face; local đã có metadata nhưng cần chuyển 5 shard cùng
  builder/requirements cập nhật vào `DA_AI`. Người dùng đã cho phép chuyển dữ liệu;
  thư mục nhận `/home/coder/iDragonCloud/DA_AI/data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games`
  đã được tạo và kiểm tra.
- Trình tải trong explorer chưa mở được hộp chọn file qua CUA; automatic review từ chối
  điều khiển Chrome bằng Windows Computer Use với thông báo “Computer Use was not
  approved to use Google Chrome”. Vì vậy chưa có file nào được truyền lên FITLAB.
- Sau khi file có mặt trên FITLAB, chạy builder với `items.csv.gz`, `products_text.csv.gz`
  và 5 shard theo thứ tự; xác minh đúng **767.045 `parent_asin`** cùng report/hash, rồi mới
  chạy CLIP smoke/full extraction. Thiếu catalog/ảnh hợp lệ thì dừng, không tạo cache giả.
- Snapshot này tách biệt với các kết quả CLIP/UFM và demo lịch sử bên dưới; metric cũ
  không chứng minh job/checkpoint còn tồn tại hiện nay. Ảnh quota gửi trước hiển thị
  **80% quota tháng Codex còn lại**; quota này độc lập với disk/compute FITLAB.

SHA256 tham chiếu từ checkout local ngày 24/10; trên FITLAB, các tệp được chuẩn hóa
LF khi truyền nên đối chiếu nội dung sau chuẩn hóa dòng:

| Tệp | SHA256 |
|---|---|
| `src/ufm_model.py` | `65d2510ce5fae6f4be197634342a8aa7577e86ce6c7015a65ca95c00478ab532` |
| `src/train_ufm_recommender.py` | `a391936dcae4b13808a9bed30518d18a779244d4740bf0466f2db8e121200d1d` |
| `src/extract_foundation_features.py` | `9526ddb7a9cd17bb1e0ea9110c9120e6b1da244ff3cba56ff4fba78ac2deb0df` |
| `src/run_ufm_training_when_ready.py` | `38cf5ed1f21259ecf90f627b18585b442f1cf5e669e9f7d4413fa076d2912ef9` |
| `src/run_ufm_ablation_suite.py` | `683d4638718b5b7d4a8a696616aebd316a9097ba3b8b7f379567011dbbe20735` |
| `ops/resume_campaign.py` | `18a9bcaac0bb2cefe2d6cadd53a82d97639d2a3b26bd9469d235180c5440cccd` |
| `ops/monitor_ufm_campaign.py` | `8fd96c442ec995b7ddda2562ea754ef25387b7dd9aa00c938761cdf9e86494c3` |
| `requirements-ufm.txt` | `a1826686aeaba4a34ecdf1ed1646f3e4982fc0a8817823403909cc52f7de2d11` |

Đến mốc này đã khôi phục workspace, kiểm kê, thêm hỗ trợ Parquet và xác minh metadata
nguồn local; **chưa chuyển shard lên FITLAB, chưa dựng catalog, chưa chạy training hoặc
có metric GPU mới**. Snapshot FITLAB ngày 03/10 bên dưới là lịch sử, không phải trạng
thái hiện tại.

---

Snapshot FITLAB ngày 03/10 lúc **20:43 UTC+07**;
đây là bằng chứng tại thời điểm kiểm tra, không phải trạng thái realtime.

## Đã hoàn tất trong đợt này

- Kiểm tra checkpoint ablation `no_uncertainty` sau SIGTERM: đọc được trạng thái
  optimizer/RNG và model hữu hạn; phục hồi supervisor để resume đúng variant.
  Trainer/model UFM đang chạy được giữ nguyên.
- Bổ sung BERT4Rec thích nghi sampled-softmax và concat CLIP; hoàn thiện trainer
  SASRec dùng chung, checkpoint/resume, xử lý ID chưa train và lưu dự đoán validation.
- Triển khai hàng đợi 11 run Toys bổ sung: SASRec/BERT4Rec/concat với ba seed
  42, 7, 2026 và hai seed UFM còn thiếu. Có smoke GPU trước các baseline seed42.
- Chuẩn bị All Beauty: **445.239 tương tác, 86.014 item**, split thời gian riêng,
  TF-IDF validation, catalog và cache train. Hàng đợi kế tiếp gồm CLIP và 12 run
  production (bốn mô hình, ba seed); chưa có CLIP/benchmark GPU cho dataset này.
- Chạy [audit validation](ufm_validation_audit_v2/report.md), tái hiện cold macro
  từ saved predictions, paired bootstrap theo user và calibration theo thời gian.
  Phiên bản v2 sửa phép tính NDCG trên rank uint8; v1 cũ không dùng để báo cáo.
- Đồng bộ demo UFM trên FITLAB và vượt qua **4/4 ca API thật**: user mới, cá nhân hóa,
  zero-shot, warm. Giao diện qua proxy trả đúng 5 sản phẩm zero-shot từ lịch sử LEGO
  trong 4,47 giây. [Bằng chứng API](demo_acceptance_20261003.json),
  [ảnh giao diện](demo_zero_shot_20261003.png). [JSON xuất từ giao diện](demo_ui_export_20261003.json)
  đã tải về và kiểm tra đủ 5 item khác nhau, zero-shot, không chứa item lịch sử.
- Lưu bản local các kết quả FITLAB và kiểm tra SHA256 khi chuyển.

## Kết quả nghiên cứu đã có

Toys: 40.000 validation cases, 100 candidates/case; chưa phải benchmark full catalog.

| NDCG@10 validation | UFM | TF-IDF |
|---|---:|---:|
| Zero-shot | 0,188403 | 0,186491 |
| Extreme cold (1–5) | 0,113300 | 0,191555 |
| Cold (6–20) | 0,189511 | 0,195763 |
| Warm (>20) | 0,651231 | 0,467661 |
| Cold macro | 0,163738 | 0,191270 |

**UFM chưa đạt mục tiêu vượt TF-IDF ở cold-start.** Chênh lệch cold macro là
−0,027532; khoảng percentile bootstrap 95% theo user [−0,030221; −0,025082].
Nhóm extreme cold có history là điểm yếu lớn nhất (0,215933 so với 0,365075).
Đây là một seed và validation đã dùng chọn checkpoint; CI mang tính thăm dò,
không thay thế kết quả nhiều seed hoặc test độc lập.

**Lưu ý Test Shift:** Phân phối zero-shot bị shift đáng kể giữa huấn luyện và đánh giá (test shift). Để giải quyết bias số liệu từ zero-shot, hệ thống đã chuyển sang lấy mẫu negative theo popularity thay vì uniform. Việc tách known-history và empty-history cho thấy UFM và TF-IDF thực tế rất gần nhau trên nhóm có lịch sử.

Temperature 1,789353 fit trên 20.000 case sớm, audit trên 20.000 case muộn:
ECE **0,164477 → 0,041497**; NLL 4,309607 → 3,906583; Brier 0,963379 → 0,927849.
Calibration không đổi thứ hạng, không khắc phục thiếu hụt cold-start và chưa được
áp dụng làm xác suất mua hàng trong demo.

Demo API mất 2,89–7,27 giây mỗi ca trong bốn lượt đo đơn. Đây là thời gian inference
server, chưa phải p50/p95, throughput hay benchmark tài nguyên.

## Đang chờ và thứ tự tiếp tục

[Snapshot gốc](campaign_snapshot_20261003.json): GPU utilization **96%**, trống
18.678 MiB; ablation 0/7 hoàn tất. Các tiến trình tồn tại khi kiểm tra:
watchdog 178930, supervisor 192498, ablation queue 192499,
followup queue 193592, dataset2 queue 194229.

Kiểm tra mới lúc **21:24 UTC+07**: ablation đã chuyển sang `ablation_training`;
watchdog, followup và dataset2 đều còn heartbeat. Hai queue sau vẫn chờ theo thứ tự.
Các gói ZIP/B64 dùng chuyển file đã được xóa trên cả local và FITLAB sau khi xác nhận
nội dung đã triển khai; dữ liệu, checkpoint, log và bản sao phục hồi được giữ lại.

1. Chờ ablation đang huấn luyện hoàn tất 7 variant; sau đó mới chạy 11 benchmark/seed Toys.
2. Sau Toys followup, chạy CLIP và 12 benchmark/seed All Beauty. Validation All Beauty
   có **37.381 case** (cold chỉ 7.381); giữ toàn bộ mẫu có sẵn, không nhân bản để đủ 40k.
3. Tổng hợp mean/std theo seed, ablation, phân tích extreme cold; chốt cấu hình bằng
   validation và khóa checkpoint/config/candidate/temperature trong manifest.
4. Chạy test UFM một đợt theo manifest đã khóa; chưa thực hiện bước này. Baseline
   cổ điển đã có test lịch sử, phải công bố rõ trong báo cáo.
5. Đo RAM/VRAM/latency nhiều lượt và xác minh giới hạn 16 GiB; hoàn thiện bảng kết quả,
   báo cáo cuối và slide sau khi có benchmark/test thật.

Queue chỉ khởi chạy sau hai probe GPU rảnh; không ép tranh GPU. Hai queue mới chưa
nối autostart khi container reboot; cửa sổ khởi chạy lần lượt 7 và 14 ngày.
SIGTERM/lỗi invariant dừng để kiểm tra. Cách chạy lại và protocol:
[docs/followup.md](../docs/followup.md). Xem realtime bằng `python3 status.py` trên FITLAB.

## Kiểm thử

**14/14 kiểm thử CPU PASS**: model/masking/gradient, cold-ID/thiếu modality, bootstrap/dtype, kế hoạch và
đường dẫn dataset2, smoke train SASRec/BERT4Rec, route/Host/Origin demo và startup.
Chi tiết: [verification_20261003.txt](verification_20261003.txt).
- **Cập nhật 23/10/2026**: Đã hoàn thiện [báo cáo tổng kết](../docs/final_report.md) (kèm mục giới hạn thực nghiệm single-seed/validation-only) và [slide thuyết trình](../docs/presentation_slides.md), bổ sung [mock catalog offline](../scripts/build_mock_demo_data.py) cho demo local, xuất bằng chứng đo tài nguyên mẫu [benchmark_resources_content.json](benchmark_resources_content.json), và xác nhận lại **14/14 test suites PASS** (`53 passed, 3 skipped` trên Windows / Python 3.14.8 + PyTorch 2.14.1+cpu).
GPU smoke và benchmark dài còn chờ; không coi CPU PASS là nghiệm thu GPU.

Biểu đồ đã render và kiểm tra trực quan:
[NDCG theo regime](../output/figures/validation_20261003/ranking_by_regime.png),
[calibration](../output/figures/validation_20261003/calibration_reliability.png).
Cùng thư mục có SVG/PDF để đưa vào báo cáo; tái tạo bằng `scripts/plot_validation_audit.py`.

[Snapshot trước 02/10](Tien_do_UFM_2026-10-02.md) · [Nguồn All Beauty](second_dataset_preparation.json)
· [Kế hoạch dataset thứ hai đã khóa hash](second_dataset_plan_v1.json)
· [Protocol bổ sung](../docs/followup.md).
