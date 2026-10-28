# Tiến độ UFM-Rec — 28/10/2026

Phạm vi giảng viên đã chốt theo xác nhận của nhóm: hệ gợi ý sản phẩm dựa trên hành vi, hình ảnh và tên/mô tả. Phân tích [validation v3 theo train count](continuation_20261028/count_effect_v3.json) đã hoàn tất trên dự đoán lưu sẵn; nhóm 1–5 tương tác có full − semantic-only NDCG@10 = −0,024101, nhóm 21+ là +0,014865. Hai biến thể count-aware và adaptive dropout đã được tách cơ chế trong mã. [Giao thức 15 lượt train](../configs/multimodal_followup_20261028.json) đã khóa và qua preflight 9 hash trên bản sao local; **chưa có lượt train mới** vì local không có CUDA và chưa truy cập được FITLAB.

Checkpoint UFM từ bản sao local đã khởi động trên Windows, 767.045 sản phẩm; [6/6 ca API](continuation_20261028/windows_ufm_verification.json) đạt. Kiểm thử mã hiện tại: **94 passed, 3 skipped, 10 subtests passed** trên Windows. Các phần lịch sử bên dưới giữ theo ngày ghi nhận, không thay thế trạng thái hiện tại.

[Benchmark Windows](continuation_20261028/windows_ufm_benchmark.json): 2 warm-up + 10 request đầy đủ, p50 21,25 giây, p95 26,35 giây, RSS server cao nhất quan sát 2.551,5 MiB; hai request đồng thời một 200 và một 429. Đây là giới hạn vận hành cần nêu rõ khi trình bày demo.

<!-- current-evidence-20261028 -->
> Cập nhật 28/10/2026: validation v3 đã hoàn tất (40.000 ca, 100 ứng viên/ca, seed 42). Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only 0,190036; UFM full 0,182056; popularity 0,042527**. Tám checkpoint UFM dùng cùng candidate hash. Đây là đánh giá lại checkpoint chọn bằng validation v1, chưa phải holdout độc lập. Test v2 đã được xem; vòng v3 này không đánh giá test. Chi tiết và giới hạn: [bằng chứng mới](continuation_20261028/README.md). Các số liệu ở phần lịch sử phải đọc theo phiên bản giao thức ghi kèm. Word/PDF cũ chưa đồng bộ cập nhật này.
<!-- /current-evidence-20261028 -->


Đối chiếu ngày **28/10/2026**: bộ sửa 1–15 trước đây đạt Linux **87 pass** / Windows **84 pass, 3 skip**. Mã bổ sung calibration/source loader đạt Linux **94 pass** / Windows **91 pass, 3 skip**; các bổ sung restore sau đó có kiểm thử riêng. Xem báo cáo sửa mục 1–15 và log để phân biệt bản mã hiện tại với bản mã đã tạo checkpoint.

Đã thực hiện phần kỹ thuật của mục 1–15; [bản tổng hợp và bằng chứng](fix_1_15_20261027/README.md) lưu bằng chứng của mốc cũ. Checkpoint/cache đã được xác minh trên máy chủ; baseline validation v2 đã chạy cùng hash; v3 đã tạo và audit. Phạm vi ba nguồn tín hiệu đã được nhóm xác nhận với giảng viên. GitHub Actions cho bản mã mới chưa được chạy; không suy luận test thêm.

Cold macro NDCG@10 validation v2: UFM 0,182051; TF-IDF 0,195672; popularity 0,042527. UFM chưa vượt baseline chính. Các mốc bên dưới giữ nguyên như lịch sử trước lần kiểm tra trực tiếp FITLAB-01 này; không dùng trạng thái thiếu checkpoint/baseline trong mốc cũ làm trạng thái hiện tại.

---

# Tiến độ UFM-Rec

## Cập nhật đối chiếu bằng chứng ngày 27/10/2026

Đã tìm thấy bộ kết quả `evaluation_v2` trong thư mục local `tmp/transfer/evaluation_v2` và lưu ba JSON nhỏ vào [reports/ufm_evaluation_v2_20261027](ufm_evaluation_v2_20261027/README.md). Hash SHA-256 của cả hai bộ mẫu NPZ và hai file metadata CSV gzip trên local khớp các hash trong report nhập về. Đây là đối chiếu artifact, chưa phải lượt huấn luyện hoặc đánh giá GPU được chạy lại trong lần chỉnh tài liệu này.

- **Test UFM đã được đánh giá trong artifact v2**: validation Overall/Cold macro NDCG@10 là **0,164132 / 0,182051**; test là **0,166193 / 0,183638**. Không tiếp tục mô tả toàn dự án bằng `test_evaluated = false`; cờ đó thuộc bộ kết quả lịch sử.
- Checkpoint trong report là `ufm_full_v1`, epoch 9, SHA-256 `5630013d0cbb6db633c4f4cf5bcd840fb703dbfc64a13c0352278624e03e11c8`. V2 dùng mẫu validation có hash `c71906e6…`, khác mẫu lịch sử `8efa1165…`; không ghép baseline hoặc temperature cũ vào bảng v2. Chưa có baseline chạy trên đúng mẫu v2 trong bộ bàn giao đã kiểm tra.
- Candidate audit v2 báo structure, exclusions, regimes và replay đều đạt. Sampling report dùng mô tả tổng quát; audit và code cho biết negative cùng regime với target, có trọng số theo số tương tác đến cuối split. Audit cũng ghi nhận 3 negative trùng positive train ngoài cửa sổ history ở validation và 7 ở test; vì vậy `passed=true` không có nghĩa toàn bộ negative đã được xác nhận là không liên quan.
- Bản ghi FITLAB ngày **24/10** bên dưới là trạng thái khôi phục tại thời điểm đó, không thể dùng để phủ nhận sự tồn tại của artifact v2 nhập về ngày 27/10. Chưa kiểm tra trực tiếp trạng thái máy chủ hoặc vị trí checkpoint hiện tại trong lần cập nhật này.
- Kết quả kiểm thử local ngày 27/10: `python -m pytest tests/ -q -p no:cacheprovider --tb=short` có **55 passed, 7 failed, 2 errors, 3 skipped** (Python 3.14.8, NumPy 2.5.4, PyTorch 2.14.1+cpu). Bộ cũ, bỏ hai file test mới, có **53 passed, 3 skipped**. Không ghi toàn bộ test đã đạt; các lệnh này không chạy lại huấn luyện hay mở thêm test recommendation.
- Bước tiếp theo: đối chiếu code/checkpoint và bộ mẫu v2; chạy baseline trên đúng validation v2; công bố việc test đã được xem và chốt giao thức đánh giá cho các biến thể sau, tránh chọn mô hình theo điểm test đã biết. Khôi phục tài nguyên FITLAB chỉ sau khi xác nhận tình trạng máy chủ thực tế.

---

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

## Hàng đợi và kế hoạch tại mốc 03/10 — thông tin lịch sử

Các bước bên dưới phản ánh kế hoạch tại mốc 03/10. Kế hoạch hiện tại đã được thay bằng mục đối chiếu 27/10 ở đầu tài liệu; không coi các job này là đang chạy và không coi test còn chưa được đánh giá.

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

[Snapshot trước 02/10](../docs/history/Tien_do_UFM_2026-10-02.md) · [Nguồn All Beauty](second_dataset_preparation.json)
· [Kế hoạch dataset thứ hai đã khóa hash](second_dataset_plan_v1.json)
· [Protocol bổ sung](../docs/followup.md).
