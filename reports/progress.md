# Tiến độ UFM-Rec

Cập nhật 03/10/2026. Snapshot FITLAB lúc **20:43 UTC+07**;
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
GPU smoke và benchmark dài còn chờ; không coi CPU PASS là nghiệm thu GPU.

Biểu đồ đã render và kiểm tra trực quan:
[NDCG theo regime](../output/figures/validation_20261003/ranking_by_regime.png),
[calibration](../output/figures/validation_20261003/calibration_reliability.png).
Cùng thư mục có SVG/PDF để đưa vào báo cáo; tái tạo bằng `scripts/plot_validation_audit.py`.

[Snapshot trước 02/10](Tien_do_UFM_2026-10-02.md) · [Nguồn All Beauty](second_dataset_preparation.json)
· [Kế hoạch dataset thứ hai đã khóa hash](second_dataset_plan_v1.json)
· [Protocol bổ sung](../docs/followup.md).
