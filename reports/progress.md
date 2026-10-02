# Tiến độ UFM-Rec

Cập nhật 02/10/2026, snapshot FITLAB mới lúc 18:23 UTC+07.

## Việc vừa hoàn tất

- Rà code, split manifest, metric và tài liệu; hợp nhất README/START để tránh lẫn mốc cũ.
- Viết [proposal 2.0](../docs/proposal.md), dựng [PDF 13 trang](../output/pdf/proposal.pdf),
  HTML và 5 hình vector; kiểm tra trực quan toàn bộ trang.
- Sửa demo nhận đường dẫn đã được code-server loại `/proxy/<port>` và dạng giữ prefix.
  Vẫn kiểm tra Host/Origin; 5 kiểm thử demo PASS trong môi trường `tmp/ufm_checks_env`.
- FITLAB mount đã đọc lại được. Marker full UFM đọc được; best.pt tồn tại,
  kích thước 396.724.716 byte. Đây là kiểm tra tồn tại/marker, chưa kiểm tra lại toàn bộ hash.
- Ablation dừng lúc 13:37 UTC+07 với `OSError: [Errno 5] Input/output error`.
  Không có variant hoàn tất; watchdog/supervisor cũ đã dừng.
- Khởi động lại watchdog qua `bash scripts/start_campaign.sh`; watchdog PID 178930,
  supervisor PID 178932, ablation queue PID 178948. Heartbeat mới 18:21-18:23 xác nhận
  `supervisor_active`, `ablations_running`, `waiting_for_full_ufm_and_idle_gpu`.
- Queue nhận `parent_complete=true`; GPU utilization 100%, trống 1.018 MiB, idle_checks=0.
  Không khởi chạy thêm trainer để tranh GPU; queue tiếp tục chờ tự động.

## Kết quả nghiên cứu đã có

| Hạng mục | Kết quả |
| --- | --- |
| Dữ liệu | 11.572.689 tương tác; catalog 767.045 item |
| CLIP | 767.045/767.045; 758.692 ảnh OK |
| UFM full | 12 epoch; best epoch 9; tổng 198.232 bước |
| NDCG@10 validation UFM | Overall 0,285611; cold macro 0,163738; warm 0,651231 |
| NDCG@10 validation TF-IDF | Overall 0,260367; cold macro 0,191270; warm 0,467661 |
| BPR-MF 3 seed | Cold macro 0,013602 ± 0,000233, độ lệch chuẩn mẫu |

Validation gồm 40.000 trường hợp, mỗi trường hợp 100 candidate; không là full-catalog
benchmark. UFM chưa vượt TF-IDF trên cold macro. Test UFM chưa chạy; baseline cổ điển
đã có test lịch sử. [Chi tiết snapshot UFM](Tien_do_UFM_2026-10-02.md).

## Chưa hoàn tất

1. Bảy ablation còn chờ GPU; chưa có kết quả để tổng hợp.
2. Bản sửa proxy đã kiểm thử local, chưa triển khai/nghiệm thu lại inference UFM trên
   FITLAB. Lỗi Bus error trước đó chưa đủ bằng chứng kết luận nguyên nhân.
3. SASRec full, BERT4Rec, concat hybrid, nhiều seed UFM, calibration, dataset thứ hai,
   VRAM 16GB, latency/RAM/VRAM và test cuối còn trong kế hoạch.
4. Watchdog phụ thuộc container/storage còn hoạt động; mất mount hoặc container vẫn
   có thể cần phục hồi. Không cam kết chạy không bao giờ gián đoạn.

Đã chuyển 18 file làm việc cũ và 4 thư mục cache vào archive có manifest,
giữ dữ liệu/model/môi trường. Lệnh xóa cache bị duyệt tự động chặn; chưa xóa vĩnh viễn
các file local đó. Danh sách cụ thể: [cleanup.json](cleanup.json).
