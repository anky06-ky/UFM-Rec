# UFM-Rec: phục hồi chiến dịch FITLAB ngày 30/09/2026

## Trạng thái xác nhận

- Lượt CLIP trước bị `SIGKILL` lúc 09:19:16 UTC ngày 29/09, sau khi commit 497.856/767.045 sản phẩm (64,91%). Foundation queue báo `stopped_with_error`; UFM queue và ablation suite dừng theo. Chưa có epoch hoặc checkpoint UFM GPU.
- `progress.json` còn nguyên cấu hình CLIP ViT-B/32 đã ghim revision, batch 64, cùng catalog/text hash; `next_row=497856`, tổng các trạng thái ảnh bằng cursor. Chưa có `complete.json` full.
- Khi kiểm tra lại ngày 30/09, không có tiến trình extractor/runner cũ. GPU Quadro RTX 6000 còn 24.021 MiB và mức sử dụng 0% tại thời điểm đo. Python UFM cũ vẫn import được PyTorch 2.12.1+cu130, transformers 4.57.6 và thấy CUDA. Dung lượng thư mục iDragonCloud còn đủ theo `df`.
- Không xác định được nguyên nhân `SIGKILL`: `dmesg` không đọc được; `memory.events` hiện tại ghi `oom_kill=0`, nhưng chỉ phản ánh cgroup đang xem, không loại trừ việc máy chủ hoặc bộ điều phối đã dừng tiến trình trước đó. Không tự gán lỗi cho OOM.

## Đã thực hiện

- Khởi động lại ba runner nền bằng `nohup` và `flock` riêng, giữ nguyên mã, nguồn dữ liệu, revision, batch, checkpoint và log hiện có. Foundation queue tự dùng `--resume`; không tạo cache giả và không chạy hai job GPU cùng lúc.
- Foundation queue chuyển sang `full_extraction_running` lúc 03:02:20 UTC ngày 30/09. Log xác nhận `extracting 767,045 products from row 497,856 on cuda`. Cursor sau đó tăng lên **497.984/767.045** lúc 03:03:55 UTC, chứng minh lượt trích đã tiếp tục và ghi chunk mới.
- UFM queue trở lại `waiting_for_complete_features_and_idle_gpu`; ablation suite ở `waiting_for_full_ufm_and_idle_gpu`. Chưa đánh giá test. Không coi việc queue sống là kết quả huấn luyện.
- Bộ 32 correctness tests local chạy lại bằng `tmp/ufm_checks_env/Scripts/python.exe`: PASS. Kiểm thử synthetic không chứng minh chất lượng recommendation trên dữ liệu thật.

## Mốc tiếp theo

Theo dõi `python src/monitor_ufm_campaign.py` và log tại `runs/foundation_queue_v1.log`, `runs/ufm_training_queue_v1.log`, `runs/ufm_ablation_suite_v1.log` trên FITLAB. Khi CLIP đủ 767.045 dòng, xác nhận `complete.json`, hash, độ phủ và audit; queue UFM mới chạy GPU smoke/resume, full UFM rồi suite bảy ablation. Nếu lặp `SIGKILL`, thu thập RSS/VRAM và sự kiện từ FITLAB trước khi thay cấu hình hoặc khởi động lại. Runner vẫn phụ thuộc container FITLAB còn sống; suite có cửa sổ khởi chạy bảy ngày từ lần phục hồi này.
