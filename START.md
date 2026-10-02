# Bắt đầu

1. Đọc [tiến độ](reports/progress.md) và [proposal mới](output/pdf/proposal.pdf).
2. Demo local: mở [Start_Demo.cmd](Start_Demo.cmd); cần dữ liệu TF-IDF trong máy.
3. FITLAB: chạy `python3 status.py` để đọc tiến trình, heartbeat và PID.
4. Nếu campaign dừng, dùng [recovery](docs/recovery.md); không chạy thêm queue
   khi watchdog/supervisor còn sống.

CLIP và UFM full đã hoàn tất. Watchdog/supervisor đã phục hồi ngày 02/10;
bảy ablation còn đợi GPU. Trạng thái realtime phải đọc trên FITLAB.

| Cần tìm | Đường dẫn |
| --- | --- |
| Đề xuất nghiên cứu | [docs/proposal.md](docs/proposal.md) |
| Việc tiếp theo | [docs/next.md](docs/next.md) |
| Demo và proxy | [docs/demo.md](docs/demo.md) |
| Training / CLIP | [docs/train.md](docs/train.md) · [docs/foundation.md](docs/foundation.md) |
| Kết quả / lịch sử | [reports/README.md](reports/README.md) |
| File đã dọn | [reports/cleanup.json](reports/cleanup.json) |

Dataset, checkpoint, môi trường Python và model cache được giữ nguyên. File tạm
chuyển vào archive có manifest; dữ liệu/model lớn không đẩy lên GitHub.
