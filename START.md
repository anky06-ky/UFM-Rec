# Bắt đầu

1. Đọc [tiến độ](reports/progress.md) và [proposal mới](output/pdf/proposal.pdf).
2. Demo local: mở [Start_Demo.cmd](Start_Demo.cmd); cần dữ liệu TF-IDF trong máy.
3. FITLAB: chạy `python3 status.py` để đọc tiến trình, heartbeat và PID.
4. Nếu campaign dừng, dùng [recovery](docs/recovery.md); không chạy thêm queue
   khi watchdog/supervisor còn sống.

CLIP và UFM full đã hoàn tất. Ngày 03/10 đã phục hồi ablation, triển khai queue
benchmark ba seed và All Beauty; demo UFM vượt qua bốn ca API thật.
Các lượt GPU còn chờ. Trạng thái realtime phải đọc trên FITLAB; hai queue bổ sung
chưa tự khởi động sau reboot. Lệnh phục hồi: [docs/followup.md](docs/followup.md).

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
