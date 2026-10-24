# Bắt đầu

1. Đọc [tiến độ](reports/progress.md) và [proposal mới](output/pdf/proposal.pdf).
2. Demo local: mở [Start_Demo.cmd](Start_Demo.cmd); cần dữ liệu TF-IDF trong máy.
3. FITLAB: chạy `python3 status.py` tại thư mục `DA_AI` để đọc heartbeat/PID.
   Snapshot 12:28 ngày 24/10 báo CLIP 0/767.045; chưa có queue/checkpoint.
4. Môi trường CUDA đã sẵn sàng tại `/home/coder/.venv-ufm/bin/python`;
   `torch.cuda.is_available()` là True. Chưa chạy campaign vì catalog metadata và
   URL ảnh còn thiếu. Năm shard metadata public đã được tải và kiểm checksum ở local;
   xem [bước tiếp theo](docs/next.md) để chuyển đầu vào sang FITLAB và dựng catalog.

Kết quả CLIP/UFM và demo API trong báo cáo là kết quả lịch sử. Workspace FITLAB-02
được khôi phục từ archive ngày 24/09 và source mới đã đồng bộ; chưa có cache
CLIP, queue, checkpoint hoặc metadata catalog cần cho backend UFM. Dùng
`Start_Demo.cmd` để mở demo local.

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
