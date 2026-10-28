# Bắt đầu

<!-- current-evidence-20261028 -->
> Cập nhật 28/10/2026: validation v3 đã hoàn tất (40.000 ca, 100 ứng viên/ca, seed 42). Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only 0,190036; UFM full 0,182056; popularity 0,042527**. Tám checkpoint UFM dùng cùng candidate hash. Đây là đánh giá lại checkpoint chọn bằng validation v1, chưa phải holdout độc lập. Test v2 đã được xem; vòng v3 này không đánh giá test. Chi tiết và giới hạn: [bằng chứng mới](reports/continuation_20261028/README.md). Các số liệu ở phần lịch sử phải đọc theo phiên bản giao thức ghi kèm. Word/PDF cũ chưa đồng bộ cập nhật này.
<!-- /current-evidence-20261028 -->


1. Đọc [bằng chứng cập nhật 27/10](reports/ufm_evaluation_v2_20261027/README.md) trước khi dùng số liệu: UFM v2 đã có test, khác bộ mẫu lịch sử. Sau đó đọc [tiến độ](reports/progress.md) và [proposal mới](output/pdf/proposal.pdf).
2. Demo local: mở [Start_Demo.cmd](Start_Demo.cmd); backend content, tự tạo catalog mẫu nếu thiếu dữ liệu. Đây không phải demo checkpoint UFM.
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
| Bằng chứng mới | [evaluation_v2](reports/ufm_evaluation_v2_20261027/README.md) |

Dataset, checkpoint, môi trường Python và model cache được giữ nguyên. File tạm
chuyển vào archive có manifest; dữ liệu/model lớn không đẩy lên GitHub.
