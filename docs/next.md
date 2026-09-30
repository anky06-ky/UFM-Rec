# Phần tiếp theo

Snapshot FITLAB 30/09/2026, 14:33 UTC+07: CLIP đã commit 588.864/767.045 sản phẩm
(khoảng 76,77%); UFM và ablation đang chờ. Đây là mốc quan sát, không phải số realtime.
Chạy `python status.py` hoặc notebooks/00_status.ipynb để đọc trạng thái mới.

## 1. Ưu tiên ngay: hoàn tất đặc trưng CLIP

- Giữ job extraction hiện tại chạy; không bật một lượt extraction trùng.
- Khi có complete.json, kiểm tra đủ 767.045 item, mapping/hash, vector hữu hạn,
  padding/mask, số ảnh tải thành công và breakdown theo cold-start regime.
- GPU dùng chung đang bận. Không đổi batch/config giữa cache đang chạy; nếu job
  dừng thì đọc log và xác nhận tiến trình trước khi resume theo docs/ops.md.

## 2. Huấn luyện UFM và ablation

- Queue UFM chờ đủ features và GPU rảnh, sau đó chạy smoke forward/backward/resume
  rồi train full. Điều kiện nghiệm thu là production completed.json và best checkpoint.
- Đọc validation overall, known-user, cold macro và từng regime. Chọn checkpoint
  bằng cold macro NDCG@10 đã chốt; chưa mở test để chọn cấu hình.
- Suite chạy 7 ablation từ đầu với cùng protocol/budget. Thu kết quả từng biến thể
  rồi so sánh contribution của uncertainty, fusion, alignment và từng modality.

## 3. Việc triển khai độc lập trong khi chờ GPU

Ưu tiên tiếp theo về code: bổ sung SASRec, rồi BERT4Rec và concat hybrid theo đúng
phương pháp gốc, dùng cùng split/candidate set. Trước mỗi lượt full cần kiểm thử
causal mask hoặc masked objective, negative sampling, cold-ID và resume.

Sau đó chuẩn bị MovieLens-1M theo thời gian cho dataset thứ hai. Nếu thiếu ảnh,
phải mô tả đây là sanity check tuần tự/nội dung, không suy diễn đã kiểm chứng đa phương thức.
BPR-MF đã có 3 seed; điều này chưa thay thế nhiều seed UFM.

## 4. Sau khi có kết quả UFM

1. Thêm nhiều seed UFM, uncertainty/gate theo regime và calibration UFM trên validation.
2. Đo số tham số, peak VRAM, throughput/latency và khả năng chạy với giới hạn 16 GiB.
3. Khóa cấu hình/temperature bằng validation; đánh giá test một đợt có kiểm soát.
4. Chuyển checkpoint full và cache hợp lệ vào demo UFM; hoàn thiện báo cáo, biểu đồ và slide.

## Đã có để sử dụng ngay

Pipeline temporal, các baseline cổ điển, BPR-MF ba seed, calibration TF-IDF thăm dò,
demo TF-IDF trên toàn catalog và báo cáo tiến trình ngày 30/09 đã có trong GitHub.
Chưa có bằng chứng UFM vượt baseline; mục tiêu cuối cùng cần kết quả thực nghiệm thật.
