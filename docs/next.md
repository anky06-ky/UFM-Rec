# Phần tiếp theo

Snapshot FITLAB 01/10/2026, 07:38 UTC+07: CLIP đã commit 594.304/767.045 sản phẩm
(77,48%). Lượt trước bị SIGKILL ngày 30/09, nên UFM và ablation trước đó đã dừng.
Supervisor đã resume CLIP từ cursor 594.304 lúc 07:52 UTC+07;
sau khi CLIP hoàn tất, nó sẽ chạy tiếp UFM rồi bảy ablation. Đây là mốc quan sát,
không phải số realtime. Cập nhật 08:13 UTC+07: CLIP đã tới 600.512/767.045
(78,29%); watchdog đang theo dõi supervisor và worker. Chi tiết trong
`reports/Progress_2026_10_01.md`.
Chạy `python status.py` hoặc notebooks/00_status.ipynb để đọc trạng thái mới.

## 1. Ưu tiên ngay: hoàn tất đặc trưng CLIP

- Giữ job extraction hiện tại chạy; không bật một lượt extraction trùng.
- Khi có complete.json, kiểm tra đủ 767.045 item, mapping/hash, vector hữu hạn,
  padding/mask, số ảnh tải thành công và breakdown theo cold-start regime.
- GPU dùng chung đang bận. Không đổi batch/config giữa cache đang chạy; nếu job
  dừng thì đọc log và xác nhận tiến trình trước khi resume theo docs/ops.md.
- Theo dõi `RECOVERY` và `FOUNDATION` bằng `python status.py`. Queue yêu cầu hai
  lượt kiểm tra GPU rảnh, VRAM còn ít nhất 8.192 MiB và utilization ≤20%.
  Nếu GPU bận trở lại ngay trước extraction, supervisor quay về chờ và watchdog
  tiếp tục giám sát trong container; không sửa cursor hoặc chạy trùng.

## 2. Huấn luyện UFM và ablation

- Queue UFM chờ đủ features và GPU rảnh, sau đó chạy smoke forward/backward/resume
  rồi train full. Điều kiện nghiệm thu là production completed.json và best checkpoint.
- Đọc validation overall, known-user, cold macro và từng regime. Chọn checkpoint
  bằng cold macro NDCG@10 đã chốt; chưa mở test để chọn cấu hình.
- Suite chạy 7 ablation từ đầu với cùng protocol/budget. Thu kết quả từng biến thể
  rồi so sánh contribution của uncertainty, fusion, alignment và từng modality.

## 3. Việc triển khai độc lập trong khi chờ GPU

Đã có mã baseline SASRec thích nghi trong `src/train_sasrec.py`, dùng self-attention
causal, train positive graph và cùng split/candidate validation; bài kiểm thử CPU
nhỏ đã chạy. Bước tiếp theo là smoke/full trên dữ liệu FITLAB sau khi các lượt
GPU ưu tiên hoàn tất, rồi BERT4Rec và concat hybrid theo phương pháp gốc. Trước
mỗi lượt full cần kiểm tra causal mask hoặc masked objective, negative sampling,
cold-ID và resume. Chưa có metric SASRec trên dataset thật.

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
