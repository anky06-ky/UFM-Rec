# Phần tiếp theo

Snapshot FITLAB 01/10/2026, 22:30 UTC+07: CLIP đã hoàn thành 767.045/767.045
sản phẩm, 758.692 ảnh OK. UFM epoch 4 đang chạy, bước 51.000; `best.pt` hiện
ghi kết quả epoch 2, cold macro NDCG@10 = 0,150295 (validation 40.000 mẫu).
Watchdog và supervisor đang sống, VRAM khoảng 3,70 GiB. Bảy ablation sẽ chạy
sau khi full UFM hoàn tất.
Đây là mốc quan sát, không phải số realtime. Chi tiết trong
`reports/Progress_2026_10_01.md`.
Tóm tắt ngắn: `reports/Bao_cao_tien_do_UFM_Rec_2026_10_01.md`.
Chạy `python status.py` hoặc notebooks/00_status.ipynb để đọc trạng thái mới.

## 1. Đặc trưng CLIP đã hoàn tất

- `complete.json` ghi đủ 767.045 item; queue UFM đã kiểm tra mapping/hash, vector
  hữu hạn, padding/mask và độ phủ trước khi bắt đầu training.
- Giữ `data/processed/toys_games_full_temporal/foundation_clip_b32_v1` và marker
  hoàn tất để UFM và ablation dùng lại cùng nguồn.

## 2. Huấn luyện UFM và ablation

- Queue UFM đã qua smoke/resume và đang train full ở epoch 4, bước 51.000.
  Điều kiện nghiệm thu là production `completed.json` và best checkpoint.
- File trạng thái ablation còn ghi lần dừng cũ ngày 30/09 khi CLIP bị SIGKILL.
  Recovery supervisor hiện đang ở stage UFM và chạy queue theo thứ tự; sau khi
  full marker xuất hiện, suite sẽ được khởi chạy lại từ đầu, không cần chạy tay.
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
