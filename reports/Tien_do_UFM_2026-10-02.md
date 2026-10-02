# Báo cáo tiến độ UFM-Rec

**Cập nhật:** 02/10/2026, 07:59 UTC+07
**Dự án:** Gợi ý sản phẩm cold-start · Amazon Reviews 2023 · Toys & Games

## Tóm tắt

Pipeline dữ liệu và CLIP đã hoàn tất trên **767.045 sản phẩm**; có **758.692 ảnh
đọc thành công**. Huấn luyện UFM full đã hoàn thành 12 epoch, tổng cộng **198.232
bước**. Checkpoint tốt nhất theo cold macro NDCG@10 là epoch 9, bước 148.674.
Training dừng sau ba epoch liên tiếp không cải thiện; checkpoint tốt nhất vẫn được
giữ lại.

UFM cải thiện NDCG@10 overall và warm so với TF-IDF theo cùng giao thức validation,
nhưng cold macro còn thấp hơn. Bảy ablation đang chờ GPU chia sẻ rảnh; lần kiểm tra
FITLAB ghi nhận GPU **100% utilization**, còn **8.352 MiB**. Recovery supervisor và
watchdog vẫn hoạt động. Dữ liệu test chưa được dùng.

## Trạng thái hạng mục

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Dữ liệu temporal | Hoàn tất | 11.572.689 tương tác đã lọc; catalog 767.045 sản phẩm |
| Đặc trưng CLIP | Hoàn tất | 767.045 vector; 758.692 ảnh OK; marker/checksum hợp lệ |
| TF-IDF, SVD, RRF | Có baseline validation | Cùng candidate protocol; chi tiết trong báo cáo 30/09 |
| BPR-MF | Hoàn tất 3 seed validation | Cold macro NDCG@10 trung bình 0,013602 ± 0,000233 |
| UFM full | Hoàn tất | 12 epoch, 198.232 bước; best epoch 9, bước 148.674 |
| Bảy ablation | Đang chờ GPU | `waiting_for_full_ufm_and_idle_gpu`; GPU chia sẻ bận, supervisor sống |
| Demo UFM | Backend đã nạp trên FITLAB | CPU, catalog 767.045; đang kiểm tra route qua proxy code-server |
| SASRec, BERT4Rec, hybrid | Chưa benchmark full | Còn cần cùng protocol và đo tài nguyên |
| Đánh giá test | Chưa chạy | Giữ test tách biệt đến khi khóa cấu hình bằng validation |

## Kết quả UFM

Checkpoint epoch 9 được chọn theo cold macro trên validation cố định 40.000 mẫu;
mỗi mẫu có 1 positive và 99 negative. Đây là sampled ranking, không phải đánh giá
xếp hạng toàn catalog.

| NDCG@10 validation | TF-IDF | UFM epoch 9 |
|---|---:|---:|
| Overall | 0,260367 | **0,285611** |
| Cold macro | **0,191270** | 0,163738 |
| Zero-shot | — | 0,188403 |
| Extreme-cold | — | 0,113300 |
| Cold | — | 0,189511 |
| Warm | 0,467661 | **0,651231** |

UFM thắng overall và warm, nhưng thua TF-IDF ở cold macro. Do mục tiêu chính là
cold-start, kết quả hiện tại chưa chứng minh UFM vượt baseline cho mục tiêu đó.
Không chọn mô hình bằng test; chưa có kết quả test để báo cáo.

## Huấn luyện và độ ổn định

Trainer đã vượt qua lỗi AMP gradient overflow trước đó bằng cơ chế hạ scale và bỏ
qua optimizer step không an toàn. Full run đã hoàn thành; các epoch cuối không vượt
best epoch 9 và training dừng theo điều kiện không cải thiện. Marker `completed.json`,
12 file kết quả epoch và `best.pt` được xác nhận trên FITLAB. Không cần chạy lại full
UFM để tiếp tục các thí nghiệm kế tiếp.

## Ablation và demo

Recovery supervisor đang giữ thứ tự campaign, suite bảy ablation được đặt trạng thái
chờ GPU rảnh. Lần kiểm tra ghi utilization 100% và 8.352 MiB trống; đây là tải GPU
chia sẻ, không phải bằng chứng suite đã bắt đầu. Không khởi chạy suite thủ công khi
supervisor còn sống để tránh tranh GPU hoặc chạy trùng.

Demo UFM tải được checkpoint và catalog trên CPU. Code-server chuyển cổng qua
`/proxy/8766/`; bản demo ban đầu trả lỗi `Loopback origin required` vì chưa hỗ trợ
origin/path của reverse proxy. Bản sửa đang được kiểm tra end-to-end trước khi công
bố demo sẵn sàng. Khi nghiệm thu, cần xác nhận status, tìm kiếm, zero-shot, Top 5 và
JSON kết quả.

## Bước tiếp theo

1. Theo dõi `python3 status.py` và để supervisor chạy bảy ablation khi GPU chia sẻ
   rảnh; không khởi chạy thêm một suite khác.
2. Hoàn tất kiểm tra demo UFM qua proxy FITLAB.
3. Chạy benchmark SASRec, BERT4Rec và hybrid nối đặc trưng trên cùng split/protocol;
   báo latency, throughput và VRAM.
4. Khóa cấu hình theo validation. Sau đó chạy một lượt test có kiểm soát và lưu
   kết quả riêng.
5. Tổng hợp ablation, baseline và giới hạn cold-start vào báo cáo/slide cuối.

Trạng thái sống và hướng dẫn hồi phục nằm trong `docs/next.md`, `docs/recovery.md`
và `docs/demo.md`.
