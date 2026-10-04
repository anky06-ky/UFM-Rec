# Báo cáo tiến độ UFM-Rec

**Cập nhật:** 02/10/2026, 18:04 UTC+07
**Dự án:** Gợi ý sản phẩm cold-start · Amazon Reviews 2023 · Toys & Games

## Tóm tắt

Pipeline dữ liệu và CLIP đã hoàn tất trên **767.045 sản phẩm**; có **758.692 ảnh
đọc thành công**. Huấn luyện UFM full đã hoàn thành 12 epoch, tổng cộng **198.232
bước**. Checkpoint tốt nhất theo cold macro NDCG@10 là epoch 9, bước 148.674.
Training dừng sau ba epoch liên tiếp không cải thiện; checkpoint tốt nhất vẫn được
giữ lại.

UFM cải thiện NDCG@10 overall và warm so với TF-IDF theo cùng giao thức validation,
nhưng cold macro còn thấp hơn. Snapshot lúc 07:59 ghi bảy ablation chờ GPU chia sẻ
rảnh; khi đó GPU dùng **100%**, còn **8.352 MiB**. Trạng thái campaign hiện chưa
kiểm tra lại được do FITLAB trả lỗi I/O ở mount dự án. Dữ liệu test chưa được dùng.

## Trạng thái hạng mục

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Dữ liệu temporal | Hoàn tất | 11.572.689 tương tác đã lọc; catalog 767.045 sản phẩm |
| Đặc trưng CLIP | Hoàn tất | 767.045 vector; 758.692 ảnh OK; marker/checksum hợp lệ |
| TF-IDF, SVD, RRF | Có baseline validation | Cùng candidate protocol; chi tiết trong báo cáo 30/09 |
| BPR-MF | Hoàn tất 3 seed validation | Cold macro NDCG@10 trung bình 0,013602 ± 0,000233 |
| UFM full | Hoàn tất theo snapshot 07:59; checkpoint hiện chưa truy cập được | 12 epoch, 198.232 bước; best epoch 9, bước 148.674 |
| Bảy ablation | Trạng thái chưa xác minh | Snapshot 07:59 là `waiting_for_full_ufm_and_idle_gpu`; workspace sau đó mất kết nối |
| Demo UFM | Backend status đã phản hồi; inference chưa nghiệm thu | API báo UFM Rec, 767.045 sản phẩm; request inference tiếp theo kết thúc bằng `Bus error` |
| SASRec, BERT4Rec, hybrid | Chưa benchmark full | Còn cần cùng protocol và đo tài nguyên |
| Đánh giá test | Chưa chạy | Giữ test tách biệt đến khi khóa cấu hình bằng validation |

## Cập nhật FITLAB lúc 18:04 UTC+07

- API UFM trả status thành công lúc 17:56 UTC+07: `backend=UFM Rec`, catalog 767.045.
- Code-server proxy `/proxy/8766/` vẫn trả `Loopback origin required`. Danh sách
  Ports cho thấy server PID 72211 chạy với tùy chọn cũ, chưa có origin/base path.
- Sau request inference tiếp theo, shell hiện `[1]+ Lỗi bus` cho tiến trình UFM;
  cổng 8766 không còn tiến trình nghe. Log `runs/demo_ufm_v1.log` không đọc được.
- Code-server báo workspace không tồn tại/mất kết nối. `stat` thư mục
  `/home/coder/iDragonCloud` và `/home/coder/iDragonCloud/DA_AI` trả `EIO`; vì vậy
  chưa thể kiểm tra `status.py`, queue, log hoặc checkpoint sau thời điểm này.
- Trang đang mở tại `127.0.0.1:8766` nhận diện chính xác **TF-IDF content baseline**.
  Lọc zero-shot trả 10 sản phẩm trong 5,31 giây; đây là thử giao diện baseline,
  không phải nghiệm thu UFM qua proxy.

Chưa đủ bằng chứng để kết luận nguyên nhân `Bus error`; thời điểm lỗi trùng với
mount FITLAB không đọc được. Cần khôi phục mount rồi kiểm tra log/checkpoint trước
khi khởi động lại UFM hoặc tiếp tục ablation.

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

1. Khôi phục workspace/mount FITLAB; xác nhận checkpoint và queue còn đọc được rồi
   mới chạy `python3 status.py`. Không khởi chạy job nếu mount còn lỗi `EIO`.
2. Kiểm tra log của `Bus error`; khởi động lại demo với origin/base path proxy đúng,
   rồi xác minh status, tìm kiếm và một request inference UFM có kiểm soát.
3. Khi workspace ổn định, để supervisor tiếp tục bảy ablation khi GPU chia sẻ rảnh;
   không khởi chạy suite thứ hai.
4. Chạy benchmark SASRec, BERT4Rec và hybrid trên cùng split/protocol; báo latency,
   throughput và VRAM.
5. Khóa cấu hình theo validation, chạy một lượt test có kiểm soát, rồi tổng hợp kết
   quả, baseline và giới hạn cold-start vào báo cáo/slide cuối.

Trạng thái sống và hướng dẫn hồi phục nằm trong `docs/next.md`, `docs/recovery.md`
và `docs/demo.md`.
