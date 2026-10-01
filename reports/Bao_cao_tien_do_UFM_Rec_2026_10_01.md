# Báo cáo tiến độ UFM-Rec

**Ngày cập nhật:** 01/10/2026, 22:17 UTC+07
**Dự án:** Gợi ý sản phẩm cold-start trên Amazon Reviews 2023 · Toys & Games

## Tóm tắt

Pipeline dữ liệu và đặc trưng CLIP đã hoàn tất trên catalog **767.045 sản phẩm**. UFM đã phục hồi sau lỗi tràn gradient AMP và tiếp tục huấn luyện. Epoch 2 đã được đánh giá trên validation; tại lần kiểm tra 22:17, epoch 3 đang chạy ở batch **5.962/16.527**, bước **39.000**. Watchdog và supervisor còn sống. Demo TF-IDF hoạt động trên dữ liệu thật và vừa được làm mới giao diện.

UFM full, bảy thí nghiệm ablation, kiểm tra test cuối và demo dùng checkpoint UFM vẫn đang chờ hoàn tất. Vì vậy chưa thể kết luận UFM tốt hơn các baseline.

## Trạng thái hạng mục

| Hạng mục | Trạng thái hiện tại | Bằng chứng / ghi chú |
|---|---|---|
| Dữ liệu theo thời gian | Hoàn tất | 11.572.689 tương tác đã lọc; catalog 767.045 sản phẩm |
| Đặc trưng CLIP | Hoàn tất | 767.045 sản phẩm; 758.692 ảnh đọc thành công; marker và kiểm tra checksum đã qua |
| Baseline TF-IDF, SVD, RRF | Có kết quả | Đã có kết quả trên protocol đánh giá lấy mẫu; xem báo cáo 30/09 |
| BPR-MF | Hoàn tất 3 seed trên validation | Cold macro NDCG@10 trung bình 0,013602 ± 0,000233; chưa đánh giá test |
| Calibration TF-IDF | Hoàn tất bước thăm dò | ECE top-1 giảm từ 0,168601 xuống 0,116684 trên audit 20.000 mẫu |
| UFM | Đang train | Epoch 2 đã ghi `best.pt`; epoch 3 ở bước 39.000 lúc 22:17; production checkpoint chưa hoàn tất |
| Bảy ablation | Đang xếp hàng | Queue chạy sau UFM full |
| Demo | Hoạt động với TF-IDF | Giao diện mới; luồng tìm LEGO → chọn lịch sử → zero-shot → Top 5 đã chạy thành công |
| SASRec, BERT4Rec, hybrid nối đặc trưng | Còn phải làm | Chưa có benchmark đầy đủ trên dữ liệu thật |

## Kết quả UFM bước đầu

Epoch 2 hoàn tất tại bước **33.040**. Trên tập validation cố định 40.000 trường hợp (mỗi trường hợp gồm 1 positive và 99 negative), UFM đạt overall NDCG@10 **0,274583**, cold macro **0,150295**, warm **0,647448**; zero-shot NDCG@10 là **0,174892**. Checkpoint tốt nhất hiện là `best.pt` ở epoch 2. Đây là kết quả validation, chưa phải kết quả test.

| Validation NDCG@10 | Overall | Cold macro | Warm |
|---|---:|---:|---:|
| TF-IDF content profile | 0,260367 | 0,191270 | 0,467661 |
| UFM, epoch 2 | 0,274583 | 0,150295 | 0,647448 |

Hai dòng dùng cùng candidate protocol 40.000 mẫu. Ở epoch 2, UFM cao hơn ở overall và warm, nhưng thấp hơn TF-IDF ở cold macro. Chưa khóa lựa chọn cuối; tiếp tục chọn checkpoint bằng validation cold macro và chỉ đánh giá test sau khi cấu hình được khóa.

Trước đó trainer dừng vì gradient AMP không hữu hạn. Mã đã được sửa để hạ scale và bỏ qua batch tràn an toàn; checkpoint bước 6.000 được tiếp tục. Trong epoch 2, overflow tại bước 33.002 làm GradScaler hạ scale; epoch vẫn hoàn tất tại bước 33.040 và validation được ghi. Lúc 22:17 epoch 3 đã tới bước 39.000. Peak VRAM quan sát khoảng **3,70 GiB**.

## Demo sau khi làm mới

- Hero xanh rừng, tương phản rõ và làm nổi bật quy mô catalog.
- Thẻ kết quả có ảnh, nhãn cold-start, thứ hạng, số tương tác và nhãn giải thích **Điểm xếp hạng**.
- Bổ sung trạng thái hover/focus, khoảng cách và đường viền dễ nhìn; bố cục thích ứng với màn hình nhỏ.
- Chạy bằng backend **TF-IDF content baseline**, không phải UFM. Luồng zero-shot Top 5 từ một sản phẩm LEGO trả đủ 5 kết quả; lần gọi hiển thị 0,17 giây xử lý trên máy demo.
- Điểm là score xếp hạng, không phải xác suất mua hàng. Thời gian trên là một lần chạy, chưa phải benchmark tải.

## Việc tiếp theo

1. Để UFM hoàn tất toàn bộ epoch và xác nhận `completed.json`, cấu hình, checkpoint tốt nhất cùng validation.
2. Chạy đủ bảy ablation theo cùng protocol và lưu kết quả cold/warm cho từng biến thể.
3. Chạy SASRec và các baseline kiến trúc còn thiếu; đo throughput, độ trễ và VRAM.
4. Khóa lựa chọn bằng validation rồi mới chạy đánh giá test có kiểm soát.
5. Chuyển checkpoint UFM đã kiểm chứng vào demo, nghiệm thu các luồng và hoàn thiện báo cáo/slide cuối.

## Cách tiếp tục và tài liệu

Trong FITLAB, chạy `python3 status.py` để xem trạng thái và log hiện thời. Watchdog được cài để tự khởi động campaign khi mở lại workspace; nếu container bị tắt hẳn thì workspace cần được mở lại để tiến trình tiếp tục.

- Tiến trình và log chi tiết: [Progress_2026_10_01.md](Progress_2026_10_01.md)
- Việc tiếp theo: [docs/next.md](../docs/next.md)
- Chạy và trình bày demo: [docs/demo.md](../docs/demo.md)
- Kiểm tra nghiệm thu demo: [demo_acceptance_2026_09_30.json](demo_acceptance_2026_09_30.json)
