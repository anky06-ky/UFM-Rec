# Tiến độ và bước tiếp theo

**Cập nhật:** 02/10/2026, 07:59 UTC+07 · FITLAB `status.py`

## Trạng thái hiện tại

- Đặc trưng CLIP hoàn tất: **767.045/767.045 sản phẩm**, **758.692 ảnh OK**.
- UFM full hoàn tất **12 epoch, 198.232 bước**. Checkpoint tốt nhất là epoch 9,
  bước 148.674; dừng sớm sau ba epoch không cải thiện.
- Validation tốt nhất: overall NDCG@10 **0,285611**, cold macro **0,163738**,
  zero-shot **0,188403**, extreme-cold **0,113300**, cold **0,189511**,
  warm **0,651231**. Đây là protocol 40.000 mẫu, mỗi mẫu 1 positive + 99 negative;
  chưa phải xếp hạng toàn catalog và chưa dùng test.
- So với TF-IDF cùng protocol, UFM cao hơn overall (**0,285611 / 0,260367**) và
  warm (**0,651231 / 0,467661**), nhưng thấp hơn cold macro
  (**0,163738 / 0,191270**). Chưa thể kết luận UFM tốt hơn toàn diện.
- Bảy ablation đang được recovery supervisor giữ hàng đợi. Lúc kiểm tra GPU dùng
  **100%**, còn **8.352 MiB**; campaign chưa có tiến trình GPU riêng đang chạy,
  nên suite chờ GPU chia sẻ rảnh. Watchdog và supervisor còn sống.
- Demo UFM đã nạp trên FITLAB bằng CPU với **767.045 sản phẩm**. Đang sửa tương
  thích reverse proxy `/proxy/8766/` để trang và API truy cập được qua code-server.

## Việc làm tiếp theo

1. Để supervisor tự chạy đủ bảy ablation khi GPU chia sẻ rảnh; theo dõi
   `python3 status.py`, `runs/ufm_ablation_suite_v1.json` và log trong `runs/`.
   Không khởi chạy suite thứ hai song song.
2. Kiểm tra và nghiệm thu demo UFM qua proxy: tải trang, trạng thái backend, tìm
   sản phẩm, zero-shot, Top 5 và JSON xuất kết quả.
3. Sau ablation, chạy benchmark SASRec, BERT4Rec và hybrid nối đặc trưng với cùng
   split/candidate protocol; đo latency, throughput và VRAM.
4. Chốt mô hình bằng validation, sau đó mới chạy một lượt test có kiểm soát.
   Báo cáo rõ hạn chế nếu cold-start vẫn kém TF-IDF.
5. Hoàn thiện báo cáo tổng hợp và slide sau khi có kết quả ablation/benchmark.

## Khởi động lại FITLAB

Mở lại workspace `DA_AI`; watchdog sẽ hồi phục campaign. Kiểm tra bằng:

```bash
python3 status.py
```

Nếu trạng thái không tiến triển, xem `docs/recovery.md` trước khi khởi chạy lại.

Chi tiết snapshot: [Báo cáo tiến độ UFM](../reports/Tien_do_UFM_2026-10-02.md).
