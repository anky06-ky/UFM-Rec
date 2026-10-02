# Cập nhật phục hồi 02/10/2026, 18:23 UTC+07

Mount đã đọc lại được. Watchdog PID 178930, supervisor 178932 và ablation queue
178948 có heartbeat mới; queue chờ GPU 100% utilization, trống 1.018 MiB.
Bản sửa proxy demo đã qua 5 kiểm thử local, chưa triển khai/nghiệm thu inference
UFM trên FITLAB. Xem [tiến độ hiện tại](../reports/progress.md) và
[proposal 2.0](proposal.md). Bước thực thi kế tiếp là để queue chạy ablation khi GPU
rảnh; không tạo queue thứ hai. Phần dưới là snapshot trước khi phục hồi mount.

---

# Tiến độ và bước tiếp theo

**Cập nhật:** 02/10/2026, 18:04 UTC+07 · kiểm tra FITLAB và demo

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
- Snapshot `status.py` lúc 07:59 ghi bảy ablation chờ GPU chia sẻ rảnh (GPU 100%,
  còn 8.352 MiB); watchdog và supervisor còn sống khi đó. Trạng thái hiện tại chưa
  xác minh được vì workspace FITLAB trả `EIO`.
- Lúc 17:56 UTC+07, API demo trả backend `UFM Rec`, catalog **767.045 sản phẩm**.
  Request suy luận tiếp theo làm tiến trình thoát với `Bus error`. Code-server sau
  đó báo workspace mất kết nối; `/home/coder/iDragonCloud` và thư mục dự án đều
  không đọc được (`EIO`). Chưa xác nhận nguyên nhân và chưa nghiệm thu proxy.
- Giao diện demo đang mở ở `127.0.0.1:8766` là **TF-IDF content baseline**; đã thử
  zero-shot, nhận 10 sản phẩm trong 5,31 giây. Đây không phải inference UFM.

## Việc làm tiếp theo

1. Khôi phục kết nối/mount FITLAB trước. Xác nhận `ls -ld /home/coder/iDragonCloud/DA_AI`,
   dung lượng và khả năng đọc `runs/ufm_full_v1/best.pt`; chạy `python3 status.py`
   để xác nhận watchdog, queue và GPU. Chưa khởi chạy lại job khi mount còn `EIO`.
2. Đọc log demo và kiểm tra checkpoint sau khi mount hoạt động. Khởi động lại UFM
   với origin/base path proxy đã cấu hình; kiểm tra `/api/status`, rồi thử một
   request suy luận có kiểm soát trước khi nghiệm thu giao diện.
3. Khi workspace ổn định, để supervisor tiếp tục bảy ablation theo queue; không
   khởi chạy suite thứ hai song song.
4. Sau ablation, chạy benchmark SASRec, BERT4Rec và hybrid trên cùng split/candidate
   protocol; đo latency, throughput và VRAM.
5. Chốt mô hình bằng validation rồi chạy một lượt test có kiểm soát. Báo cáo rõ
   hạn chế nếu cold-start vẫn kém TF-IDF; hoàn thiện báo cáo/slide sau benchmark.

## Khởi động lại FITLAB

Mở lại workspace `DA_AI` sau khi iDragonCloud mount hoạt động; watchdog có thể hồi
phục campaign nếu checkpoint và queue còn truy cập được. Kiểm tra bằng:

```bash
python3 status.py
```

Nếu trạng thái không tiến triển, xem `docs/recovery.md` trước khi khởi chạy lại.

Chi tiết snapshot: [Báo cáo tiến độ UFM](../reports/Tien_do_UFM_2026-10-02.md).