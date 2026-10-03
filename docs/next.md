# Bước tiếp theo — 03/10/2026

Code, dữ liệu CPU thứ hai, audit/calibration và demo UFM đã được triển khai.
Xem [tiến độ có bằng chứng](../reports/progress.md) và [protocol hàng đợi](followup.md).

1. **Chờ GPU FITLAB rảnh:** ablation → benchmark Toys/ba seed → CLIP/benchmark All Beauty.
   Các queue đã khởi động, không cần mở thêm trainer. Kiểm tra `python3 status.py`.
2. Sau khi có kết quả, tổng hợp mean/std và ablation. Ưu tiên giải thích thiếu hụt
   extreme cold; UFM cold macro hiện thấp hơn TF-IDF khoảng 0,027532.
3. Khóa cấu hình, checkpoint SHA256 và calibration bằng validation, tạo manifest
   trước test. Không chọn lại mô hình hoặc temperature theo kết quả test.
4. Chạy một đợt test cho các cấu hình đã khóa. Ghi rõ test baseline cổ điển đã xem
   trước đây; test UFM vẫn chưa chạy.
5. Đo VRAM/RAM, latency p50/p95, throughput và giới hạn 16 GiB. Hoàn thiện báo cáo
   cuối cùng và slide từ kết quả thật. Demo đã nghiệm thu cả xuất JSON.

Hai queue mới có heartbeat trong `status.py` nhưng chưa có autostart qua reboot.
Nếu container dừng hoặc queue báo lỗi, xem log/checkpoint và
[hướng dẫn phục hồi](recovery.md), rồi chạy lại đúng lệnh trong [followup.md](followup.md).
Không kết luận đã hoàn tất từ việc chỉ nhìn thấy queue đang sống.
