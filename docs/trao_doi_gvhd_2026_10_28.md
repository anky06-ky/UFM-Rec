# Nội dung trao đổi với giảng viên — 28/10/2026

“Phạm vi hệ gợi ý sản phẩm dùng hành vi, hình ảnh và tên/mô tả sản phẩm đã được nhóm trao đổi và chốt với thầy. Nhóm dùng CLIP để lấy đặc trưng nội dung và mô hình hóa lịch sử hành vi, rồi thử học trọng số dung hợp giữa hai nhánh. Cold-start sản phẩm là trường hợp nhóm phân tích sâu; khi không có lịch sử người dùng, hệ thống dùng fallback và chưa có bằng chứng cá nhân hóa người dùng mới.

Nhóm đã khôi phục checkpoint thật, kiểm toán bảy ablation và đánh giá lại trên cùng validation v3 gồm 40.000 ca, mỗi ca 100 ứng viên. Cold macro NDCG@10 của full UFM là 0,182056, semantic-only là 0,190036, TF-IDF là 0,195682. Vì vậy nhóm chưa kết luận cơ chế dung hợp giúp cải thiện cold-start. Các checkpoint hiện có một seed; validation v3 cùng quần thể đã dùng chọn checkpoint trước đây nên chưa phải holdout độc lập.

Demo đã chạy UFM thật trên 767.045 sản phẩm, trả được gợi ý zero-shot và trọng số nhánh. Kết quả demo chứng minh luồng ứng dụng hoạt động; chất lượng recommendation được đánh giá riêng bằng benchmark. Phân tích v3 theo số tương tác cho thấy full thấp hơn semantic-only nhiều nhất ở nhóm 1–5 tương tác (−0,024101 NDCG@10), nhưng cao hơn ở nhóm warm (+0,014865). Nhóm đã tách hai cải tiến count-aware và adaptive dropout để kiểm tra riêng, khóa 3 seed và cùng ngân sách; chưa có kết quả train mới được xác minh.”

Ba câu xin góp ý:

1. Với phạm vi ba nguồn tín hiệu đã chốt, thầy có yêu cầu thêm kết quả UFM trên All Beauty hay Toys & Games đủ cho phần thực nghiệm chính?
2. Nhóm đã khóa full, semantic-only và fixed-fusion qua ba seed trước hai cải tiến. Thầy muốn dùng thêm chỉ số nghiệm thu nào ngoài Cold macro NDCG@10?
3. Với test v2 đã được xem và negative sampling có tính hồi cứu, cách đánh giá tiếp theo nào phù hợp để kết luận đóng góp trong phạm vi đồ án?

Bằng chứng mang theo: [kết quả và calibration v3](../reports/continuation_20261028/README.md), [request UFM thật](../reports/continuation_20261028/demo_ufm_verification.json), [checklist còn lại](checklist_hoan_thien_2026_10_27.md). File Word/PDF cũ chưa đồng bộ nội dung mới nên chưa dùng làm bản nộp cuối.
