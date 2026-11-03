# Xin giảng viên góp ý giao thức đánh giá — 30/10/2026

Đây là đề xuất để trao đổi, **chưa phải quyết định của giảng viên**. Phạm vi ba nguồn tín hiệu đã được nhóm xác nhận; ba điểm cần chốt thêm là chỉ số nghiệm thu, cách đánh giá cuối và yêu cầu dataset thứ hai.

## Tin nhắn có thể gửi thầy/cô

> Thưa thầy/cô, nhóm đã đánh giá lại trên validation v3 gồm 40.000 ca, 100 ứng viên mỗi ca. Cold macro NDCG@10 của UFM full là **0,182056**, thấp hơn semantic-only CLIP (**0,190036**) và TF-IDF (**0,195682**); popularity đạt **0,042527**. Vì vậy nhóm chưa kết luận UFM cải thiện chất lượng cold-start.
>
> Nhóm dự định dùng **Cold macro NDCG@10** làm chỉ số chính, báo thêm **Recall@10 và MRR@10**, kết quả từng nhóm sản phẩm theo số tương tác và từng seed. Kế hoạch 15 lượt gồm năm biến thể × ba seed; mỗi biến thể kiểm tra riêng một câu hỏi về đóng góp của nhánh semantic, cách dung hợp và hai điều chỉnh theo số tương tác. Nhóm sẽ báo cả kết quả không cải thiện.
>
> Checkpoint cũ đã được chọn bằng validation v1; v3 dùng quần thể validation liên quan, còn test v2 đã được xem. Nhóm xin thầy/cô góp ý: **(1)** bộ chỉ số và đối chứng tối thiểu như trên có phù hợp không; **(2)** có cần một nguồn holdout cuối độc lập, chưa dùng để chọn mô hình, hay chấp nhận kết quả thăm dò kèm giới hạn; **(3)** UFM trên All Beauty có bắt buộc không, hay Toys & Games đủ làm thực nghiệm chính? Sau khi thống nhất, nhóm sẽ cố định giao thức rồi mới chạy và tổng hợp vòng thực nghiệm tiếp theo.

## Câu hỏi mà 15 lượt train cần trả lời

Manifest hiện tại gồm năm biến thể `full`, `semantic_only`, `fixed_fusion`, `count_aware_fusion`, `adaptive_dropout`, mỗi biến thể chạy seed **42, 7, 2026** trên Toys & Games. Bảng này giải thích mục đích; nó không thay config và chưa phải kết quả.

| Biến thể | Câu hỏi kiểm chứng | Đối chiếu chính |
| --- | --- | --- |
| `semantic_only` | Nhánh semantic dùng CLIP đạt mức nào khi tắt nhánh hành vi? | Mốc ablation nội bộ cho `full`; TF-IDF là baseline hệ thống riêng. |
| `full` | Thêm nhánh hành vi và dung hợp học được có cải thiện so với semantic-only, nhất là ở nhóm ít tương tác? | `semantic_only`, `fixed_fusion` và TF-IDF trên cùng bộ candidates. |
| `fixed_fusion` | Trọng số dung hợp theo độ tin cậy của `full` có hơn trọng số cố định 0,5/0,5 không? | `full`; các nhánh và phần alignment vẫn được giữ. |
| `count_aware_fusion` | Phạt độ tin cậy nhánh hành vi ở sản phẩm ít tương tác có giúp nhóm lạnh mà không làm giảm quá nhiều nhóm warm? | `full` với cùng ID dropout cố định. |
| `adaptive_dropout` | Thay ID dropout trong lúc train theo số tương tác có giúp nhóm lạnh không? | `full`; không cộng penalty của `count_aware_fusion`. |

Tất cả lượt chạy cần dùng cùng nguồn dữ liệu, feature, candidate protocol, ngân sách và quy tắc chọn checkpoint. Báo từng seed, **trung bình ± độ lệch chuẩn mẫu** của Cold macro NDCG@10; kèm Recall@10, MRR@10 và số ca theo zero-shot (0), extreme-cold (1–5), cold (6–20), warm (21+). TF-IDF và UFM có thiết kế/history khác nhau, nên chênh lệch giữa hai hệ thống không tự chứng minh tác dụng riêng của một thành phần.

Kết quả cũ trên **một seed/checkpoint** cho `full − semantic_only` là **−0,024101** ở nhóm 1–5 và **+0,014865** ở nhóm 21+. Đây là quan sát để đặt giả thuyết; chưa chứng minh hai cải tiến mới hiệu quả hoặc chỉ ra nguyên nhân suy giảm.

## Quy tắc cho đánh giá cuối

Chọn checkpoint, variant và mọi siêu tham số bằng validation theo quy tắc đã ghi trước. Nếu giảng viên yêu cầu kết luận trên tập độc lập, nhóm cần thống nhất **nguồn dữ liệu và cách tạo holdout chưa từng dùng để chọn mô hình**, khóa candidate sampling/metric trước khi xem kết quả, rồi đánh giá mô hình đã chọn trên holdout đó. **Không dùng điểm holdout để chọn lại mô hình hoặc seed.** Chia lại dữ liệu test v2 đã xem, hoặc đổi tên validation v3, không tự tạo ra một holdout độc lập.

Nếu không có nguồn đánh giá độc lập phù hợp, báo rõ kết quả là **thăm dò trên validation**, nêu việc test v2 đã được xem và giới hạn của negative sampling hồi cứu. All Beauty hiện có dữ liệu và baseline tham chiếu nhưng chưa có UFM xác minh; việc train trên tập này phụ thuộc câu trả lời của giảng viên và nằm ngoài 15 lượt Toys & Games.
