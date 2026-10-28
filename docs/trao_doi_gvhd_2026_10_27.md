# Trao đổi UFM Rec với giảng viên — sau sửa mục 1–15

Đối chiếu hoàn tất ngày **28/10/2026**: mã hiện tại trên FITLAB-01 đạt **87 test pass, không skip**; Windows local **84 pass, 3 skip**. Xem báo cáo sửa mục 1–15 và log để phân biệt bản mã hiện tại với bản mã đã tạo checkpoint.

Cập nhật ngày 27/10/2026. Nhóm xin góp ý về giao thức và phạm vi nghiệm thu; chưa khẳng định UFM đã cải thiện cold-start.

## Ý tưởng và bằng chứng mới

UFM gợi ý sản phẩm ít hoặc chưa có tương tác bằng cách kết hợp văn bản/ảnh CLIP đóng băng với lịch sử hành vi qua Transformer. Cổng độ tin cậy học được điều chỉnh trọng số hai nhánh; uncertainty là proxy học được. Người dùng chưa có lịch sử dùng fallback.

Đã kiểm tra trực tiếp FITLAB-01: checkpoint full epoch 9, mã huấn luyện và cache CLIP khớp hash/mapping/mask. Cache có 767.045 item, 767.044 text, 758.692 ảnh. Forward CPU hai ca validation hữu hạn; metric từ predictions lịch sử khớp marker.

Trên đúng 40.000 ca validation v2, Cold macro NDCG@10: **UFM 0,182051; TF-IDF 0,195672; popularity 0,042527**. Overall: **0,164132; 0,181498; 0,040904**. UFM chưa vượt TF-IDF trên bộ này. TF-IDF dùng 10 item history, UFM dùng 20; cần ablation có kiểm soát để giải thích nguyên nhân. Không ghép baseline lịch sử vào v2.

Audit mới xét mọi positive trước target, tìm được 7 negative trùng positive trong 6 ca validation v2. Nhóm giữ nguyên v2 và tạo validation v3 loại known positives; audit v3 đạt nhưng chưa có metric mô hình v3. V3 vẫn lấy negative cùng regime và dùng popularity cuối split, nên vẫn là đánh giá hồi cứu.

Local sau sửa có 84 test pass, 3 skip dành cho Linux. Bản mã hiện tại được kiểm thử riêng trên FITLAB-01; mã tạo checkpoint được giữ để tái lập. Chưa chạy GitHub Actions. Demo TF-IDF có 30 ảnh local, số tương tác mẫu là mô phỏng và được gắn nhãn.

Nguồn: [báo cáo sửa 1–15 và log](../reports/fix_1_15_20261027/README.md), [protocol](protocol_current.md), [checklist](checklist_hoan_thien_2026_10_27.md).

## Test và phần còn thiếu

Test v2 đã được đánh giá trong artifact cũ: Overall/Cold macro NDCG@10 **0,166193 / 0,183638**. Đã nhập lịch sử vào ledger; không gọi đây là test chưa mở hoặc manifest khóa trước test. Đợt sửa này không chạy suy luận test mới.

Cần giảng viên chốt phạm vi; kiểm toán các ablation đã lưu trên FITLAB-01; đánh giá biến thể mới, multi-seed và dataset thứ hai nếu thuộc yêu cầu. Chưa có bằng chứng count-aware/adaptive cải thiện chất lượng. Word/PDF cũ cần xuất lại sau khi thống nhất nội dung.

## Lời trình bày gợi ý

“Dạ thầy, nhóm đã xác minh checkpoint cùng đặc trưng đa phương thức trên FITLAB-01 và sửa các lỗi kiểm thử. Nhóm đã chạy popularity và TF-IDF trên đúng bộ validation của UFM. Cold macro của UFM hiện là 0,182051, thấp hơn TF-IDF 0,195672 nên nhóm chưa kết luận mô hình cải thiện.

Nhóm cũng phát hiện positive đã biết lọt vào negative của bộ mẫu cũ. Nhóm giữ bằng chứng cũ, tạo bộ validation mới loại các trường hợp này và kiểm tra lịch sử theo thời gian. Bộ mới cần đánh giá mô hình lại. Test v2 đã được xem và có ghi nhận. Nhờ thầy góp ý protocol, các đối chứng/ablation tối thiểu và phạm vi một hay hai dataset để nhóm kết luận có cơ sở.”

## Ba câu hỏi cần chốt

1. Ưu tiên hoàn thiện cold-start sản phẩm trên Toys hay bắt buộc thêm UFM All Beauty?
2. Full, semantic-only, fixed-fusion, count-aware/adaptive và phân tích gate cần tổ hợp ablation nào, tối thiểu bao nhiêu seed?
3. Giữ sampled retrospective làm kết quả chính có disclosure hay bổ sung đánh giá availability/holdout khác, khi test v2 đã được xem?
