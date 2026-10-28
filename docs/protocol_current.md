# Phạm vi và giao thức làm việc của UFM Rec

<!-- current-evidence-20261028 -->
> Cập nhật 28/10/2026: validation v3 đã hoàn tất (40.000 ca, 100 ứng viên/ca, seed 42). Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only 0,190036; UFM full 0,182056; popularity 0,042527**. Tám checkpoint UFM dùng cùng candidate hash. Đây là đánh giá lại checkpoint chọn bằng validation v1, chưa phải holdout độc lập. Test v2 đã được xem; vòng v3 này không đánh giá test. Chi tiết và giới hạn: [bằng chứng mới](../reports/continuation_20261028/README.md). Các số liệu ở phần lịch sử phải đọc theo phiên bản giao thức ghi kèm. Word/PDF cũ chưa đồng bộ cập nhật này.
<!-- /current-evidence-20261028 -->


Bản làm việc ngày 27/10/2026 do nhóm chuẩn bị theo yêu cầu xử lý các mục 1–15. Chưa có xác nhận của giảng viên trong cuộc trao đổi này; không ghi là thầy/cô đã duyệt.

## Phạm vi thực hiện trước

Tập chính là Toys & Games, bài toán cold-start sản phẩm; metric chọn mô hình là Cold macro NDCG@10. Mô hình full và baseline phải có cùng dataset, mapping, candidates, tie-breaking và metric. Người dùng mới chưa có lịch sử dùng fallback; chưa tuyên bố đã có cá nhân hóa cho nhóm này. All Beauty là phần mở rộng cho tới khi xác nhận lại yêu cầu nghiệm thu.

## V2 dùng để đối chiếu artifact đã có

[Config cố định](../configs/evaluation_protocol_v2.json) giữ nguyên validation có hash `c71906e6…`, 40.000 ca, 10.000 ca mỗi regime và 99 negative cùng regime với target. Trọng số lấy mẫu dùng popularity đến cuối split; loại target và history được ghi nhận. Đây là giao thức hồi cứu, không phải train-only hoặc online/full-catalog.

Không sửa v2 tại chỗ để tránh làm mất khả năng so sánh với dự đoán UFM đang có. Positive đã biết ngoài history window được audit và công bố; không coi negative là nhãn không thích đã xác nhận. Baseline ưu tiên popularity và TF-IDF, chỉ chạy validation trong đợt sửa này.

## V3 sửa loại positive đã biết

Sample builder hiện hỗ trợ loại mọi positive có timestamp nhỏ hơn target, bao gồm ngoài history window và các sự kiện sớm hơn trong cùng split. Không dùng sự kiện cùng timestamp hoặc tương lai để loại negative. CLI mặc định tạo validation v3 ở output mới; chế độ legacy chỉ dùng để replay v2.

V3 vẫn dùng pool/weights hồi cứu cho tới khi có thiết kế và dữ liệu availability theo thời gian đủ rõ. Không đổi tên nó thành đánh giá online. V3 có bộ mẫu khác nên cần dự đoán UFM/baseline mới; không ghép metric v2 vào bảng v3.

## Test đã được xem

Test v2 đã xuất hiện trong artifact. [Sổ đánh giá](../reports/test_evaluation_ledger.json) nhập lại bằng chứng này, giữ thời điểm chạy chưa xác minh là null. Các lệnh final-test, re-evaluate và baseline đều dùng cùng ledger; đổi output/checkpoint/candidate vẫn không tự tạo ra một test độc lập khi dataset đã có lịch sử truy cập.

Các lệnh mặc định chỉ chạy validation. Muốn đánh giá lại test đã xem phải chỉ định mục đích thăm dò; báo cáo không mặc định `first_evaluation=true`. Chưa có đủ cơ sở để gọi điểm test đã có là nghiệm thu độc lập sau manifest khóa trước khi chạy.

## Điều còn cần giảng viên góp ý

Thầy/cô cần xác nhận mức nghiệm thu: một hay hai dataset; đối chứng/ablation và số seed tối thiểu; giữ sampled retrospective làm kết quả chính hay bổ sung protocol availability/holdout khác. Mã, config và dữ liệu được giữ thành phiên bản có thể đối chiếu để áp dụng góp ý sau này.
