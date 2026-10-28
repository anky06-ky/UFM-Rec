# Hướng làm tiếp theo của UFM-Rec

Cập nhật 28/10/2026. **Phạm vi giảng viên đã chốt theo thông tin nhóm cung cấp:** hệ khuyến nghị sản phẩm dùng lịch sử hành vi, hình ảnh và tên/mô tả sản phẩm. Toys & Games là tập thực nghiệm chính; cold-start sản phẩm là phân tích trọng tâm trong phạm vi hệ gợi ý này. Trường hợp người dùng không có lịch sử hiện là fallback; chưa có kết quả chứng minh cá nhân hóa người dùng mới. Cold macro NDCG@10 và ba seed dưới đây là **giao thức nhóm khóa để kiểm chứng**, không ghi là giảng viên đã duyệt từng chi tiết.

## Căn cứ để chọn hướng

Validation v3 có 40.000 ca, 100 ứng viên/ca, một seed 42. Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only CLIP 0,190036; UFM full 0,182056**. Full chưa vượt hai đối chứng này. Bảy ablation đã có kết quả, nhưng chưa có độ biến thiên giữa các seed. Nguồn: [kết quả và hash](../reports/continuation_20261028/README.md).

Vì vậy câu hỏi nghiên cứu tiếp theo là: **dung hợp thêm hành vi giúp hay làm giảm chất lượng ở từng mức train count, và điều chỉnh theo tần suất có giúp không?** Kết quả semantic-only tốt hơn full là bằng chứng để khảo sát; chưa tự chứng minh ID embedding hoặc gate là nguyên nhân.

## 1. Chốt phạm vi và kế hoạch thí nghiệm

- Phạm vi ba nguồn tín hiệu đã được nhóm xác nhận với giảng viên. Nếu cần phản hồi thêm, tập trung vào cách nghiệm thu chỉ số và bộ test, hoặc yêu cầu tập dữ liệu thứ hai; dùng [lời trình bày cập nhật](trao_doi_gvhd_2026_10_28.md).
- Giữ riêng v1/v2/v3; mọi mô hình trong một bảng dùng cùng candidate hash. V3 vẫn là negative sampling hồi cứu theo regime của target, không phải đánh giá online/full-catalog.
- Test v2 đã được xem. V3 dùng lại quần thể validation đã tham gia chọn checkpoint, chưa tạo holdout độc lập. Cần thống nhất thiết kế nghiệm thu tiếp theo trước khi mở thêm test; không chọn mô hình theo test đã biết.
- Đã khóa source, dữ liệu, features, candidates, selection rule, ngân sách và thứ tự 15 run trong [manifest thực nghiệm](../configs/multimodal_followup_20261028.json). [Runner](../scripts/run_locked_ufm_followup.py) đã qua dry-run trên bản sao FITLAB, kiểm tra 9 đầu vào SHA; chưa có train mới.

**Đầu ra cần có:** cấu hình/manifest của vòng chạy mới và phạm vi nghiệm thu rõ ràng.

## 2. Làm đối chứng vững trước khi thêm cải tiến

- Ưu tiên **full UFM, semantic-only và fixed-fusion**. Dùng cùng CLIP, history, loss/ngân sách và quy tắc chọn checkpoint.
- Dùng các seed đã khai báo **42, 7, 2026**; báo từng seed và mean ± sample standard deviation. Không chọn seed đẹp hoặc dùng bootstrap user thay cho nhiều lần huấn luyện.
- Nếu train bằng mã hiện tại, train lại các đối chứng trên cùng snapshot mã. Checkpoint lịch sử vẫn dùng đúng archived source khi nạp; không bỏ kiểm tra hash để gộp hai phiên bản mã.
- Báo thêm zero-shot/extreme-cold/cold/warm, có/không history và repeat/new purchase. Giữ TF-IDF làm mốc so sánh hệ thống; nó dùng history/objective khác UFM nên không quy chênh lệch cho riêng một thành phần.

**Đầu ra cần có:** bảng so sánh tái lập được, từng seed, config và predictions. Chỉ mở job train khi có tài nguyên phù hợp; không dừng job khác để lấy GPU.

Trạng thái 28/10: máy local chỉ có PyTorch CPU, SSH FITLAB-01 không kết nối, tab code-server không khả dụng. Chưa chạy 15 run mới. Runner mặc định chỉ kiểm tra; `--execute` cần GPU có quyền truy cập và source đúng hash.

Lệnh checkout, chạy từng run và khôi phục gói sao lưu có trong [runbook FITLAB](gpu_followup_runbook.md). Gói chỉ chứa artifact của một run để tránh sao lưu lặp và phải được xác minh SHA trên máy trước khi dọn `latest.pt` phía FITLAB.

## 3. Kiểm chứng cải tiến theo train count

1. Phân tích chất lượng và phân phối trọng số CF/semantic theo train count, tách các ca không có tín hiệu.
2. Đánh giá `count_aware_fusion` trước: cơ chế đã có mã nhưng chưa có kết quả train được xác minh. So với full và fixed-fusion trên cùng code/seed/budget.
3. Đánh giá `adaptive_dropout` thành thí nghiệm riêng sau đó; không gộp hai thay đổi rồi quy kết tác dụng cho một cơ chế.

Không lấy trọng số CF dưới một ngưỡng tùy ý làm tiêu chí thành công. Kết luận dựa trên chất lượng cold, ảnh hưởng ở warm và độ ổn định qua seed. Nếu không cải thiện, báo giới hạn một cách trung thực; đây vẫn là kết quả thực nghiệm có ích.

Đã phân tích [dự đoán v3 theo train count](../reports/continuation_20261028/count_effect_v3.json): full − semantic-only lần lượt **−0,002424** (count 0), **−0,024101** (1–5), **+0,002586** (6–20), **+0,014865** (21+). Đây là chênh lệch trên checkpoint cũ một seed, không phải kết quả của cải tiến mới. Mã `count_aware_fusion` đã được tách khỏi `adaptive_dropout`: biến thể thứ nhất giữ fixed ID dropout như full; biến thể thứ hai chỉ đổi dropout theo count. Bằng chứng tác dụng sau train còn chờ 15 run.

## 4. Nghiệm thu ứng dụng trên máy

- UFM thật trên FITLAB đã qua 6 ca API và UI zero-shot. UFM trên Windows cũng đã khởi động từ bản sao SHA và qua 6/6 ca API với 767.045 item, checkpoint SHA `5630013d0cbb6db633c4f4cf5bcd840fb703dbfc64a13c0352278624e03e11c8`: [receipt Windows](../reports/continuation_20261028/windows_ufm_verification.json). Thêm [ca thiếu modality](../reports/continuation_20261028/windows_ufm_missing_modalities.json): history chứa item chỉ có text hoặc chỉ có ảnh vẫn trả gợi ý, tìm kiếm và loại history đúng; không có item thiếu cả hai trong cache thực. JPEG local thiếu trả 404 có kiểm soát. [Kiểm tra giao diện Chrome local](../reports/continuation_20261028/windows_ufm_ui_visual_check.json) đã tìm LEGO, chọn history, trả Top 5 zero-shot và tải được 5/5 ảnh kết quả; file export local chưa xác minh tải xuống. Export UFM đã có bằng chứng trên FITLAB.
- File export UFM thật đã được tải về và chuyển vào `reports/continuation_20261028/ui_export_ufm_zero_shot.json`; đây là bằng chứng luồng export, không phải metric recommendation.
- Đã đo [benchmark HTTP Windows](../reports/continuation_20261028/windows_ufm_benchmark.json) trên UFM thật: 2 warm-up, 10 request có một item lịch sử và K=5, p50 **21,25 giây**, p95 **26,35 giây**; RSS server cao nhất quan sát **2.551,5 MiB**. Với hai request đồng thời, một thành công và một trả 429 vì server chỉ chạy một suy luận tại một thời điểm. Đây là số của một máy CPU/một cấu hình request, chưa phải throughput đồng thời hoặc peak training RAM.
**Đầu ra cần có:** demo UFM chạy lại từ bản sao local, toàn bộ ca API/UI và benchmark có bằng chứng.

## 5. Báo cáo và bàn giao

- Đồng bộ báo cáo, slide và generator Word; xuất Word/PDF rồi kiểm tra từng trang. Bản Word/PDF cũ chưa là bản nộp cuối.
- Rà diff, đồng bộ repo và chạy GitHub Actions. Không đưa dataset/checkpoint/ZIP lớn vào Git; giữ manifest và hướng dẫn khôi phục.
- Ghi riêng phần đã chạy và phần chưa có kết quả train, nhất là 15 lượt đa seed và cải tiến. Không gọi trạng thái chuẩn bị là hoàn tất thực nghiệm.

**Đầu ra cần có:** hướng dẫn cài/chạy, CI đạt và bộ báo cáo cùng phiên bản.

Nhánh [`followup/multimodal-20261028`](https://github.com/anky06-ky/UFM-Rec/tree/followup/multimodal-20261028) đã được đẩy lên GitHub; [CI Windows và Ubuntu đều đạt](https://github.com/anky06-ky/UFM-Rec/actions/runs/40693524511). [PDF tiến độ 2 trang](../output/pdf/UFM_Rec_Tien_Do_2026_10_28.pdf) đã render và kiểm tra trang; đây là tài liệu trao đổi, không thay thế bản Word/PDF báo cáo toàn văn. Script Word hiện thiếu `Mau_Bao_Cao_Nhom_4.docx` trong workspace và chưa có renderer LibreOffice đi kèm để nghiệm thu từng trang, nên chưa xuất lại bản toàn văn.

## Sau khi hoàn tất phần chính

UFM trên All Beauty chỉ ưu tiên nếu giảng viên giữ yêu cầu dataset thứ hai. Retrieval/ANN, fine-tune encoder và các tính năng mở rộng nên để sau khi có kết quả đối chứng ổn định.

Xem [checklist chi tiết](checklist_hoan_thien_2026_10_27.md), [bản sao dữ liệu đã kiểm tra](../reports/continuation_20261028/backup_handoff.md) và [hướng dẫn demo](demo.md).
