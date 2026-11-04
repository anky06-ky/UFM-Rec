# Hướng dẫn trình bày và trả lời bảo vệ UFM Rec

Cập nhật 03/11/2026: nhóm chốt đồ án với checkpoint và bằng chứng hiện có; vòng 15 lượt đa seed là hướng phát triển. Bộ nộp: [báo cáo Word](../output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx), [PDF](../output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf), [PPTX](../output/presentation/UFM_Rec_Bao_Ve_20261103.pptx). Đây là kịch bản đề xuất; nhóm chưa diễn tập và chưa chốt người phụ trách từng phần.

## Mở đầu trong khoảng 45 giây

“Nhóm xây dựng hệ khuyến nghị sản phẩm từ lịch sử hành vi, ảnh và tên/mô tả. Hệ thống kết hợp nhánh hành vi với nhánh semantic dùng CLIP đóng băng, có cơ chế điều chỉnh trọng số theo độ tin cậy học được. Nhóm đã triển khai demo UFM thật trên 767.045 sản phẩm. Trong đánh giá validation v3 gồm 40.000 ca, full UFM đạt cold macro NDCG@10 là 0,182056, thấp hơn semantic-only 0,190036 và TF-IDF 0,195682. Nhóm trình bày cả kết quả này và giới hạn giao thức; hai hướng điều chỉnh theo tần suất sản phẩm chưa có kết quả huấn luyện mới.”

## Trình bày nội dung trong 8 phút

1. **Phút 0–1:** bài toán, ba nguồn tín hiệu và ý nghĩa các nhóm train count 0, 1–5, 6–20, 21+. Phân biệt sản phẩm mới với người dùng không có history.
2. **Phút 1–3:** CLIP đóng băng → adapter → trung bình history ở nhánh semantic; ID embedding → causal Transformer ở nhánh hành vi; uncertainty proxy và fusion. Nêu mask khi nhánh không khả dụng.
3. **Phút 3–4:** split thời gian 80/10/10, cùng bộ 100 candidates; checkpoint lịch sử chọn bằng v1, kết quả đang trình bày là v3 thăm dò.
4. **Phút 4–6:** bảng v3 và hai hình trong báo cáo; full thua cold macro, chênh lệch lớn ở 1–5 tương tác. Không quy nguyên nhân cho gate/ID chỉ từ chênh lệch điểm. Sau đó nêu kết quả ghép UFM + TF-IDF ở §5.12: 0,2155 so với 0,2038 của TF-IDF trên 20.000 ca muộn, α chọn trên 20.000 ca sớm. Bảng test v2 là lịch sử riêng.
5. **Phút 6–7:** calibration ECE/NLL/Brier; giải thích confidence có điều kiện trên candidate set và temperature không cải thiện thứ hạng.
6. **Phút 7–8:** giới hạn, phần đã chạy và phần cần kiểm chứng. Hai biến thể count-aware và adaptive dropout là hai can thiệp riêng.

Nguồn 14 slide: [presentation_slides.md](presentation_slides.md). Dùng số liệu từ [evidence.json](../reports/nontraining_20261029/evidence.json), không gõ lại từ bản Word/PDF/PPTX trong `output/history/`.

## Demo trong khoảng 3 phút

Chuẩn bị trước giờ trình bày:

- Khởi động bằng `D:\Nam3_ki1\UFM-Rec_Backups\Start_UFM_From_Backup.cmd`; xác nhận nhãn **UFM Rec**, catalog **767.045**, counts không mô phỏng. Launcher này nằm ngoài Git và cần bản backup trên máy. Không mặc định endpoint FITLAB còn sống.
- Đợi khởi động xong, kiểm tra `/api/status`; đường dẫn localhost do launcher in ra. Lượt nghiệm thu ngày 29/10 dùng cổng 8773 riêng.
- Chuẩn bị history `B004S8F7QM`, K=5; thử trước các regime. Chế độ Nhanh (mặc định) trả kết quả khoảng 0,4–0,8 giây; Toàn catalog khoảng 7,5 giây, tới 13 giây khi RAM chật, nên báo trước khi chọn. Tắt máy ảo và ứng dụng nặng trước giờ trình bày.
- Dùng [ảnh zero-shot](../reports/demo_20261103/demo_ui_zero_shot_20261103.png), [ảnh UFM + TF-IDF](../reports/demo_20261103/demo_ui_hybrid_20261103.png), [ảnh lỗi mạng](../reports/demo_20261103/demo_ui_offline_20261103.png) và [JSON đã xuất](../reports/demo_20261103/demo_ui_downloads/ufm-rec-demo.json) làm dự phòng. Nút xuất JSON đã đạt trong 5/5 ca UI tự động trên Windows; thử lại trên máy bảo vệ. Chưa quay video.

Trong buổi trình bày:

1. Cho thấy backend/catalog thật. Chạy một lần không history, giải thích đây là fallback, chưa cá nhân hóa người dùng mới.
2. Thêm sản phẩm `B004S8F7QM`, chọn `Zero-shot`, chạy Top 5. Trong lúc chờ, giải thích zero-shot là sản phẩm có train count bằng 0.
3. Chỉ ra `score`, thanh trọng số Hành vi/Ngữ nghĩa và uncertainty: đây là tín hiệu mô hình, không phải xác suất mua. Ảnh hiển thị tải lỗi không làm thay đổi CLIP embedding đã lưu.
4. Đổi Mô hình sang UFM + TF-IDF, Phạm vi Toàn catalog, chạy lại: chỉ ra điểm kết hợp và dòng điểm UFM · TF-IDF. α = 0,55 lấy từ mục 5.12, chưa hiệu chỉnh cho xếp hạng trên catalog.
5. Xuất JSON sau khi request thành công; kiểm tra model hash, history, regime, mô hình, phạm vi, kết quả và trường giải thích. Ca Chrome headless đã đạt và lưu [receipt](../reports/demo_20261103/demo_ui_end_to_end_20261103.json); khi trình bày vẫn bấm lại trên máy bảo vệ.

Các ca API đã kiểm tra: không history, có history và bốn regime đều đạt 6/6; sau khi tăng tốc, sáu ca được phát lại ngày 04/11 và cho cùng sản phẩm, cùng thứ tự. Năm ca UI đạt 5/5 gồm zero-shot, export JSON, placeholder, chế độ UFM + TF-IDF và lỗi mạng. History toàn ASIN không có trong catalog trả HTTP 400 với thông báo `Unknown ASIN`; API không âm thầm coi nó là history rỗng. Nguồn: [receipt API](../reports/nontraining_20261029/demo_api_verification.json), [phát lại và độ trễ](../reports/demo_20261103/demo_modes.json), [receipt UI](../reports/demo_20261103/demo_ui_end_to_end_20261103.json), [ca unknown](../reports/nontraining_20261029/demo_unknown_history.json).

## Câu hỏi dễ gặp

**1. Vì sao gọi đa phương thức?** Sản phẩm có embedding từ ảnh và text; mô hình kết hợp chúng với tín hiệu hành vi. Transformer nằm ở nhánh hành vi. Nhánh semantic dùng adapter và tổng hợp history bằng trung bình có mask.

**2. TF-IDF và semantic-only khác nhau thế nào?** TF-IDF là baseline nội dung theo từ vựng. Semantic-only trong bảng ablation dùng CLIP và đường semantic của UFM. Hai baseline khác representation và thiết kế học; không dùng tên thay thế cho nhau.

**3. Vì sao full UFM thua baseline?** Kết quả hiện tại chứng minh có suy giảm trên cold macro của giao thức này. Nó chưa chứng minh nguyên nhân. Chênh lệch full − semantic-only tại nhóm 1–5 tương tác là −0,024101; nhóm có history cần phân tích gate/uncertainty và đối chứng nhiều seed để kiểm chứng giả thuyết.

**4. Vậy đóng góp hiện tại là gì?** Một hệ thống tái lập được từ dữ liệu đa phương thức đến suy luận, thiết kế fusion/proxy và thực nghiệm phân tích cả điểm mạnh lẫn hạn chế. UFM đứng riêng chưa vượt đối chứng; khi ghép điểm với TF-IDF thì vượt TF-IDF trên phần validation không dùng để chọn tham số (§5.12). Không tuyên bố tính mới toàn diện của kiến trúc. CLIP, Transformer và BPR là thành phần kế thừa.

**5. Uncertainty có phải Bayes không?** Đây là proxy học được với đầu ra dương, dùng để điều chỉnh trọng số. Chưa có bằng chứng cho posterior Bayes hoặc uncertainty được hiệu chuẩn tuyệt đối.

**6. ECE giảm có làm gợi ý tốt hơn không?** Temperature scaling thay đổi confidence; không đổi thứ hạng trong cùng bộ candidates. Trên audit validation v3, ECE 0,082061 → 0,037478; NLL 4,497667 → 4,258086; Brier 0,969342 → 0,960605. Đây không phải xác suất người dùng mua hàng.

**7. V3 có phải test sạch không?** Không. Checkpoint được chọn bằng validation v1, và v3 vẫn dùng quần thể validation liên quan. Test v2 đã được xem. Nhóm cần thống nhất phạm vi thăm dò hoặc nguồn nghiệm thu độc lập trước khi mở thêm test.

**8. 100 candidates có đại diện toàn sàn không?** Chưa. Mỗi ca có một positive và 99 negatives; sampling theo regime target và thống kê cuối split có tính hồi cứu. Điểm này không tương đương full-catalog hoặc đánh giá online.

**9. Nhiều seed và hai cải tiến đã chạy chưa?** Chưa. Kết quả lịch sử hiện có một seed 42. Kế hoạch khóa 15 run gồm năm biến thể × ba seed; count-aware và adaptive dropout là hai biến thể riêng. Nhóm chốt đồ án trước vòng này và trình bày nó là hướng phát triển, kèm notebook Colab có resume.

**10. All Beauty đã chứng minh khả năng tổng quát chưa?** Chưa. Có dữ liệu và baseline tham chiếu, chưa có UFM được xác minh trên All Beauty. Dataset thứ hai chỉ cần triển khai nếu nằm trong phạm vi nghiệm thu; train riêng hai dataset cũng không tự là transfer learning.

**11. Ghép UFM với TF-IDF có phải chọn trên chính dữ liệu báo cáo không?** Không. α được chọn trên 20.000 ca sớm theo thời gian; kết quả lấy trên 20.000 ca muộn chưa dùng để chọn. Trước khi trộn, điểm tính lại của hai mô hình khớp hoàn toàn số đã công bố. Mức tăng cold macro so với TF-IDF là 0,011789, khoảng tin cậy 95% [0,009796; 0,013744]. Đây vẫn là phân tích sau: v3 không phải test độc lập và chỉ có một seed. Demo có chế độ UFM + TF-IDF dùng lại α này khi xếp hạng trên catalog, nhưng α chưa được hiệu chỉnh lại cho bối cảnh đó.

## Việc nhóm còn cần làm trước khi bảo vệ

- Kiểm tra Word/PDF và slide 1 ghi đúng ba thành viên Nguyễn Thành Phong, Trần An Kỳ và Nguyễn Hoàng Hải; thông tin dựng báo cáo nằm trong `TEAM` ở `scripts/build_school_report_docx.py`.
- Phân công người trình bày từng phần.
- Diễn tập có bấm giờ và phân công; thử lại xuất JSON, mạng và ảnh trên máy bảo vệ. Có ảnh dự phòng từ UFM thật; quay video nếu nhóm cần trình diễn khi không có mạng.
- Nếu sau này chạy thêm train, thay bảng/hình và câu trả lời bằng bằng chứng của model mới; không giữ kết luận cũ nếu số liệu đã đổi.
