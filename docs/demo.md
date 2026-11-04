# Chạy và trình bày demo UFM Rec

Demo có hai chế độ. Chương 6 của [báo cáo](final_report.md) mô tả kiến trúc và kết quả nghiệm thu; tài liệu này chỉ hướng dẫn chạy và trình bày.

| Chế độ | Khi nào dùng | Dữ liệu |
| --- | --- | --- |
| **UFM Rec** (mô hình thật) | Máy có bản sao dữ liệu đầy đủ | Checkpoint epoch 9, đặc trưng CLIP, catalog 767.045 sản phẩm |
| **Content** (TF-IDF) | Máy không có dữ liệu lớn | Tự tạo catalog mẫu 30 sản phẩm thật; số tương tác là mô phỏng |

## Khởi động

- **UFM trên máy có bản sao:** mở `D:\Nam3_ki1\UFM-Rec_Backups\Start_UFM_From_Backup.cmd`. Launcher chạy `Start_Demo.cmd --backend ufm --port 8771` với dữ liệu trong `restored_complete_20261028`. Đợi cửa sổ server in dòng `DEMO READY` rồi mới thao tác.
- **Máy khác:** mở `Start_Demo.cmd` trong thư mục dự án. Mặc định (`--backend auto`) launcher dùng UFM khi thấy checkpoint hoàn tất (`runs/ufm_full_v1/best.pt` và `completed.json`), ngược lại dùng content với catalog mẫu. Chọn rõ bằng `--backend ufm` hoặc `--backend content`; đổi cổng bằng `--port`.
- Trình duyệt tự mở `http://127.0.0.1:<cổng>` khi server sẵn sàng; mặc định cổng 8765. Giữ cửa sổ server mở khi trình bày, nhấn Ctrl+C để dừng. Server chỉ nhận kết nối từ chính máy đó.

Dữ liệu lớn (checkpoint, CLIP, catalog) không nằm trên GitHub; cách khôi phục và mã băm ở [bàn giao backup](../reports/continuation_20261028/backup_handoff.md).

Với UFM Rec, giao diện có thêm hai ô (chế độ content ẩn hai ô này):

| Ô | Lựa chọn | Ý nghĩa |
| --- | --- | --- |
| **Mô hình** | UFM; UFM + TF-IDF | UFM + TF-IDF trộn 55% điểm UFM với 45% điểm TF-IDF sau khi chuẩn hóa (mục 5.12 của báo cáo) |
| **Phạm vi chấm** | Nhanh (mặc định); Toàn catalog | Nhanh: UFM chấm lại tối đa 5.500 ứng viên gần nhất. Toàn catalog: chấm mọi sản phẩm, đúng thứ hạng của mô hình nhưng chậm hơn |

## Kiểm tra trước khi trình bày

1. Nhãn backend trên giao diện là **UFM Rec** và số sản phẩm là **767.045**. Nếu thấy chế độ content, đang chạy catalog mẫu chứ không phải mô hình thật.
2. Tìm "Cards Against Humanity", thêm `B004S8F7QM` vào lịch sử và chạy thử một lượt.
3. Thời gian trên CPU: chế độ Nhanh khoảng 0,4–0,8 giây; Toàn catalog khoảng 7,5 giây khi máy còn nhiều RAM trống, tới 13 giây khi RAM chật; lọc theo một nhóm sản phẩm nhanh hơn. Tắt máy ảo và ứng dụng nặng trước khi trình bày; nút "Tìm gợi ý" đếm số giây trong lúc chờ.
4. Mở sẵn ảnh dự phòng [zero-shot](../reports/demo_20261103/demo_ui_zero_shot_20261103.png), [UFM + TF-IDF](../reports/demo_20261103/demo_ui_hybrid_20261103.png) và [mất mạng](../reports/demo_20261103/demo_ui_offline_20261103.png) phòng khi không chạy được.

## Kịch bản 3 phút

1. Giới thiệu catalog Toys & Games 767.045 sản phẩm và chỉ nhãn backend đang chạy.
2. Tìm "Cards Against Humanity", thêm `B004S8F7QM` vào lịch sử. Lịch sử xếp từ cũ đến mới, tối đa 20 sản phẩm, không trùng.
3. Giữ Mô hình **UFM**, Phạm vi **Nhanh**; chọn Top K = 5, nhóm **Zero-shot**, bấm "Tìm gợi ý cho tôi". Chỉ ra mọi kết quả có 0 tương tác train và dòng **Hành vi 0% · Ngữ nghĩa 100%**: mô hình vẫn gợi ý được sản phẩm chưa có tương tác nhờ văn bản và ảnh.
4. Đổi sang nhóm **Warm** và chạy lại: phần Hành vi khoảng 46–48% vì các sản phẩm này đã có tương tác trong train.
5. Đổi Mô hình sang **UFM + TF-IDF**, Phạm vi **Toàn catalog**, nhóm **Tất cả sản phẩm**; báo trước thời gian chờ. Chỉ ra "Điểm kết hợp" và dòng điểm UFM · TF-IDF; các gợi ý đầu là sản phẩm khớp từ khóa với lịch sử.
6. Bấm "Xuất JSON" để lưu đầu vào, mô hình và kết quả. Khi đổi lịch sử, bộ lọc hoặc chế độ, kết quả cũ bị làm mờ và nút xuất bị khóa đến khi chạy lại.
7. Bấm "Xóa tất cả" rồi chạy với mọi nhóm: không có lịch sử thì demo xếp theo độ phổ biến trong train, chưa cá nhân hóa cho người dùng mới.

## Cách giải thích đúng

- Zero-shot nghĩa là không có tương tác trong tập train, không nhất thiết là sản phẩm vừa ra mắt.
- Điểm xếp hạng, trọng số và độ bất định là tín hiệu của mô hình, không phải xác suất mua hàng; không so điểm giữa hai mô hình.
- Ảnh trên màn hình chỉ để xem: mô hình dùng đặc trưng CLIP đã trích xuất sẵn, nên ảnh tải lỗi không làm đổi điểm.
- Chế độ Nhanh là xấp xỉ: với UFM, top 10 trùng trung bình 84% với Toàn catalog (100% ở nhóm Cold và Warm, 71% ở Zero-shot); với UFM + TF-IDF trùng 99%. Cần đúng thứ hạng của mô hình thì chọn Toàn catalog.
- Trọng số trộn của UFM + TF-IDF được chọn trên đánh giá 100 ứng viên, chưa hiệu chỉnh cho xếp hạng trên catalog.
- Demo xếp hạng trên catalog hoặc nhóm đã lọc; số liệu trong báo cáo đo trên 100 ứng viên mỗi ca, nên không phải chất lượng của demo.

## Xử lý sự cố

- **Cổng bận:** chạy lại với `--port` khác, ví dụ 8766; không dừng tiến trình không rõ nguồn.
- **Thiếu thư viện:** launcher báo `Missing dependencies`; chạy `py -m pip install -r requirements.txt` rồi mở lại.
- **Lỗi 429:** server chỉ xử lý một lượt suy luận mỗi lúc; đợi lượt trước xong rồi bấm lại.
- **Lỗi 400 "Unknown ASIN":** lịch sử chứa mã không có trong catalog; chọn sản phẩm từ ô tìm kiếm.
- **Thiếu dữ liệu UFM:** khôi phục từ bản sao lưu theo hướng dẫn bàn giao, hoặc dùng chế độ content; không tạo dữ liệu giả.

## Bằng chứng nghiệm thu

[6/6 ca API ngày 29/10](../reports/nontraining_20261029/demo_api_verification.json), [phát lại 6 ca, độ trễ và độ trùng của chế độ Nhanh ngày 04/11](../reports/demo_20261103/demo_modes.json), [5/5 ca giao diện](../reports/demo_20261103/demo_ui_end_to_end_20261103.json), [lịch sử toàn mã lạ trả 400](../reports/nontraining_20261029/demo_unknown_history.json), [đo độ trễ 28/10](../reports/continuation_20261028/windows_ufm_benchmark.json), [ca thiếu phương thức](../reports/continuation_20261028/windows_ufm_missing_modalities.json).
