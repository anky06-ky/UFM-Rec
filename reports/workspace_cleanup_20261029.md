# Dọn workspace ngày 29/10/2026

Phạm vi: toàn bộ `D:\Nam3_ki1\UFM-Rec`, gồm file theo dõi bởi Git và file local. Người dùng yêu cầu dọn file tạm/trùng và giữ phần train lại. Không chạy train, commit hoặc push.

## Đã sắp xếp

- Rút gọn README gốc để chỉ ngay trạng thái, PDF, báo cáo nguồn, slide và checklist hiện hành. Các snapshot FITLAB và số liệu v1/v2 còn ở hồ sơ lịch sử được liên kết riêng.
- Cập nhật mục lục `reports/README.md`; thêm `docs/README.md` và `output/README.md` để phân biệt tài liệu đang sửa với bằng chứng cũ.
- Chuyển **4 PDF cũ** từ `output/pdf/` sang `output/history/pdf/`: báo cáo chi tiết 28/09, tiến trình 30/09, tiến độ 28/10 và proposal. SHA-256 trước/sau khớp; nội dung không thay đổi. `output/pdf/` còn PDF hiện trạng ngày 29/10.
- Sửa các liên kết liên quan và đường dẫn xuất của hai script dựng PDF lịch sử, để chạy lại không tạo bản cũ cạnh PDF hiện tại.
- Chuyển `tmp/`, `.pytest_cache/` và các thư mục `__pycache__/` ra `D:\Nam3_ki1\UFM-Rec_Backups\cleanup_20261029_nontraining\temporary_files`. Đã kiểm tra không có file tracked/reparse point và đối chiếu số file/dung lượng trước–sau. Tổng **3.945 file, 243.667.967 byte (khoảng 232,38 MiB)** đã rời workspace; **0 file bị xóa**, chưa giải phóng dung lượng ổ đĩa. [Receipt](cleanup_20261029.json) ghi từng thư mục.
- Bộ kiểm duyệt tự động chặn lệnh xóa cache với lý do duy nhất “blocked by policy”. Vì vậy dùng phương án chuyển có thể khôi phục; chưa thực hiện xóa vĩnh viễn.
- Kiểm tra liên kết Markdown trong repo phát hiện và sửa 10 liên kết lịch sử hỏng. Bản ghi khôi phục 30/09 không còn thì ghi rõ sự thiếu đó và dẫn tới hướng dẫn vận hành hiện có, không tạo lại bằng chứng lịch sử.

## Những bản giống nhau cần giữ

Rà SHA phát hiện 10 nhóm có nội dung trùng ngoài cache. Chúng là snapshot mã đã chạy, manifest nguồn hoặc báo cáo graph theo seed; không phải bản sao vô nghĩa. Giữ nguyên để truy nguồn thí nghiệm và kiểm tra hash. Danh sách cụ thể ở trường `retained_duplicate_groups` trong receipt.

Hai notebook Colab là hai cách dùng trình duyệt và VS Code; không xóa một bản như file trùng. Dữ liệu raw Parquet khoảng 928 MiB, dữ liệu đã xử lý, checkpoint ngoài repo, mã nguồn, config, test và bằng chứng metric được giữ nguyên. Các hình/PDF trong thư mục lịch sử dùng để đọc các mốc cũ; không dùng làm kết quả mới.

## Sao lưu và khôi phục

Trước khi sửa tài liệu/đường dẫn, đã sao lưu 12 file liên quan vào `D:\Nam3_ki1\UFM-Rec_Backups\cleanup_20261029_nontraining`; `manifest.json` lưu SHA. Các PDF chuyển vị trí vẫn nằm trong repo, có đường dẫn trước/sau và SHA trong [receipt](cleanup_20261029.json). Không có file dữ liệu/thực nghiệm bị xóa.

Cache đã chuyển có thể khôi phục từ `temporary_files` hoặc tái tạo: Python tạo lại bytecode, pytest tạo lại cache, launcher content tạo lại `tmp/mock_demo_data`. Công cụ PDF tạo lại ảnh công thức khi xuất; thư viện render PyMuPDF tạm có trong bản lưu nếu cần rà trang PDF về sau. Đã lưu [receipt kiểm tra 11 trang PDF](nontraining_20261029/pdf_verification.json), không giữ hàng nghìn file runtime tạm trong thư mục làm việc.

## Trạng thái công việc

Sau khi dọn: kiểm tra **340 liên kết Markdown local**, không còn đích thiếu trong phạm vi kiểm tra; `git diff --check` đạt. Bộ test vẫn đạt **116 passed, 3 skipped, 10 subtests passed** (một cảnh báo cũ). Dry-run [preflight](nontraining_20261029/preflight_after_cleanup.txt) đạt **9/9 hash, 15 job**, không train. Nội dung các file tracked trong `src/`, `configs/`, `ops/`, `data/`, `notebooks/`, `tests/` không đổi so với HEAD; test/notebook mới vẫn được giữ ở working tree. Bốn PDF lịch sử và PDF hiện tại đều khớp SHA.

Xem [việc đã hoàn thành và còn thiếu](nontraining_20261029/status.md). Dọn file không thay kết quả train, đánh giá nhiều seed, kiểm tra UI/export hay bộ Word/PPTX cuối. Các thay đổi vẫn ở working tree để nhóm xem lại; chưa tạo commit.
