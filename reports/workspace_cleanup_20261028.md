# Dọn workspace ngày 28/10/2026

Đã đưa **998 tệp, 125.311.382 byte (khoảng 125 MB)** ra khỏi thư mục làm việc:

- **937 tệp / khoảng 93,5 MB** trong 7 thư mục cache và bộ pyarrow cài tạm: đưa vào **Recycle Bin**, có thể phục hồi bằng chức năng thông thường của Windows.
- **61 tệp / khoảng 31,8 MB** trong 31 đường dẫn: chuyển sang `D:\Nam3_ki1\UFM-Rec_Backups\workspace_cleanup_20261028`, đối chiếu SHA-256 từng tệp sau khi chuyển. Gồm transfer cũ, bản sao sửa lỗi, log/ảnh thử, script sửa một lần và ba tiện ích chuyển dữ liệu đã dùng xong.
- File export `ufm-rec-demo.json` được chuyển từ gốc repo vào `reports/continuation_20261028/ui_export_ufm_zero_shot.json`. Đã xác minh đây là UFM full thật, đúng checkpoint, catalog 767.045 sản phẩm và 5 kết quả zero-shot không trùng history.

`tmp/` hiện chỉ giữ `mock_demo_data/` vì demo offline cần catalog/ảnh này. Giữ `.git`, CI, mã nguồn, tests, cấu hình, requirements, notebooks, dữ liệu gốc, hình demo, tài liệu, Word/PDF liên quan và toàn bộ bằng chứng thực nghiệm. Các bản báo cáo cũ còn được giữ vì liên quan đồ án; trạng thái chưa đồng bộ với số liệu mới đã được ghi rõ.

Không dọn thư mục backup/checkpoint/dataset ngoài repo. Bản khôi phục đầy đủ và các receipt SHA trước đó được giữ nguyên.

## Khôi phục khi cần

- Tệp chuyển ra: xem `workspace_cleanup_20261028/cleanup_manifest.json`; trường `source`, `destination` và SHA ghi chính xác vị trí trước/sau.
- Cache/công cụ cài tạm: có thể phục hồi từ Recycle Bin hoặc cài lại dependency bằng requirements phù hợp.
- Bản `docs/next.md` trước khi cập nhật nằm ở `workspace_cleanup_20261028/docs_next_before_cleanup.md`.

Việc dọn này thu gọn folder dự án; dữ liệu trong Recycle Bin và thư mục lưu riêng vẫn chiếm dung lượng ổ đĩa.

## Kiểm tra sau khi dọn

Đã chạy lại `python -m pytest tests/ -q -p no:cacheprovider` với `PYTHONDONTWRITEBYTECODE=1`: **93 passed, 3 skipped, 10 subtests passed** trong 24,52 giây. Ba skip dành riêng Linux; warning chuyển tensor sang scalar đã có từ trước. Log nằm tại `workspace_cleanup_20261028/post_cleanup_tests.log` trong thư mục lưu ngoài repo. `git diff --check` đạt.

File export UFM được kiểm tra độc lập bằng [receipt](continuation_20261028/ui_export_verification.json). `tmp/` chỉ còn catalog demo offline; các cache được dọn không bị tạo lại trong lượt kiểm thử này.

Hướng đi tiếp theo: [kế hoạch ưu tiên](../docs/next.md).
