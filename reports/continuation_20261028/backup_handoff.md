# Bàn giao bản sao dữ liệu ngày 28/10/2026

## Đã kiểm tra trên máy Windows

- **22/22 gói chính**: SHA-256 hợp lệ, tổng archive **13.150.278.921 byte**.
- Đã khôi phục **796 tệp, 15.010.960.088 byte**; `all_files_sha256_verified=true`, không còn tệp thiếu.
- Gói bổ sung **137 tệp**, archive **170.127.949 byte**: predictions validation v3, calibration, kế hoạch chạy, mã nguồn đang làm việc. Đã khôi phục và kiểm tra từng tệp.
- Metadata gốc: **5 shard Parquet, 972.667.016 byte**, cùng manifest đã sao chép và đối chiếu SHA sang thư mục ngoài repo.

Thư mục chính: `D:\Nam3_ki1\UFM-Rec_Backups`.

Các vị trí cần giữ:

- `fitlab01_20261028/`: 22 ZIP, manifest và `restore_receipt.json`.
- `restored_complete_20261028/`: catalog, split, TF-IDF, CLIP features/masks, checkpoints, predictions/config/log gốc và snapshot code đã khôi phục.
- `continuation_delta_20261028/` và `restored_delta_20261028/`: gói bổ sung, manifest và receipt đầy đủ.
- `raw_meta_Toys_and_Games_20261028/`: dữ liệu metadata Parquet và `backup_receipt.json`.
- `_superseded_20261028/`: bản khôi phục thử và các đoạn tải cũ; đã gom riêng sau khi bản đầy đủ đạt checksum.
- `local_source_final_20261028/` và `restored_local_source_final_20261028/`: mã nguồn/tài liệu local ở thời điểm bàn giao, 259 tệp đã kiểm tra khôi phục; dữ liệu lớn được giữ trong các gói riêng kể trên.

Tên Linux `docs/FITLAB.md` và `docs/fitlab.md` cùng tồn tại. Bản thứ hai được bảo toàn dưới `.restore_case_conflicts/`, có mapping trong receipt. Không ghi đè mã đang làm việc bằng source cũ trong snapshot.

## Phạm vi ảnh và thực nghiệm

CLIP image embeddings đã nằm trên máy; embeddings không phải bộ JPEG gốc. Catalog đầy đủ giữ tên/ASIN/URL ảnh, hiển thị ảnh qua mạng. Chế độ mẫu 30 sản phẩm có ảnh offline trong repo. Không gọi đây là đã tải offline toàn bộ 758.692 JPEG.

Demo UFM trên FITLAB-01 đã qua 6 ca API và giao diện zero-shot. Bản UFM dùng toàn bộ dữ liệu vừa khôi phục trên Windows cần được nghiệm thu riêng; các test trên Windows dùng dữ liệu nhỏ, không thay thế bước này. Bản content 30 sản phẩm vẫn phục vụ minh họa offline.

`Start_UFM_From_Backup.cmd` ở thư mục backup truyền đầy đủ đường dẫn catalog/checkpoint/cache/snapshot cho launcher hiện tại. Giao diện đã sửa ảnh cao tràn xuống nội dung và nghiệm thu lại trên UFM thật. File xuất JSON đã tải về sau đó, được chuyển vào [ui_export_ufm_zero_shot.json](ui_export_ufm_zero_shot.json) và kiểm tra backend, checkpoint SHA, catalog size, Top 5, regime và loại history; xem [receipt](ui_export_verification.json). Việc này không thay thế toàn bộ nghiệm thu ứng dụng trên Windows.

## Dọn thư mục

Đã xóa `bundle_000.zip` trùng trong repo sau khi đối chiếu SHA với bản ở ngoài repo; inventory trùng cũng được xử lý tương tự. Gói probe được chuyển sang `fitlab01_20261028/auxiliary/`.

Lệnh xóa đệ quy các bản thử bị kiểm duyệt tự động chặn. Các bản này được chuyển an toàn vào `_superseded_20261028/`, vẫn chiếm dung lượng và vẫn có thể khôi phục. Dữ liệu gốc trên FITLAB được giữ nguyên.
