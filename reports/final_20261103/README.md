# Hoàn thiện đồ án UFM Rec ngày 03/11/2026

Rà từ commit `4b72a2d` cùng các thay đổi chưa commit từ 29/10. Không chạy train; chưa commit/push (nhóm tự commit theo hướng dẫn ở cuối).

## Quyết định của nhóm

- **GVHD: Phan Hồ Viết Trường**, theo bìa mẫu trường ngày 27/10. Báo cáo/slide trước đó ghi nhầm "ThS. Nguyễn Thái Anh"; đã sửa trong báo cáo, slide và các script dựng.
- **Nhóm 4 người:** Nguyễn Thành Phong (2474802010304), Trần An Kỳ (2474802010460), Nguyễn Hoàng Hải (2474802010099) và Hân (chưa có họ tên đầy đủ, MSSV, nhiệm vụ).
- **Chốt đồ án với kết quả hiện có.** Số liệu chính là validation v3, một seed 42; vòng 15 lượt đa seed và hai can thiệp theo train count là hướng phát triển.

## Bộ nộp

- [Word theo mẫu Trường Đại học Văn Lang](../../output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx) và [PDF 20 trang](../../output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf): bìa có viền và logo, lời cảm ơn, bảng phân công, mục lục (số trang La Mã cho phần đầu), từ viết tắt, tóm tắt, sáu chương, tài liệu tham khảo. Dựng bằng `python scripts/build_school_report_docx.py` từ [báo cáo nguồn](../../docs/final_report.md).
- [PPTX 14 slide](../../output/presentation/UFM_Rec_Bao_Ve_20261103.pptx): sửa tên nhóm, GVHD, ngày và dòng hướng tiếp theo trên slide 1, 13, 14.
- SHA-256, kích thước, số trang: [receipt](submission_artifacts.json).

Mẫu trường được khôi phục từ Git: `Bao_Cao_Do_An_UFM_Rec_Nhom_4.docx` ngày 27/10 không còn trên đĩa nhưng còn trong một checkpoint của Cline. Bản này nay nằm ở [output/history/docx](../../output/history/docx/Bao_Cao_Do_An_UFM_Rec_Nhom_4_20261027.docx) và chỉ được dùng làm nguồn bìa. Phần kết quả của nó theo giao thức cũ (ví dụ "UFM Overall 0,285611 vượt TF-IDF") nên không được tái sử dụng.

## Dọn workspace

- `tmp/` (96 file ảnh/PDF render nháp, 19,2 MB) và notebook `UFM_Colab_L4_Train.ipynb` được đưa vào Thùng rác Windows. Notebook L4 chỉ khác chữ T4/L4, nay thay bằng biến `GPU_TYPE` trong [notebook trình duyệt](../../notebooks/UFM_Colab_Browser_Train.ipynb).
- `.git` giảm từ khoảng 1,26 GiB xuống 17 MiB loose: bốn blob mồ côi 1.255 MB (bản sao `bundle_000.zip`, `bundle_003.zip` cùng một bản tải dở và `full-00004-of-00005.parquet`) được đưa vào Thùng rác. Nội dung các gói này đã được khôi phục và đối chiếu SHA trong `D:\Nam3_ki1\UFM-Rec_Backups\restored_complete_20261028`; parquet còn trong `data/raw`. `git fsck` không báo lỗi.
- 114 blob nháp nhỏ khác (4,3 MB) được xuất sang `D:\Nam3_ki1\UFM-Rec_Backups\git_lost_found_20261103` kèm `index.json` trước khi dọn. Năm commit checkpoint mồ côi của Zed và Cline đã được đối chiếu; chúng không chứa nội dung nào thiếu trong lịch sử hoặc thư mục làm việc.
- 21 file được Git giữ nguyên byte (`-text`) đã bị đổi sang CRLF mà `git status` không báo. Chúng đã được trả về đúng byte của commit; nội dung chỉ khác ký tự xuống dòng.
- Chuyển vào `docs/history/`: tin nhắn GVHD 27/10 và 28/10, checklist 27/10, ghi chú sửa FITLAB-01, spec `.kiro` về cold-start. Notebook All Beauty ở thư mục gốc chuyển vào `notebooks/`. Bộ Word/PDF/PPTX hiện trạng 29/10 chuyển vào `output/history/`.
- Kiểm tra lại 386 link nội bộ Markdown/HTML: không còn link hỏng.

## Sửa lỗi

- Hai chỗ trong `tests/test_ufm_core.py` đổi tensor có gradient sang số bằng `float()`, gây cảnh báo; nay dùng `.item()`.
- Mục lục trong báo cáo nguồn trỏ sai tiêu đề Chương 5.
- Script dựng Word/PDF/slide cũ: sửa GVHD và tên nhóm; bản Word/PDF hiện trạng nay xuất vào `output/history/`.
- Thêm [3 test](../../tests/test_school_report.py) cho script dựng báo cáo: mọi bảng/hình có chú thích đánh số, link repo thành đường dẫn in được, thay chữ trên bìa không lặp chữ.

## Kiểm tra

- `python -m pytest tests/ -q`: **119 passed, 3 skipped, 10 subtests passed**, không cảnh báo.
- Đã render và xem cả 20 trang PDF và slide 1, 13, 14. Font Times New Roman được nhúng. File Word có id hình không trùng, đủ liên kết và thứ tự thẻ section hợp lệ. Word chưa được mở trên máy này vì không cài Microsoft Office.

## Việc còn lại

1. Điền họ tên, MSSV, nhiệm vụ và mức hoàn thành của Hân vào `TEAM` trong `scripts/build_school_report_docx.py`, rồi chạy lại script. Sửa chữ "Hân" trên slide 1 bằng PowerPoint.
2. Mở bản Word, nhấn F9 nếu số trang trong mục lục lệch. Rà lại mô tả nhiệm vụ của ba thành viên (lấy từ bản 27/10) và lời cảm ơn (đang dùng "giảng viên hướng dẫn"; có thể đổi thành "Thầy" hoặc "Cô").
3. Diễn tập demo và phần trình bày trên máy bảo vệ, phân công người nói.
4. Commit và push, rồi xem CI trên GitHub Actions.
5. Dọn Thùng rác nếu muốn giải phóng khoảng 1,3 GB.

Thư mục sao lưu ngoài repo (`D:\Nam3_ki1\UFM-Rec_Backups`, khoảng 21 GB) không bị động tới. Nếu cần chỗ trống, các phần trùng lặp là `raw_meta_Toys_and_Games_20261028` (928 MB, trùng parquet trong `data/raw`), cặp `continuation_delta_20261028` và `restored_delta_20261028` (ZIP và bản giải nén, khoảng 163 MB mỗi bản), cùng các file cài đặt `.msi`/`.zip` trong `document_runtime` (bản giải nén cạnh đó mới là thứ script cần). Giữ `restored_complete_20261028` (checkpoint, cache CLIP cho demo), các ZIP trong `colab_training_inputs_20261029` (đầu vào Colab) và các thư mục `*_extracted` của `document_runtime` (dựng báo cáo).
