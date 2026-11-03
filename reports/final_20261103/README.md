# Hoàn thiện đồ án UFM Rec ngày 03/11/2026

Rà từ commit `4b72a2d` cùng các thay đổi chưa commit từ 29/10. Không chạy train. Lần hoàn thiện đầu được nhóm commit (`eaca833`) và đẩy lên cả `main` lẫn `followup/multimodal-20261028`; CI Windows/Ubuntu đạt. Báo cáo 60 trang và đợt lọc trùng bên dưới là thay đổi sau commit đó.

## Quyết định của nhóm

- **GVHD: Phan Hồ Viết Trường**, theo bìa mẫu trường ngày 27/10. Báo cáo/slide trước đó ghi nhầm "ThS. Nguyễn Thái Anh"; đã sửa trong báo cáo, slide và các script dựng.
- **Nhóm 4 người:** Nguyễn Thành Phong (2474802010304), Trần An Kỳ (2474802010460), Nguyễn Hoàng Hải (2474802010099) và Hân (chưa có họ tên đầy đủ, MSSV, nhiệm vụ).
- **Chốt đồ án với kết quả hiện có.** Số liệu chính là validation v3, một seed 42; vòng 15 lượt đa seed và hai can thiệp theo train count là hướng phát triển.

## Bộ nộp

- [Word theo mẫu Trường Đại học Văn Lang](../../output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx) và [PDF 60 trang](../../output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf). Phần đầu 13 trang (bìa không đánh số, các trang sau đánh số La Mã i–xii): bìa, lời cảm ơn, lời nói đầu, trang nhận xét của giảng viên, bảng phân công, mục lục, từ viết tắt, danh mục 36 bảng và 13 hình, tóm tắt. Phần chính (47 trang): bảy chương (giới thiệu; cơ sở lý thuyết; dữ liệu và thiết kế; mô hình và cài đặt; thực nghiệm; demo; kết luận), 18 tài liệu tham khảo và phụ lục. Có 26 công thức đánh số.
- Nội dung tổng hợp từ các tài liệu `.md` trong `docs/` (đề xuất, phạm vi, trích xuất CLIP, huấn luyện, giao thức, demo, Colab, kịch bản bảo vệ) và bằng chứng trong `reports/`. Mọi số liệu lấy từ tệp bằng chứng; số trong bản Word 27/10 không được dùng lại khi chưa kiểm chứng.
- [PPTX 14 slide](../../output/presentation/UFM_Rec_Bao_Ve_20261103.pptx): sửa tên nhóm, GVHD, ngày và dòng hướng tiếp theo trên slide 1, 13, 14.
- Dựng lại: `python scripts/build_report_figures.py` (10 hình từ dữ liệu thật, gồm [lịch sử huấn luyện 12 epoch](ufm_full_v1_training_history.json)) rồi `python scripts/build_school_report_docx.py`. SHA-256, kích thước, số trang: [receipt](submission_artifacts.json).

Mẫu trường được khôi phục từ Git: `Bao_Cao_Do_An_UFM_Rec_Nhom_4.docx` ngày 27/10 không còn trên đĩa nhưng còn trong một checkpoint của Cline. Bản này nay nằm ở [output/history/docx](../../output/history/docx/Bao_Cao_Do_An_UFM_Rec_Nhom_4_20261027.docx) và chỉ được dùng làm nguồn bìa. Phần kết quả của nó theo giao thức cũ (ví dụ "UFM Overall 0,285611 vượt TF-IDF") hoặc chưa có nguồn (ví dụ "3,8 triệu người dùng"; số thật là 6.158.534) nên không được tái sử dụng.

## Dọn workspace

- `tmp/`, các thư mục `__pycache__` và notebook `UFM_Colab_L4_Train.ipynb` được đưa vào Thùng rác Windows. Notebook L4 chỉ khác chữ T4/L4, nay thay bằng biến `GPU_TYPE` trong [notebook trình duyệt](../../notebooks/UFM_Colab_Browser_Train.ipynb).
- `.git` giảm từ khoảng 1,26 GiB xuống 17 MiB loose: bốn blob mồ côi 1.255 MB (bản sao `bundle_000.zip`, `bundle_003.zip` cùng một bản tải dở và `full-00004-of-00005.parquet`) được đưa vào Thùng rác. Nội dung các gói này đã được khôi phục và đối chiếu SHA trong `D:\Nam3_ki1\UFM-Rec_Backups\restored_complete_20261028`; parquet còn trong `data/raw`. `git fsck` không báo lỗi.
- 114 blob nháp nhỏ khác (4,3 MB) được xuất sang `D:\Nam3_ki1\UFM-Rec_Backups\git_lost_found_20261103` kèm `index.json` trước khi dọn. Năm commit checkpoint mồ côi của Zed và Cline đã được đối chiếu; chúng không chứa nội dung nào thiếu trong lịch sử hoặc thư mục làm việc.
- 21 file được Git giữ nguyên byte (`-text`) đã bị đổi sang CRLF mà `git status` không báo. Chúng đã được trả về đúng byte của commit; nội dung chỉ khác ký tự xuống dòng.
- Chuyển vào `docs/history/`: tin nhắn GVHD 27/10 và 28/10, checklist 27/10, ghi chú sửa FITLAB-01, spec `.kiro` về cold-start. Notebook All Beauty ở thư mục gốc chuyển vào `notebooks/`. Bộ Word/PDF/PPTX hiện trạng 29/10 chuyển vào `output/history/`.

## Lọc file trùng nội dung

Quét SHA-256 toàn bộ project và thư mục sao lưu: project chỉ có 11 nhóm trùng (khoảng 90 KB), phần trùng lớn nằm ở thư mục sao lưu. 37 mục (1.674.111.820 byte) được đưa vào Thùng rác, mỗi mục chỉ bỏ khi còn giữ một bản giống hệt (cùng SHA-256, hoặc cùng kích thước và CRC trong ZIP gốc) hoặc là tệp tạm/bộ cài đã giải nén. Danh sách từng mục và bản được giữ: [biên nhận lọc trùng](dedup_20261103.json).

- Trong project: 9 tệp của `reports/fix_1_15_20261027/fitlab01_finish1/executed_source/` giống hệt mã hiện tại (giữ `reevaluate_ufm_checkpoints.py` vì khác) và một manifest trùng `source_manifest.json`.
- Trong thư mục sao lưu: bản sao 928 MB parquet trùng `data/raw`; hai thư mục giải nén `restored_delta_20261028` và `restored_local_source_final_20261028` (mọi tệp có trong ZIP gốc hoặc bản sao đầy đủ); `bundle_005/` giải nén dở; slide nháp; bộ cài LibreOffice/Pandoc/Poppler đã giải nén; các tệp mẫu `evaluation_v2` trùng trong bản dọn 28/10; bản sao tài liệu trong `git_lost_found_20261103`.
- Giữ có chủ đích: ba tệp `graph_report.json` của ba seed BPR-MF (kết quả của ba lượt chạy, trùng vì cùng đồ thị); `restored_complete_20261028` và các ZIP đầu vào Colab (bản sao lưu đầy đủ mà demo và notebook dùng).

## Sửa lỗi

- Hai chỗ trong `tests/test_ufm_core.py` đổi tensor có gradient sang số bằng `float()`, gây cảnh báo; nay dùng `.item()`.
- Script dựng Word/PDF/slide cũ: sửa GVHD và tên nhóm; bản Word/PDF hiện trạng nay xuất vào `output/history/`.
- Thêm [4 test](../../tests/test_school_report.py) cho script dựng báo cáo: bảng/hình có chú thích đánh số liên tục, số công thức tách khỏi công thức, link repo thành đường dẫn in được, thay chữ trên bìa không lặp chữ.

## Kiểm tra

- `python -m pytest tests/ -q`: **120 passed, 3 skipped, 10 subtests passed**, không cảnh báo.
- Đã render và xem cả 60 trang PDF; LibreOffice không báo lỗi công thức nào; mục lục và danh mục bảng/hình giống nhau giữa hai lượt dàn trang. Font được nhúng. File Word có 16 hình với id không trùng, đủ liên kết và thứ tự thẻ section hợp lệ. Word chưa được mở trên máy này vì không cài Microsoft Office.
- Kiểm tra 393 link nội bộ Markdown/HTML: không còn link hỏng.

## Việc còn lại

1. Điền họ tên, MSSV, nhiệm vụ và mức hoàn thành của Hân vào `TEAM` trong `scripts/build_school_report_docx.py`, rồi chạy lại script. Sửa chữ "Hân" trên slide 1 bằng PowerPoint.
2. Mở bản Word, nhấn F9 nếu số trang trong mục lục lệch. Rà lại mô tả nhiệm vụ của ba thành viên (lấy từ bản 27/10) và lời cảm ơn (đang dùng "giảng viên hướng dẫn"; có thể đổi thành "Thầy" hoặc "Cô").
3. Diễn tập demo và phần trình bày trên máy bảo vệ, phân công người nói.
4. Commit và push các thay đổi sau `eaca833`, rồi xem CI.
5. Dọn Thùng rác nếu muốn thật sự giải phóng khoảng 2,9 GB.
