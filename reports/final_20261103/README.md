# Hoàn thiện đồ án UFM Rec ngày 03/11/2026

Rà từ commit `4b72a2d` cùng các thay đổi chưa commit từ 29/10. Không chạy train. Lần hoàn thiện đầu là commit `eaca833`; báo cáo 60 trang và đợt lọc trùng là commit `7376900`. Cả hai đã có trên `main` lẫn `followup/multimodal-20261028`, CI Windows/Ubuntu đạt. Phần "Sắp xếp lại tài liệu" là commit `9e92aba`, "Tinh gọn thư mục" là commit `431c1b4`; gộp requirements và phần "Ghép UFM với TF-IDF" là commit `c272748`, hai chế độ mới của demo là commit `13785a9`; các commit này đã có trên `main` lẫn `followup/multimodal-20261028`, CI đạt. Phần "Tăng tốc demo" ngày 04/11 (sửa chuẩn hóa, số đo, báo cáo và slide) là thay đổi sau đó.

## Quyết định của nhóm

- **GVHD: Phan Hồ Viết Trường**, theo bìa mẫu trường ngày 27/10. Báo cáo/slide trước đó ghi nhầm "ThS. Nguyễn Thái Anh"; đã sửa trong báo cáo, slide và các script dựng.
- **Nhóm 3 người (xác nhận ngày 04/11):** Nguyễn Thành Phong (2474802010304), Trần An Kỳ (2474802010460) và Nguyễn Hoàng Hải (2474802010099). Đã bỏ dòng thành viên thứ tư khỏi bảng phân công, metadata báo cáo và slide 1; cập nhật nguồn dựng để các lần xuất sau vẫn giữ đúng ba người.
- **Chốt đồ án với kết quả hiện có.** Số liệu chính là validation v3, một seed 42; vòng 15 lượt đa seed và hai can thiệp theo train count là hướng phát triển.

## Bộ nộp

- [Word theo mẫu Trường Đại học Văn Lang](../../output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx) và [PDF 63 trang](../../output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf). Phần đầu 13 trang (bìa không đánh số, các trang sau đánh số La Mã i–xii): bìa, lời cảm ơn, lời nói đầu, trang nhận xét của giảng viên, bảng phân công, mục lục, từ viết tắt, danh mục 38 bảng và 14 hình, tóm tắt. Phần chính (50 trang): bảy chương (giới thiệu; cơ sở lý thuyết; dữ liệu và thiết kế; mô hình và cài đặt; thực nghiệm; demo; kết luận), 18 tài liệu tham khảo và phụ lục. Có 27 công thức đánh số.
- Nội dung tổng hợp từ các tài liệu `.md` trong `docs/` (đề xuất, phạm vi, trích xuất CLIP, huấn luyện, giao thức, demo, Colab, kịch bản bảo vệ) và bằng chứng trong `reports/`. Mọi số liệu lấy từ tệp bằng chứng; số trong bản Word 27/10 không được dùng lại khi chưa kiểm chứng.
- [PPTX 14 slide](../../output/presentation/UFM_Rec_Bao_Ve_20261103.pptx): sửa tên nhóm, GVHD, ngày và dòng hướng tiếp theo trên slide 1, 13, 14. Ngày 03/11 thêm kết quả ghép UFM + TF-IDF vào slide 7 và 13; ngày 04/11 viết lại slide 10 theo số đo tăng tốc và slide 11 với 5/5 ca UI cùng ảnh giao diện mới; và sửa nhãn chương lệch theo báo cáo cũ ở slide 3, 4, 5, 13 (cả ghi chú thuyết trình).
- Dựng lại: `python scripts/build_report_figures.py` (11 hình từ dữ liệu thật, gồm [lịch sử huấn luyện 12 epoch](ufm_full_v1_training_history.json)) rồi `python scripts/build_school_report_docx.py`. SHA-256, kích thước, số trang: [receipt](submission_artifacts.json).

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

## Sắp xếp lại tài liệu

`docs/` chỉ còn 7 tệp: mục lục, nguồn báo cáo, nguồn slide, kịch bản bảo vệ, hướng dẫn demo, hướng dẫn Colab và runbook GPU.

- 14 tài liệu theo mốc chuyển vào `docs/history/`: proposal 2.0, phạm vi, giao thức, kế hoạch 28–29/10, tin nhắn GVHD 30/10, thí nghiệm 03/10, sáu ghi chép FITLAB (môi trường, CLIP, huấn luyện, GPU, vận hành, khôi phục), sao lưu và trang giới thiệu HTML cũ. Link trong mọi tệp Markdown/HTML được cập nhật theo.
- Đưa vào Thùng rác `docs/proposal.html` (trùng 99,9% `proposal.md`; PDF lịch sử còn trong `output/history/pdf/`). `build_proposal.py` nay chỉ dựng PDF và SVG từ `docs/history/proposal.md`; chạy thử trong thư mục tạm cho văn bản PDF và 5 SVG giống hệt bản cũ.
- Đưa vào Thùng rác `build_report_docx.py`, `polish_report_docx.py`, `latex_to_omml.py`: bộ dựng Word 27/10 cần `Mau_Bao_Cao_Nhom_4.docx` và `report_equations_omml.json`, hai tệp chưa từng có trong Git, nên không chạy được. Bản Word 27/10 vẫn ở `output/history/docx/`.
- Giữ `proposal-v1.html` và hai báo cáo tiến độ 28/09 vì đo độ trùng cho thấy chúng là tài liệu riêng: đề xuất TF-IDF/SVD/RRF tháng 9 khác proposal UFM v1; hai báo cáo 28/09 chỉ trùng 2–17% với bản 22:23.
- `status.py` và notebook `00_status` trỏ sang runbook GPU thay cho `docs/next.md`.
- [Hướng dẫn demo](../../docs/demo.md) viết lại và đối chiếu với mã demo cùng bằng chứng: lịch sử `B004S8F7QM` như ảnh dự phòng, độ trễ 15–28 giây toàn catalog và 6–11 giây khi lọc nhóm, nhãn "CF · Semantic", lỗi 429/400, thiếu thư viện.
- Báo cáo mục 6.3 sửa một câu: chế độ `auto` chọn UFM chỉ dựa vào checkpoint hoàn tất; thiếu tệp hoặc sai mã băm thì demo dừng chứ không chuyển sang chế độ nội dung. Dựng lại vẫn 60 trang; so văn bản từng trang, chỉ trang 8 (mục 6.5 sang trang 40) và trang 38–40 thay đổi.

## Tinh gọn thư mục

Quét SHA-256 toàn repo: 388 tệp, chỉ còn một nhóm trùng có chủ đích (`graph_report.json` của ba seed BPR-MF); không cặp thư mục nào chung từ một nửa nội dung trở lên.

- `tmp/` (255 tệp, 157 thư mục, phần lớn là hồ sơ tạm của LibreOffice khi dựng báo cáo) và ba thư mục `__pycache__` được đưa vào Thùng rác. `build_school_report_docx.py` nay để tệp tạm trong `%TEMP%\ufm_rec_school_report`; dựng thử ra PDF 60 trang có văn bản giống hệt bản đã commit.
- Bản cũ trong `output/` chuyển vào `output/history/`: `demo/` ngày 30/09 cùng các hình proposal, báo cáo 28/09–30/09 và `validation_20261003/`. `build_proposal.py` và `plot_validation_audit.py` ghi vào vị trí mới. `output/figures/` chỉ còn hai bộ hình báo cáo đang dùng.
- Gộp năm tệp requirements thành một `requirements.txt`, giữ nguyên các phiên bản đã ghim, thêm `python-docx` còn thiếu và bỏ `requirements-lock.txt` rỗng. CI nay cài một tệp; vì có `python-docx`, 4 test của bộ dựng báo cáo không còn bị bỏ qua trên CI. Đã đối chiếu trên PyPI: các gói mới có bản cho Python 3.12 trên Windows và Linux, không xung đột với phiên bản đã ghim.
- Sau commit `9e92aba`, tệp Word bị extension Office Viewer của VS Code lưu đè: mất dòng "SVTH 4" trên bìa và dòng "LỜI CẢM ƠN" trong mục lục. Đã khôi phục bản trong commit; SHA khớp receipt.
- Thư mục sao lưu: sáu ZIP đầu vào Colab (3,33 GiB) được đưa vào Thùng rác theo yêu cầu của nhóm, sau khi 49/49 tệp nguồn trong `restored_complete_20261028` khớp SHA-256 với `manifest.json`. Giữ `restored_complete_20261028` vì checkpoint, đặc trưng CLIP và dữ liệu đã xử lý không có trên GitHub (`.gitignore` loại `*.pt`, `*.npy`, `*.gz`; GitHub chỉ có 390 tệp, khoảng 27 MB), và giữ `document_runtime` để dựng lại báo cáo.

## Ghép UFM với TF-IDF

Nhóm chọn thử tối ưu không cần huấn luyện lại. [`scripts/evaluate_hybrid_v3.py`](../../scripts/evaluate_hybrid_v3.py) tính lại điểm của TF-IDF và UFM full cho 40.000 ca × 100 ứng viên của validation v3 trên CPU (229 giây), chuẩn hóa điểm theo từng ca rồi trộn α·z(UFM) + (1 − α)·z(TF-IDF).

- Điểm tính lại khớp hoàn toàn số đã công bố ở cả bốn nhóm (độ lệch 0), và hai đầu mút α = 0, α = 1 cho đúng thứ hạng của từng mô hình.
- α được chọn trên 20.000 ca sớm theo thời gian (cùng cách chia với hiệu chuẩn), α = 0,55. Kết quả báo cáo trên 20.000 ca muộn: cold macro **0,215540** so với 0,203751 của TF-IDF và 0,189057 của UFM full. Chênh lệch +0,011789, khoảng tin cậy 95% bootstrap [0,009796; 0,013744]. Kết hợp hơn TF-IDF ở cả bốn nhóm. Chi tiết: [`hybrid_v3.json`](../hybrid_20261103/hybrid_v3.json).
- Báo cáo thêm mục 5.12 (nay là Bảng 5.15, vì mục 5.11 có thêm một bảng ngày 04/11; công thức 5.1; mục Thảo luận thành 5.13), cập nhật tóm tắt, mục 1.6, 7.1, 7.2 và 7.3. Slide 7, 13 và câu hỏi 11 trong [kịch bản bảo vệ](../../docs/defense_guide.md) nêu kết quả cùng giới hạn: phân tích sau trên validation, một seed, demo chưa dùng điểm ghép. Ngày 04/11 demo đã có chế độ này (mục dưới).

## Tăng tốc demo và hai chế độ mới (04/11)

Nhóm chọn sửa giao diện, thêm chế độ UFM + TF-IDF vào demo và tăng tốc demo; không huấn luyện lại, không đổi checkpoint.

- **Tính sẵn vector ngữ nghĩa của sản phẩm** khi khởi động; bảng CLIP được ánh xạ từ tệp và chỉ đọc lúc đó. Phát lại 6 ca API ngày 29/10 ở chế độ toàn catalog: cùng sản phẩm, cùng thứ tự, điểm lệch tối đa 3,3 × 10⁻⁶. Chấm toàn catalog p50 7,53 giây (ngày 29/10: 14,8–16,2 giây cho cùng yêu cầu).
- **Chế độ Nhanh:** UFM chấm lại hợp của 2.000 ứng viên theo vector ngữ nghĩa, 2.000 theo vector hành vi, 1.000 theo TF-IDF và 500 phổ biến nhất; p50 0,42 giây (UFM) và 0,60 giây (UFM + TF-IDF). Trên 25 lịch sử validation v3, top 10 trùng 84,4% với toàn catalog ở UFM (70,8% ở zero-shot, 100% ở lạnh và ấm) và 98,8–100% ở UFM + TF-IDF.
- **UFM + TF-IDF trong demo:** dùng α = 0,55 của mục 5.12; cả hai phạm vi chuẩn hóa theo toàn bộ sản phẩm được phép, chế độ Nhanh ước lượng phân phối điểm UFM từ mẫu cố định 4.096 sản phẩm. Bản đầu chuẩn hóa trên tập ứng viên của chế độ Nhanh nên top 10 chỉ trùng 64% với toàn catalog; đã sửa trước lượt đo cuối.
- **Giao diện:** thêm ô Mô hình và Phạm vi chấm, dòng tóm tắt (thời gian, mô hình, phạm vi, số sản phẩm đã chấm), thanh trọng số Hành vi/Ngữ nghĩa, nút đếm giây khi chờ, dải cảnh báo rút gọn có nút "Chi tiết". Chrome tự động đạt 5/5 ca trên catalog thật, thêm ca UFM + TF-IDF toàn catalog.
- **Bộ nhớ và độ trễ:** sau khi khởi động, tiến trình dùng khoảng 2,6–2,7 GiB bộ nhớ riêng, cao nhất 2,9 GiB; working set 3,9 GiB gồm các trang ánh xạ từ tệp CLIP. RAM trống trước lượt đo khoảng 4,9 GB; trước lượt nghiệm thu giao diện chỉ khoảng 1,1 GB, khi đó một lượt UFM + TF-IDF toàn catalog mất 13,3 giây.
- **Tài liệu:** báo cáo viết lại mục 5.11 (thêm bảng độ trễ theo chế độ và bảng độ trùng), cập nhật mục 3.8 (PCN-04 đạt ở chế độ Nhanh), 4.12, 6.1, 6.2 (thêm hình chế độ ghép), 6.4, 6.5, 7.2, 7.3, tóm tắt và phụ lục; slide 10 (tăng tốc) và 11 (5/5 ca UI, ảnh mới). [Hướng dẫn demo](../../docs/demo.md) và [kịch bản bảo vệ](../../docs/defense_guide.md) cập nhật chế độ, thời gian và ảnh dự phòng.
- **Bằng chứng:** [`demo_modes.json`](../demo_20261103/demo_modes.json) (dựng bằng `scripts/evaluate_demo_modes.py`), [nghiệm thu giao diện](../demo_20261103/demo_ui_end_to_end_20261103.json) (dựng bằng `scripts/verify_demo_ui_chrome.mjs`). Script đo từ chối ghi đè; số liệu cuối lấy từ lượt chạy với mã đã sửa chuẩn hóa.

## Sửa lỗi

- Hai chỗ trong `tests/test_ufm_core.py` đổi tensor có gradient sang số bằng `float()`, gây cảnh báo; nay dùng `.item()`.
- Script dựng Word/PDF/slide cũ: sửa GVHD và tên nhóm; bản Word/PDF hiện trạng nay xuất vào `output/history/`.
- Thêm [4 test](../../tests/test_school_report.py) cho script dựng báo cáo: bảng/hình có chú thích đánh số liên tục, số công thức tách khỏi công thức, link repo thành đường dẫn in được, thay chữ trên bìa không lặp chữ.

## Kiểm tra

- Ngày 04/11: `python -m pytest tests/ -q` đạt **121 passed, 3 skipped, 10 subtests passed**. Bản 63 trang đã render và xem lại các trang thay đổi (tóm tắt, mục 3.8, 4.12, 5.11, Chương 6, 7.2–7.3, phụ lục C); font vẫn được nhúng. Slide 10 và 11 được render bằng LibreOffice để kiểm tra bố cục. Kiểm tra 411 link nội bộ: không link hỏng.
- `python -m pytest tests/ -q`: **120 passed, 3 skipped, 10 subtests passed**, không cảnh báo; chạy lại sau khi sắp xếp tài liệu cho cùng kết quả.
- Bản 61 trang sau khi thêm mục 5.12: so văn bản từng trang với bản trước, chỉ mục lục, danh mục, tóm tắt, mục 1.6 và các trang từ mục 5.12 trở đi thay đổi; đã render và xem lại trang tóm tắt, mục 5.12–5.13 và Chương 7; font vẫn được nhúng. Slide 7 và 13 được render bằng LibreOffice để kiểm tra bố cục.
- Bản 60 trang trước đó: đã render và xem cả 60 trang PDF; LibreOffice không báo lỗi công thức nào; mục lục và danh mục bảng/hình giống nhau giữa hai lượt dàn trang. Font được nhúng. File Word có 16 hình với id không trùng, đủ liên kết và thứ tự thẻ section hợp lệ. Word chưa được mở trên máy này vì không cài Microsoft Office.
- Kiểm tra 395 link nội bộ Markdown/HTML sau khi sắp xếp lại: không còn link hỏng.

## Việc còn lại

1. Rà thông tin ba thành viên và phân công nhiệm vụ trong Word/PDF; đối chiếu slide 1 với `TEAM` trong `scripts/build_school_report_docx.py`.
2. Mở bản Word, nhấn F9 nếu số trang trong mục lục lệch. Rà lại mô tả nhiệm vụ của ba thành viên (lấy từ bản 27/10) và lời cảm ơn (đang dùng "giảng viên hướng dẫn"; có thể đổi thành "Thầy" hoặc "Cô").
3. Diễn tập demo và phần trình bày trên máy bảo vệ, phân công người nói.
4. Kiểm tra CI trên GitHub sau khi cập nhật đợt tăng tốc demo ngày 04/11 và bộ nộp ba thành viên.
5. Thùng rác ổ D: đang giữ khoảng 7,4 GB (giới hạn 8 GB), gồm sáu ZIP Colab. Dọn Thùng rác thì các tệp này mất hẳn.
6. Sao lưu `restored_complete_20261028` (hoặc ít nhất `runs/ufm_full_v1`, `foundation_clip_b32_v1` và `content/`) sang Google Drive hoặc ổ ngoài. Model đã train không có trên GitHub; ngoài máy này chỉ còn bản gốc trên FITLAB theo ghi nhận ngày 28/10, mà container FITLAB có thể bị thu hồi.
