# UFM Rec

Hệ khuyến nghị sản phẩm dùng lịch sử hành vi, ảnh và tên/mô tả trên Amazon Reviews 2023 Toys & Games. Nhánh hành vi dùng Transformer; nhánh semantic dùng CLIP đóng băng và adapter. Cold-start sản phẩm là nhóm phân tích trọng tâm.

**Trạng thái ngày 03/11/2026:** nhóm chốt đồ án với checkpoint và bằng chứng hiện có. Bộ nộp theo mẫu Trường Đại học Văn Lang đã dựng và rà từng trang. Còn điền họ tên/MSSV/nhiệm vụ của thành viên Hân, diễn tập và commit/push. Chi tiết: [hồ sơ hoàn thiện 03/11](reports/final_20261103/README.md).

## Bộ nộp

- [Báo cáo Word theo mẫu trường](output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx) · [PDF 60 trang](output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf) · [nguồn báo cáo](docs/final_report.md): bảy chương, 36 bảng, 13 hình, 26 công thức, phụ lục. Dựng lại bằng `python scripts/build_report_figures.py` rồi `python scripts/build_school_report_docx.py`.
- [Slide bảo vệ 14 trang](output/presentation/UFM_Rec_Bao_Ve_20261103.pptx) · [nguồn slide](docs/presentation_slides.md) · [kịch bản bảo vệ và câu hỏi](docs/defense_guide.md).
- [Demo](docs/demo.md): mở [Start_Demo.cmd](Start_Demo.cmd).
- [Bắt đầu](START.md) · [mục lục tài liệu](docs/README.md) · [mục lục bằng chứng](reports/README.md) · [các bản xuất](output/README.md).

## Kết quả chính

Validation v3 có 40.000 ca, 100 ứng viên/ca, một seed lịch sử 42. Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only CLIP 0,190036; UFM full 0,182056; popularity 0,042527**. Full chưa vượt hai đối chứng nội dung; chênh lệch lớn nhất ở nhóm 1–5 tương tác. Tám checkpoint UFM và hai baseline dùng cùng candidate hash; [JSON đối chiếu](reports/nontraining_20261029/evidence.json) truy được nguồn.

Checkpoint chọn bằng validation v1; v3 dùng lại quần thể validation liên quan nên không phải holdout độc lập. Test v2 đã được xem. Sampling theo regime target và thống kê cuối split có tính hồi cứu; kết quả không tương đương full-catalog hoặc online. [Giao thức](docs/protocol_current.md) và [ledger test](reports/test_evaluation_ledger.json) ghi giới hạn.

Demo UFM trên Windows dùng checkpoint epoch 9, catalog 767.045 sản phẩm và counts thật: [API 6/6 ca](reports/nontraining_20261029/demo_api_verification.json), [UI tự động 4/4 ca](reports/nontraining_20261029/demo_ui_end_to_end_20261029.json). Kiểm thử local ngày 03/11 đạt **120 passed, 3 skipped, 10 subtests passed**.

## Chạy và tái lập

Mở [Start_Demo.cmd](Start_Demo.cmd): launcher chọn UFM khi có checkpoint hoàn tất, hoặc content khi thiếu checkpoint. Có thể chọn rõ `Start_Demo.cmd --backend content` hoặc `Start_Demo.cmd --backend ufm`. Chế độ content tự tạo catalog mẫu 30 sản phẩm trong `tmp/mock_demo_data`; counts của catalog mẫu là mô phỏng.

Máy hiện tại có bản sao đầy đủ ở `D:\Nam3_ki1\UFM-Rec_Backups\restored_complete_20261028` và launcher ngoài Git `D:\Nam3_ki1\UFM-Rec_Backups\Start_UFM_From_Backup.cmd`. [Bàn giao backup](reports/continuation_20261028/backup_handoff.md) ghi checksum và khôi phục.

```powershell
python -m pytest tests/ -q
python scripts/build_current_evidence.py
python scripts/build_report_figures.py
python scripts/build_school_report_docx.py
```

Lệnh dựng báo cáo không train. Nó cần python-docx và bộ Pandoc/LibreOffice portable trong `D:\Nam3_ki1\UFM-Rec_Backups\colab_training_inputs_20261029\document_runtime` (đổi bằng biến `UFM_DOC_RUNTIME`, `UFM_PANDOC`). Bìa lấy từ [bản Word mẫu trường ngày 27/10](output/history/docx/Bao_Cao_Do_An_UFM_Rec_Nhom_4_20261027.docx). [Dữ liệu](data/README.md) lớn giữ ngoài Git.

## Hướng phát triển

[Manifest 15 run](configs/multimodal_followup_20261028.json) đã khóa năm biến thể × ba seed (42, 7, 2026), gồm hai can thiệp `count_aware_fusion` và `adaptive_dropout`. [Hướng dẫn Colab](docs/colab.md) dùng notebook trình duyệt (T4 mặc định, L4 bằng `GPU_TYPE`) hoặc VS Code; notebook lưu và resume theo đoạn. [Runbook GPU](docs/gpu_followup_runbook.md) và [bước tiếp theo](docs/next.md) ghi trình tự. [Đề xuất metric/holdout gửi giảng viên ngày 30/10](docs/trao_doi_gvhd_2026_10_30.md) áp dụng cho vòng này.

## Thư mục

- `src/`, `tests/`, `scripts/`: mô hình, pipeline, demo và công cụ tái lập.
- `docs/`: nguồn báo cáo/slide và hướng dẫn; `docs/history/` giữ tài liệu cũ (tin nhắn GVHD 27–28/10, checklist 27/10, ghi chú FITLAB, spec Kiro).
- `reports/`: metric, receipt, manifest và snapshot nguồn.
- `output/`: chỉ chứa bộ nộp hiện hành; `output/history/` giữ mọi bản Word/PDF/PPTX cũ.
- `notebooks/`: notebook Colab, notebook FITLAB cũ và notebook All Beauty.
- `data/`, `runs/`: dữ liệu/checkpoint; không xóa khi dọn cache. `tmp/`, `__pycache__/`, `.pytest_cache/` có thể tái tạo.

Proposal cũ còn dùng để đối chiếu phạm vi: [Markdown](docs/proposal.md), [HTML](docs/proposal.html), [PDF lịch sử](output/history/pdf/proposal.pdf). [README lịch sử](docs/history/readme-20261002.md) và [proposal ban đầu](docs/history/proposal-v1.pdf) giữ mốc cũ.
