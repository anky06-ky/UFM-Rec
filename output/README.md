# Các bản xuất của UFM Rec

## Bộ nộp hiện hành ngày 03/11/2026

- [Báo cáo Word theo mẫu Trường Đại học Văn Lang](docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx) và [PDF 61 trang](pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf): bìa mẫu trường, lời cảm ơn, lời nói đầu, trang nhận xét của giảng viên, bảng phân công, mục lục, từ viết tắt, danh mục bảng/hình, tóm tắt, bảy chương, tài liệu tham khảo và phụ lục. Dựng từ [nguồn báo cáo](../docs/final_report.md) bằng `scripts/build_report_figures.py` và `scripts/build_school_report_docx.py`; đã render và xem từng trang.
- [Slide bảo vệ 14 trang](presentation/UFM_Rec_Bao_Ve_20261103.pptx): cùng số liệu với báo cáo; [nguồn slide](../docs/presentation_slides.md).
- Hình của báo cáo: `figures/report_20261103/` (10 hình dựng từ bằng chứng) cùng [biểu đồ cold macro](figures/current_v3_20261029/cold_macro.png) và [chênh lệch theo train count](figures/current_v3_20261029/count_effect.png) (có SVG cùng tên để biên tập).
- SHA-256, kích thước và số trang: [receipt bộ nộp](../reports/final_20261103/submission_artifacts.json).

Ô họ tên/MSSV/nhiệm vụ của Hân trên bìa và bảng phân công đang để trống có đánh dấu `[...]`; điền rồi dựng lại trước khi nộp.

Mở tệp Word bằng Microsoft Word. Extension Office Viewer của VS Code lưu đè tệp khi mở và làm mất dòng thành viên trên bìa cùng một dòng mục lục.

## Bản lịch sử

`history/` giữ mọi bản cũ, không dùng để nộp:

- `history/docx/`: báo cáo Word theo mẫu trường ngày 27/10 (số liệu giao thức cũ, khôi phục từ Git và là nguồn bìa của bản hiện hành) và Word hiện trạng 29/10.
- `history/pdf/`: proposal, báo cáo chi tiết 28/09, tiến trình 30/09, tiến độ Nhóm 4 ngày 27/10, tiến độ 28/10 và PDF hiện trạng 29/10.
- `history/presentation/`: PPTX hiện trạng 29/10 (GVHD và tên nhóm chưa sửa).
- `history/figures/`: hình của proposal, báo cáo 28/09–30/09 và validation 03/10; tài liệu trong `docs/history/` vẫn tham chiếu.
- `history/demo/`: ảnh và JSON demo ngày 30/09, không phải lượt nghiệm thu 29/10.
