# Tài liệu UFM Rec

Đọc [hồ sơ hoàn thiện 03/11](../reports/final_20261103/README.md) trước. Nhóm chốt đồ án với kết quả hiện có; vòng 15 lượt đa seed là hướng phát triển.

## Bộ nộp và bảo vệ

| Tệp nguồn | Nội dung | Bản nộp |
| --- | --- | --- |
| [final_report.md](final_report.md) | Báo cáo 63 trang: bảy chương, tài liệu tham khảo, phụ lục | [Word](../output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx), [PDF](../output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf) |
| [presentation_slides.md](presentation_slides.md) | Nội dung 14 slide | [PPTX](../output/presentation/UFM_Rec_Bao_Ve_20261103.pptx) |
| [defense_guide.md](defense_guide.md) | Kịch bản trình bày và câu hỏi dễ gặp | |
| [demo.md](demo.md) | Chạy demo, kiểm tra trước buổi bảo vệ, xử lý sự cố | |

Bản để đọc và nộp là Word/PDF/PPTX; tệp `.md` là nguồn để sửa. Không sửa số liệu trực tiếp trong Word/PDF: sửa `final_report.md`, chạy `python scripts/build_report_figures.py` rồi `python scripts/build_school_report_docx.py`. Mọi số liệu phải có tệp bằng chứng trong `reports/`. Thông tin bìa (GVHD, thành viên, MSSV, nhiệm vụ) nằm trong `ADVISOR` và `TEAM` của script dựng báo cáo; slide 1 và `presentation_slides.md` phải sửa cùng lúc.

## Khi train tiếp

- [colab.md](colab.md): chạy vòng UFM trên Colab, ngắt phiên và resume.
- [gpu_followup_runbook.md](gpu_followup_runbook.md): chạy vòng đa seed trên FITLAB và mang kết quả về máy.

## Lịch sử

`history/` giữ tài liệu theo mốc. Số liệu trong đó theo giao thức của từng thời điểm; không dùng thay validation v3.

- Đề xuất và phạm vi: [đề xuất ban đầu](history/proposal-v1.html) (TF-IDF, SVD, RRF), [proposal UFM v1](history/proposal-v1.pdf), [proposal 2.0](history/proposal.md), [phạm vi](history/scope.md), [trang giới thiệu cũ](history/overview.html).
- Giao thức và trao đổi: [giao thức](history/protocol_current.md), [kế hoạch 28–29/10](history/next.md), [thí nghiệm 03/10](history/followup.md), [checklist 27/10](history/checklist_hoan_thien_2026_10_27.md), tin nhắn GVHD [27/10](history/trao_doi_gvhd_2026_10_27.md), [28/10](history/trao_doi_gvhd_2026_10_28.md), [30/10](history/trao_doi_gvhd_2026_10_30.md).
- Vận hành FITLAB: [môi trường](history/fitlab.md), [trích CLIP](history/foundation.md), [huấn luyện UFM](history/train.md), [two-tower GPU](history/gpu.md), [lượt chạy dài](history/ops.md), [khôi phục chiến dịch](history/recovery.md), [sao lưu](history/backup_restore.md), [sửa FITLAB-01](history/fix_workspace_fitlab01.md).
- Báo cáo tiến độ 28/09–02/10, [README cũ](history/readme-20261002.md) và spec Kiro về cold-start (`history/kiro_cold_start_spec/`).

Bản Word/PDF/PPTX cũ nằm trong `output/history/`, dựng bằng `build_current_report_docx.py`, `build_current_report_pdf.py`, `build_proposal.py` và `build_advisor_update_pdf.py`.
