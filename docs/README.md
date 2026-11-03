# Đọc và chỉnh sửa tài liệu UFM Rec

Đọc [hồ sơ hoàn thiện 03/11](../reports/final_20261103/README.md) trước. Nhóm chốt đồ án với kết quả hiện có; vòng 15 lượt đa seed là hướng phát triển.

## Bộ tài liệu hiện hành

- [Báo cáo nguồn](final_report.md) → [Word theo mẫu trường](../output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx), [PDF](../output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf); dựng bằng `scripts/build_school_report_docx.py`.
- [Nguồn 14 slide](presentation_slides.md) → [PPTX bảo vệ](../output/presentation/UFM_Rec_Bao_Ve_20261103.pptx).
- [Kịch bản bảo vệ và câu hỏi](defense_guide.md), [hướng dẫn demo](demo.md).
- [Giao thức](protocol_current.md), [tin nhắn đề xuất metric/holdout gửi giảng viên ngày 30/10](trao_doi_gvhd_2026_10_30.md).

Chỉ sửa số liệu báo cáo/slide theo cùng bảng bằng chứng và protocol. Thông tin bìa (GVHD, thành viên, MSSV, nhiệm vụ) nằm trong `ADVISOR` và `TEAM` của script dựng báo cáo; slide 1 và `presentation_slides.md` phải sửa cùng lúc.

## Vận hành khi cần

- [Colab](colab.md), [runbook GPU](gpu_followup_runbook.md), [kế hoạch tiếp theo](next.md).
- [Backup/restore](backup_restore.md), [khôi phục môi trường](recovery.md), [vận hành](ops.md).
- [Nền tảng đặc trưng](foundation.md), [huấn luyện](train.md), [GPU](gpu.md), [phạm vi](scope.md).

## Tài liệu tham khảo và lịch sử

`history/` giữ các báo cáo có ngày, README cũ, proposal v1, tin nhắn GVHD 27–28/10, checklist 27/10, ghi chú sửa FITLAB-01 và spec Kiro về cold-start. `proposal.md`, `proposal.html`, `overview.html` là tài liệu thiết kế/giới thiệu cũ; không dùng số trong đó thay validation v3.

Không sửa trực tiếp PDF/DOCX để cập nhật metric. Dựng lại từ nguồn: `build_school_report_docx.py` tạo bộ nộp hiện hành; `build_current_report_docx.py`, `build_current_report_pdf.py`, `build_proposal.py` và `build_advisor_update_pdf.py` dựng các bản lịch sử vào `output/history/`.
