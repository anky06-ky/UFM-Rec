# UFM-Rec

<!-- current-evidence-20261028 -->
> Cập nhật 28/10/2026: validation v3 đã hoàn tất (40.000 ca, 100 ứng viên/ca, seed 42). Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only 0,190036; UFM full 0,182056; popularity 0,042527**. Tám checkpoint UFM dùng cùng candidate hash. Đây là đánh giá lại checkpoint chọn bằng validation v1, chưa phải holdout độc lập. Test v2 đã được xem; vòng v3 này không đánh giá test. Chi tiết và giới hạn: [bằng chứng mới](reports/continuation_20261028/README.md). Các số liệu ở phần lịch sử phải đọc theo phiên bản giao thức ghi kèm. Word/PDF cũ chưa đồng bộ cập nhật này.
<!-- /current-evidence-20261028 -->

> Phạm vi giảng viên đã chốt theo xác nhận của nhóm: **hệ gợi ý sản phẩm kết hợp hành vi, hình ảnh và tên/mô tả**. Cold-start sản phẩm là phân tích trọng tâm. Phân tích [v3 theo train count](reports/continuation_20261028/count_effect_v3.json) cho thấy full thấp hơn semantic-only 0,024101 NDCG@10 ở nhóm 1–5 tương tác và cao hơn 0,014865 ở nhóm 21+; một seed chưa đủ kết luận cải tiến. [Giao thức 15 run](configs/multimodal_followup_20261028.json) đã khóa, chưa có kết quả huấn luyện mới.


Hệ khuyến nghị sản phẩm cold-start kết hợp CLIP text/image, lịch sử tương tác
và độ tin cậy học được. Amazon Reviews 2023 - Toys & Games.

## Mở nhanh

| Tài liệu | Nội dung |
| --- | --- |
| [Báo cáo tổng kết](docs/final_report.md) · [Slide thuyết trình](docs/presentation_slides.md) | Báo cáo 6 chương (kèm mục giới hạn thực nghiệm) và 14 slide bảo vệ đồ án |
| [Proposal PDF](output/pdf/proposal.pdf) | Bản 2.0, 13 trang, 5 hình: kiến trúc, phương trình, thực nghiệm, tiến độ và tham khảo |
| [Markdown](docs/proposal.md) · [HTML](docs/proposal.html) | Bản chỉnh sửa và bản xem trình duyệt |
| [Tiến độ hiện tại](reports/progress.md) | Việc vừa hoàn tất và phần còn thiếu |
| [Bắt đầu](START.md) · [Demo](docs/demo.md) | Cách chạy và proxy (hỗ trợ tự sinh mock catalog offline) |
| [Khôi phục](docs/recovery.md) · [Báo cáo](reports/README.md) | Vận hành và bằng chứng lịch sử |

Bản dùng trao đổi hiện tại: [nội dung với giảng viên](docs/trao_doi_gvhd_2026_10_28.md), [báo cáo nguồn](docs/final_report.md) và [slide nguồn](docs/presentation_slides.md). File Word/PDF cũ chưa được xuất lại từ nội dung sửa trong lần này; chưa dùng các bản đó làm bản đã đồng bộ.

Việc cần làm và điều kiện nghiệm thu từng mục: [checklist hoàn thiện đồ án](docs/checklist_hoan_thien_2026_10_27.md).

Hướng triển khai tiếp theo theo thứ tự ưu tiên: [docs/next.md](docs/next.md). Thư mục đã dọn cache và tệp thao tác một lần; [nhật ký và cách khôi phục](reports/workspace_cleanup_20261028.md).

Bàn giao ngày 28/10: [kết quả validation v3 và nghiệm thu UFM thật](reports/continuation_20261028/README.md), [bản sao dữ liệu đã khôi phục đủ](reports/continuation_20261028/backup_handoff.md), [lời trao đổi mới với giảng viên](docs/trao_doi_gvhd_2026_10_28.md). Bản sao nằm ngoài repo tại `D:\Nam3_ki1\UFM-Rec_Backups`; có launcher `Start_UFM_From_Backup.cmd` dùng dữ liệu đã phục hồi.

[PDF tiến độ trước buổi gặp](output/pdf/UFM_Rec_Tien_Do_2026_10_28.pdf) được tạo từ JSON bằng chứng và đã kiểm tra cả hai trang. Nhánh [`followup/multimodal-20261028`](https://github.com/anky06-ky/UFM-Rec/tree/followup/multimodal-20261028) có [CI Windows/Ubuntu đạt](https://github.com/anky06-ky/UFM-Rec/actions/runs/40693524511). Báo cáo Word/PDF toàn văn cũ chưa đồng bộ và 15 lượt train mới chưa chạy.

## Kết quả lịch sử đã đo

- CLIP hoàn tất **767.045/767.045** item; **758.692** ảnh OK.
- UFM full (`seed=42`) hoàn tất **12 epoch**, best epoch **9**, tổng 198.232 bước.
- Validation NDCG@10 overall **0,285611**, cold macro **0,163738**.
  TF-IDF tương ứng **0,260367 / 0,191270**: UFM chưa vượt mục tiêu cold-start.
- Đây là kết quả của các lượt chạy lịch sử; chúng không chứng minh rằng các queue
  hoặc checkpoint còn nằm trên FITLAB hiện tại.
- Demo **UFM** vượt qua 4/4 ca API thật và giao diện zero-shot qua proxy.
- Audit validation: cold macro thấp hơn TF-IDF 0,027532; calibration thăm dò giảm
  ECE 0,164477 → 0,041497. [Số liệu](reports/ufm_validation_audit_v2/report.md).
- Audit cũ ghi `test_evaluated = false`; artifact `evaluation_v2` tìm thấy ngày 27/10 đã có UFM validation và test. Xem mục cập nhật bên dưới. Baseline cổ điển đã có test trong lịch sử.
- Báo cáo và slide đã được đối chiếu ngày 27/10. Kiểm thử local ngày 28/10 trước khi bổ sung restore từng phần: **91 passed, 3 skipped, 10 subtests passed**. Ba skip dành riêng Linux. GitHub Actions chưa được xác nhận.

## Artifact mới đối chiếu ngày 27/10/2026

- [Bằng chứng evaluation_v2](reports/ufm_evaluation_v2_20261027/README.md): UFM validation Overall/Cold macro NDCG@10 **0,164132 / 0,182051**, test **0,166193 / 0,183638**. Hash mẫu và metadata local khớp report; chưa chạy lại GPU trong lần chỉnh tài liệu.
- V2 dùng bộ ứng viên khác audit lịch sử; đã có baseline cùng validation v2: TF-IDF cold macro 0,195672, popularity 0,042527. Không so điểm v2 với TF-IDF/BPR-MF/SVD cũ hoặc tiếp tục ghi “test chưa mở”.
- Bản ghi FITLAB 24/10 bên dưới là trạng thái tại thời điểm khôi phục, không phải xác nhận realtime; artifact mới không tự xác nhận job/checkpoint còn trên máy chủ.


## Trạng thái workspace FITLAB-02 — cập nhật 24/10/2026

`DA_AI` được mở lại từ archive ngày 24/09; source hiện tại đã được đồng bộ từ
commit `faad9ed`. Dữ liệu Toys/TF-IDF còn nguyên. Môi trường riêng
`/home/coder/.venv-ufm/bin/python` đã xác nhận PyTorch `2.14.1+cu132` và CUDA
trên Quadro RTX 6000. Snapshot 12:28 báo CLIP **0/767.045**, không có queue,
checkpoint hay cache CLIP. `items.csv.gz` và `products_text.csv.gz` còn trên FITLAB,
nhưng catalog URL ảnh và metadata nguồn vẫn thiếu ở đó. Năm shard Parquet chính
thức đã được tải về local và xác minh SHA-256; chưa chuyển chúng lên FITLAB nên
chưa thể dựng catalog hoặc chạy CLIP/training. Chi tiết và các bước tiếp theo ở
[báo cáo tiến độ](reports/progress.md).

Metric trên dùng **40.000 validation cases, 1 positive + 99 negative**;
không phải benchmark full-catalog. Uncertainty là proxy độ tin cậy học được.

## Chạy và tái lập

Mở [Start_Demo.cmd](Start_Demo.cmd) để launcher tự chọn **UFM khi có checkpoint hoàn tất**, hoặc **content** khi chưa có checkpoint. UFM yêu cầu cache và nguồn huấn luyện đúng hash; thiếu dữ liệu sẽ báo đường dẫn cần khôi phục. Có thể chọn rõ `Start_Demo.cmd --backend content` hoặc `Start_Demo.cmd --backend ufm`. Trên máy có dependency trong `requirements.txt`, chế độ content tự tạo catalog mẫu 30 sản phẩm tại `tmp/mock_demo_data` khi thiếu catalog thật. Trên FITLAB, `python3 status.py` đọc heartbeat/PID hiện tại.
Dữ liệu: [data/README.md](data/README.md).

`python scripts/build_proposal.py` tái tạo PDF/HTML/SVG từ docs/proposal.md,
cần ReportLab và Arial (Windows) hoặc DejaVu Sans (Linux).

## Cấu trúc

| Folder | Nội dung |
| --- | --- |
| src / tests / scripts | Mô hình, pipeline, demo API, kiểm thử và vận hành |
| demo / docs | Giao diện, báo cáo cuối, slide, proposal, hướng dẫn; docs/history giữ phiên bản cũ |
| reports / output | Metric nhỏ, tiến độ, PDF và hình |
| data / runs | Dataset, checkpoint, log; không đẩy file lớn lên GitHub |
| archive / tmp | File làm việc cũ và cache local; không theo dõi trong Git |

**Còn chờ:** 15 lượt train theo giao thức đã khóa trên GPU, đánh giá cải tiến tách biệt, benchmark vận hành đủ mẫu, đồng bộ Word/PDF và xác nhận GitHub Actions. Bản sao FITLAB đã được khôi phục và kiểm tra SHA trên máy local; test v2 đã có kết quả, v3 là phân tích validation. Xem [bước tiếp theo](docs/next.md) và [giao thức đã khóa](configs/multimodal_followup_20261028.json).
[Lịch sử README](docs/history/readme-20261002.md) · [Proposal ban đầu](docs/history/proposal-v1.pdf).
