# UFM-Rec

Hệ khuyến nghị sản phẩm cold-start kết hợp CLIP text/image, lịch sử tương tác
và độ tin cậy học được. Amazon Reviews 2023 - Toys & Games.

## Mở nhanh

| Tài liệu | Nội dung |
| --- | --- |
| [Proposal PDF](output/pdf/proposal.pdf) | Bản 2.0, 13 trang, 5 hình: kiến trúc, phương trình, thực nghiệm, tiến độ và tham khảo |
| [Markdown](docs/proposal.md) · [HTML](docs/proposal.html) | Bản chỉnh sửa và bản xem trình duyệt |
| [Tiến độ hiện tại](reports/progress.md) | Việc vừa hoàn tất và phần còn thiếu |
| [Bắt đầu](START.md) · [Demo](docs/demo.md) | Cách chạy và proxy |
| [Khôi phục](docs/recovery.md) · [Báo cáo](reports/README.md) | Vận hành và bằng chứng lịch sử |

## Trạng thái 02/10/2026

- CLIP hoàn tất **767.045/767.045** item; **758.692** ảnh OK.
- UFM full hoàn tất **12 epoch**, best epoch **9**, tổng 198.232 bước.
- Validation NDCG@10 overall **0,285611**, cold macro **0,163738**.
  TF-IDF tương ứng **0,260367 / 0,191270**: UFM chưa vượt mục tiêu cold-start.
- FITLAB đọc lại được mount sau EIO; watchdog/supervisor đã phục hồi.
  Ablation còn chờ GPU chia sẻ (18:23 UTC+07: 100% utilization, trống 1.018 MiB).
- Demo TF-IDF đã chạy; inference UFM qua proxy chưa nghiệm thu.
- UFM chưa đánh giá test. Baseline cổ điển đã có test trong lịch sử.

Metric trên dùng **40.000 validation cases, 1 positive + 99 negative**;
không phải benchmark full-catalog. Uncertainty là proxy độ tin cậy học được.

## Chạy và tái lập

Mở [Start_Demo.cmd](Start_Demo.cmd) trên máy có dữ liệu và dependency trong
requirements.txt. Trên FITLAB, `python3 status.py` đọc heartbeat/PID hiện tại.
Dữ liệu: [data/README.md](data/README.md).

`python scripts/build_proposal.py` tái tạo PDF/HTML/SVG từ docs/proposal.md,
cần ReportLab và Arial (Windows) hoặc DejaVu Sans (Linux).

## Cấu trúc

| Folder | Nội dung |
| --- | --- |
| src / tests / scripts | Mô hình, pipeline, demo API, kiểm thử và vận hành |
| demo / docs | Giao diện, proposal, hướng dẫn; docs/history giữ phiên bản cũ |
| reports / output | Metric nhỏ, tiến độ, PDF và hình |
| data / runs | Dataset, checkpoint, log; không đẩy file lớn lên GitHub |
| archive / tmp | File làm việc cũ và cache local; không theo dõi trong Git |

**Còn thiếu:** 7 ablation; SASRec/BERT4Rec/concat benchmark; nhiều seed; calibration UFM;
dataset thứ hai; tài nguyên; test cuối; demo UFM và slide. Xem [bước tiếp theo](docs/next.md).
[Lịch sử README](docs/history/readme-20261002.md) · [Proposal ban đầu](docs/history/proposal-v1.pdf).
