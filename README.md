# UFM-Rec

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

## Trạng thái hiện tại

- CLIP hoàn tất **767.045/767.045** item; **758.692** ảnh OK.
- UFM full (`seed=42`) hoàn tất **12 epoch**, best epoch **9**, tổng 198.232 bước.
- Validation NDCG@10 overall **0,285611**, cold macro **0,163738**.
  TF-IDF tương ứng **0,260367 / 0,191270**: UFM chưa vượt mục tiêu cold-start.
- Đã phục hồi ablation sau SIGTERM; hàng đợi GPU trên FITLAB đang chạy theo thứ tự.
- Đã triển khai queue benchmark SASRec/BERT4Rec thích nghi/concat và UFM ba seed.
  All Beauty đã chuẩn bị CPU; queue CLIP/benchmark đợi sau Toys.
- Demo **UFM** vượt qua 4/4 ca API thật và giao diện zero-shot qua proxy.
- Audit validation: cold macro thấp hơn TF-IDF 0,027532; calibration thăm dò giảm
  ECE 0,164477 → 0,041497. [Số liệu](reports/ufm_validation_audit_v2/report.md).
- UFM chưa đánh giá test (`test_evaluated = false`). Baseline cổ điển đã có test trong lịch sử.
- Đã hoàn thiện bản báo cáo tổng kết ([docs/final_report.md](docs/final_report.md)), slide bảo vệ ([docs/presentation_slides.md](docs/presentation_slides.md)), script tạo mock catalog cho demo local ([scripts/build_mock_demo_data.py](scripts/build_mock_demo_data.py)) và kiểm thử **14/14 test suites PASS** (53 passed, 3 skipped trên Windows).

Metric trên dùng **40.000 validation cases, 1 positive + 99 negative**;
không phải benchmark full-catalog. Uncertainty là proxy độ tin cậy học được.

## Chạy và tái lập

Mở [Start_Demo.cmd](Start_Demo.cmd) trên máy có dependency trong `requirements.txt` (nếu chưa có dữ liệu 10GB, hệ thống tự động tạo bộ catalog mẫu 30 sản phẩm vào `tmp/mock_demo_data`). Trên FITLAB, `python3 status.py` đọc heartbeat/PID hiện tại.
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

**Còn chờ thu hoạch từ FITLAB:** kết quả GPU của 7 ablation, benchmark/nhiều seed và All Beauty;
đo tài nguyên lặp 30 chu kỳ trên full catalog; khóa manifest và mở đánh giá test cuối một đợt duy nhất.
Xem [bước tiếp theo](docs/next.md) và [protocol đã triển khai](docs/followup.md).
[Lịch sử README](docs/history/readme-20261002.md) · [Proposal ban đầu](docs/history/proposal-v1.pdf).
