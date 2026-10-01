# UFM Rec - Bắt đầu ở đây

## Xem tiến độ

Mở [notebooks/00_status.ipynb](notebooks/00_status.ipynb) rồi Run All, hoặc mở terminal tại thư mục dự án:

```bash
python status.py
```

Lệnh chỉ đọc trạng thái, không khởi động training và không cấp phát GPU.
Mốc mới nhất lấy từ file thật trên máy đang chạy; xem thời gian heartbeat để nhận biết trạng thái cũ.
Ngày 01/10 có thêm `WATCHDOG` và `RECOVERY`: watchdog giữ supervisor chạy trong
container hiện tại, nối tiếp CLIP → UFM → ablation. Xem [hướng dẫn khôi phục](docs/recovery.md).

## Tìm đúng phần cần dùng

| Nơi | Nội dung |
|---|---|
| [docs/next.md](docs/next.md) | Việc cần làm tiếp và điều kiện hoàn tất |
| [reports/Progress_2026_10_01.md](reports/Progress_2026_10_01.md) | Bản tiến trình mới nhất đã xác minh |
| [docs/scope.md](docs/scope.md) | Phạm vi đã chốt theo proposal |
| [docs/ops.md](docs/ops.md) | Queue, log, resume và vận hành |
| [docs/recovery.md](docs/recovery.md) | Watchdog, checkpoint và cách khởi động lại sau khi FITLAB tạo container mới |
| [docs/train.md](docs/train.md) | Huấn luyện UFM |
| [docs/foundation.md](docs/foundation.md) | Trích CLIP text/image |
| notebooks/01_setup.ipynb | Cài đặt/preflight; đọc kỹ trước khi chạy |
| notebooks/02_baseline.ipynb | Theo dõi đối chứng Transformer |
| notebooks/03_clip.ipynb | Theo dõi chi tiết CLIP |
| src/ | Mã pipeline và mô hình |
| runs/ | Log, checkpoint và trạng thái thật |
| data/ | Dữ liệu, mapping và features |
| archive/ | Gói chuyển file cũ, bản sao tài liệu và bản sao trước sắp xếp |

Đợt sắp xếp chỉ đổi tên tài liệu/notebook và gom file phụ. Đường dẫn trong src/, data/, runs/
được giữ để queue, checkpoint và kiểm tra fingerprint tiếp tục dùng cùng cấu hình.
Trên FITLAB, tmp/, .venv_ufm và cache được ẩn khỏi Explorer; chúng vẫn tồn tại và hoạt động.

## Demo và GitHub

Demo TF-IDF trên máy local: mở Start_Demo.cmd trong bản checkout đã có dữ liệu.
FITLAB đang ưu tiên extraction/training; không tự mở thêm một job GPU để trình bày demo.
Repo: https://github.com/anky06-ky/UFM-Rec

## Khi cần hoàn tác tên file

Danh sách đổi tên nằm trong archive/layout.json; bản nội dung trước thay đổi nằm trong
archive/layout-backup.zip. Có thể đổi ngược từng cặp đường dẫn sau khi kiểm tra không ghi đè
file mới. Không giải nén ghi đè src/, data/ hoặc runs/.
