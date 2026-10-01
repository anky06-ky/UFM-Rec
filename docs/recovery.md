# Chạy liên tục và khôi phục chiến dịch FITLAB

Ngày 01/10/2026, CLIP đang chạy từ checkpoint. `scripts/watch_campaign.py` theo dõi
supervisor, queue và worker trong container hiện tại. Nếu supervisor hoặc worker
bị SIGKILL, watchdog chờ mọi process cũ kết thúc rồi chạy lại supervisor. Queue
chỉ dùng `--resume` khi có progress/checkpoint chưa hoàn tất. Các stage chạy tuần tự:
CLIP, UFM full, sau đó bảy ablation. Các lỗi checksum, cấu hình và invariant vẫn
dừng và hiện `needs_attention` để tránh lặp một thử nghiệm sai.

```bash
python status.py
cat runs/campaign_watchdog_v1.json
tail -n 30 runs/campaign_watchdog_v1.log
tail -n 30 runs/campaign_recovery_20261001.log
```

Python của campaign nằm tại `.venv_ufm_runtime/bin/python` trên iDragonCloud.
Watchdog dùng `/usr/bin/python3`, nên không phụ thuộc venv tạm trong `/tmp`.
Nếu container FITLAB được tạo lại, vào thư mục dự án và chạy:

```bash
bash scripts/start_campaign.sh
python status.py
```

Không chạy thêm queue hay trainer thủ công khi watchdog còn hoạt động. File lock
và kiểm tra process ngăn hai lượt GPU cùng chạy. Cache, checkpoint, status và log
được giữ dưới `data/` và `runs/` trên iDragonCloud; không đưa chúng lên GitHub.

## Tự khởi động khi mở lại workspace

Chạy một lần trên FITLAB:

```bash
python3 scripts/install_autostart.py
```

Installer giữ các settings/task hiện có, lưu bản sao trong `runs/autostart_backups/`,
thêm task `UFM: keep campaign running`. Code-server hiện yêu cầu lựa chọn Allow
Automatic Tasks cho tất cả workspace đã tin cậy. Người dùng đã đồng ý bật tùy chọn
này trên FITLAB ngày 01/10/2026. Khi cài mới cần chọn Allow trong hộp thoại.
Khi code-server mở lại thư mục dự án, task gọi `start_campaign.sh`. Khóa watchdog
ngăn việc mở lại tab tạo thêm GPU job. Cơ chế `runOn: folderOpen` được mô tả trong
[tài liệu VS Code](https://code.visualstudio.com/docs/debugtest/tasks#_run-behavior).

Lúc 20:36 ngày 01/10, worker CLIP bị SIGKILL ở cursor 764.032. Supervisor đã lên
lịch retry sau 120 giây; heartbeat watchdog còn tới 20:37:38. PID 1 của container
mới bắt đầu lúc 20:38:14, trước khi retry diễn ra. Toàn bộ process cũ mất theo
container. Cache và venv trên iDragonCloud còn nguyên, đã resume thành công.

FITLAB không có cron hoặc user systemd đang hoạt động. Task tự bật khi workspace
được mở/kết nối lại; nếu container bị thu hồi và workspace chưa được mở thì chưa
có process nào chạy. Có thể dùng lệnh khởi động thủ công ở trên. Watchdog mới ghi
thêm định danh container và bộ đếm memory/oom vào log để chẩn đoán lần ngắt sau.

## UFM dừng do gradient AMP ngày 01/10

CLIP đã hoàn tất 767.045 item. Full UFM dừng sau bước 6.000 vì gradient không
hữu hạn trong khi AMP scale là 524.288; checkpoint bước 6.000 có trọng số hữu hạn.
Trainer hiện để GradScaler giảm scale và bỏ qua batch bị tràn. Trên FITLAB đã
chạy `scripts/migrate_ufm_amp_checkpoint.py` một lần để cập nhật SHA mã trong
checkpoint/config, giữ bản gốc `latest.pre_amp_fix.pt` và
`config.pre_amp_fix.json`. Watchdog được mở lại sau khi ghi `repair_applied` vào
recovery status. UFM queue đã gọi trainer với `--resume` và log xác nhận bước
6.000. Lần tràn tiếp theo ở batch 6.072 được xử lý: scale giảm, training vẫn
tiến tới bước 7.600. Kiểm tra log và `python status.py` để theo dõi tiếp.
