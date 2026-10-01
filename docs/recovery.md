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

FITLAB hiện không có cron hoặc user systemd đang hoạt động. Watchdog tự khôi phục
process trong container đang sống, nhưng không thể tự chạy khi cả container đã
bị trường thu hồi. Khi đó cần mở FITLAB và chạy lệnh khởi động ở trên; cursor
và checkpoint trên iDragonCloud vẫn còn để resume.
