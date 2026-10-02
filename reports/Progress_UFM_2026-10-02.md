# Tiến trình UFM-Rec — 01–02/10/2026

**Mốc mới nhất 02/10/2026, 07:59 UTC+07:** CLIP và UFM full đã hoàn tất.
Best checkpoint là epoch 9, bước 148.674; validation NDCG@10 overall **0,285611**,
cold macro **0,163738**, warm **0,651231**. Bảy ablation đang đợi GPU chia sẻ rảnh;
watchdog và supervisor còn sống. [Báo cáo tóm tắt mới nhất](Tien_do_UFM_2026-10-02.md).

Các mốc dưới đây ghi lại diễn biến trong ngày; số tiến độ cũ là ảnh chụp tại
thời điểm tương ứng.

## Trạng thái đã kiểm tra

Mốc FITLAB khoảng 07:38 UTC+07: CLIP đã commit **594.304/767.045 item
(77,48%)**, trong đó **591.071 ảnh OK**. Cache còn `text.npy`, `image.npy`,
`modalities.npy`, `progress.json`; cursor và cấu hình batch 64 được giữ để resume.
Chưa có `complete.json` full CLIP hay `runs/ufm_full_v1/completed.json`.

Job CLIP trước bị `SIGKILL` lúc 14:57 UTC+07 ngày 30/09. Queue UFM và ablation
đã dừng theo lỗi nguồn. Khi kiểm tra ngày 01/10, không có runner cũ còn sống;
`memory.events` của container báo `oom_kill=0`, không đủ để xác định nguyên nhân.
GPU dùng chung đang bận; ví dụ lần kiểm tra 07:38 utilization 32%, trên ngưỡng
20% của queue.

## Việc đã làm hôm nay

- Kiểm tra cursor, cấu hình, file cache, checkpoint baseline, Python isolated,
  log queue, process và dung lượng iDragonCloud.
- Khởi động supervisor với khóa `flock`; queue foundation đã có heartbeat mới và
  đợi GPU rảnh. GPU bận trở lại đúng lúc chuẩn bị trích, queue dừng an toàn mà
  không đổi cache. Supervisor đã được cập nhật và khởi động lại (PID **611579**
  tại mốc 07:49). Nó chạy CLIP → UFM → 7 ablation theo thứ tự; thử lại tối đa ba
  lượt SIGKILL, còn trường hợp GPU tranh chấp thì đợi lại tối đa 168 giờ.
- Lúc **07:52 UTC+07**, log xác nhận `Frozen CLIP loaded; extracting 767,045
  products from row 594,304 on cuda`. Đây là xác nhận resume từ cache đã lưu;
  số item mới cần đọc từ `progress.json`, chưa thể gọi extraction hoàn tất.
- Mốc **07:59:50 UTC+07**: cursor tăng tới **596.480/767.045 (77,76%)**,
  ảnh OK **593.247**. `status.py` báo supervisor sống và progress mới đang cập nhật.
- `status.py` hiển thị trạng thái recovery, foundation, UFM, ablation và đánh dấu
  heartbeat cũ. Xem `docs/ops.md` để theo dõi và khôi phục khi container đổi.
- Thêm baseline SASRec ID-only với causal attention, negative sampling loại toàn
  bộ positive train của user, cùng candidates validation, checkpoint/resume và
  CPU smoke fixture. Chưa chạy benchmark SASRec trên dữ liệu thật.

## Điều kiện hoàn thành tiếp theo

1. Full CLIP có marker 767.045 item, SHA256 ba bảng, vector hữu hạn và mask đúng.
2. UFM smoke/resume, full checkpoint và validation cold macro hoàn tất trên cùng
   cấu hình; sau đó suite bảy ablation hoàn thành với mỗi biến thể train riêng.
3. Chạy SASRec trên dữ liệu thật; bổ sung BERT4Rec, concat hybrid, dữ liệu thứ hai, nhiều seed,
   đo VRAM/latency, mở test sau khi khóa cấu hình bằng validation.
4. Chuyển checkpoint UFM đã kiểm chứng vào demo và báo cáo kết quả thật.

**Đã có để trình bày:** demo TF-IDF toàn catalog, BPR-MF ba seed, calibration
TF-IDF và báo cáo 30/09. Chưa có kết quả UFM full hoặc kết luận UFM hơn baseline.
Tiến trình GPU phụ thuộc tài nguyên FITLAB đang chia sẻ.

## Cập nhật 08:13 UTC+07

CLIP tiếp tục tăng tới **600.512/767.045 (78,29%)**, ảnh OK **597.278**.
Supervisor cũ PID 611579, queue PID 611580 và worker PID 612215 đều còn sống.
Watchdog PID **618648** đã khởi động trên FITLAB và nhận diện cả ba tiến trình;
không có worker thứ hai. Python isolated được sao lưu vào
`.venv_ufm_runtime` trên iDragonCloud; import Torch CUDA 2.12.1 và
Transformers 4.57.6 đã qua kiểm tra. Mã supervisor mới sẽ được watchdog dùng
khi cần khởi động lại: retry không giới hạn các lỗi ngắt worker/GPU bận/hết
cửa sổ chờ, với backoff tối đa 30 phút; lỗi validation/checksum vẫn dừng để
người vận hành xử lý. Watchdog cũng sửa trạng thái bàn giao nếu marker CLIP đã
hoàn tất nhưng queue bị ngắt trước khi ghi `extraction_complete`.

Kiểm thử local: **40/40 đạt**; thêm **3/3 kiểm thử khôi phục** đạt trên Linux FITLAB.
FITLAB không có cron hay user systemd đang hoạt động. Task code-server đã được
cài để gọi `bash scripts/start_campaign.sh` khi mở lại workspace; checkpoint và
venv trên iDragonCloud vẫn còn. Nếu container chưa được mở lại thì không có
process nào có thể chạy.
Hướng dẫn chi tiết: [docs/recovery.md](../docs/recovery.md).

## Cập nhật 21:40 UTC+07

CLIP đã hoàn tất **767.045/767.045 sản phẩm (100%)**, trong đó **758.692 ảnh OK**.
Marker `complete.json` đã được queue UFM kiểm tra về checksum, mapping, vector
hữu hạn và mask. GPU smoke/resume UFM đã qua **5 bước**. Demo TF-IDF hiện có vẫn
chạy CPU; demo UFM cần đợi checkpoint full đã kiểm chứng.

Full UFM đã dừng ở epoch 1 do gradient AMP tràn sau bước **6.000/16.527** dù
loss còn hữu hạn. Checkpoint bước 6.000 có trọng số hữu hạn. Trainer đã sửa để
GradScaler hạ scale và bỏ qua batch bị tràn. Checkpoint và config được chuyển
đúng SHA mã mới, lưu bản gốc trong `runs/ufm_full_v1/` trên FITLAB. Kiểm thử
**6/6 UFM** (gồm ca gradient tràn) và **5/5 campaign** đạt trên FITLAB.

Lúc 21:43, log xác nhận `RESUME PASS` ở bước 6.000. AMP lại tràn tại batch 6.072;
GradScaler giảm scale từ 524.288 xuống 262.144 và trainer tiếp tục tới **bước
7.600**, VRAM peak khoảng 3,70 GiB. Watchdog và supervisor vẫn sống.
Full checkpoint và bảy ablation **chưa hoàn tất**. Queue sẽ chạy ablation sau
full UFM; theo dõi `python status.py` và log trong `runs/`. Mã đã push lên
GitHub commit `8e30895` và các cập nhật tài liệu sau đó.

## Cập nhật 21:56 UTC+07

Epoch 1 đã được kiểm chứng bằng `runs/ufm_full_v1/epoch_001.json` và
`runs/ufm_full_v1/best.pt` trên FITLAB. Cold macro NDCG@10 validation là
**0,14961466666666667**, tại bước **16.521**. Mã xử lý AMP đã đi qua các batch
gradient tràn, tự hạ scale, bỏ optimizer step không an toàn và tiếp tục train.

Lúc 21:56:32, log cho thấy epoch 2 ở batch **3.180/16.527**, bước **19.700**,
VRAM peak khoảng **3,70 GiB**. `status.py` xác nhận watchdog và supervisor
còn sống, UFM đang chạy, ablation xếp hàng; production `completed.json` chưa có.
Demo TF-IDF đã có kiểm thử nghiệm thu bốn chế độ trong
`reports/demo_acceptance_2026_09_30.json`; demo UFM chờ production checkpoint.

## Cập nhật 22:01 UTC+07

Kiểm tra trực tiếp trên FITLAB xác nhận CLIP vẫn ở mức **767.045/767.045**;
UFM còn chạy ở epoch 2, batch **8.083/16.527**, bước **24.600**. Watchdog và
supervisor còn sống, bảy ablation vẫn đợi UFM full. AMP overflow tại batch 8.036
được xử lý bằng cách hạ scale và trainer tiếp tục. Production checkpoint chưa
hoàn tất.

Giao diện `demo/index.html` được làm mới với hero xanh rừng, thẻ sản phẩm có
hover, bố cục thích ứng cho màn nhỏ và nhãn hiển thị rõ điểm xếp hạng. Đã mở
demo backend TF-IDF trên catalog thật; luồng LEGO → một sản phẩm lịch sử → lọc
zero-shot → Top 5 trả đủ kết quả. Đây vẫn là demo baseline, chưa dùng UFM.
Tóm tắt dành cho người đọc: `reports/Tien_do_UFM_2026-10-02.md`.

## Cập nhật 22:11 UTC+07

Epoch 2 hoàn tất tại bước **33.040**. Trên validation cố định 40.000 mẫu với
1 positive + 99 negative, UFM đạt overall NDCG@10 **0,274583**, cold macro
**0,150295**, warm **0,647448**, zero-shot **0,174892**. `best.pt` hiện là
checkpoint epoch 2. Cùng protocol, TF-IDF content validation có overall
**0,260367**, cold macro **0,191270**, warm **0,467661**: UFM epoch 2 cao hơn ở
overall/warm nhưng thấp hơn ở cold macro. Đây chưa phải test hay kết luận cuối.

Trainer đã tiếp tục epoch 3 và ở bước **39.000** lúc 22:17:30. AMP overflow tại
bước 33.002 được xử lý, giảm scale; epoch 2 vẫn hoàn tất và ghi validation.
Watchdog/supervisor còn sống; bảy ablation chờ UFM full.

## Cập nhật 22:30 UTC+07

FITLAB `status.py` lúc 22:30:28 xác nhận CLIP **767.045/767.045**, 758.692 ảnh
OK; UFM epoch 4 ở batch **1.441/16.527**, bước **51.000**, VRAM **3,70 GiB**.
Watchdog còn sống, recovery supervisor ở stage `ufm_running`; full
`completed.json` chưa có. Epoch 2 vẫn là checkpoint tốt nhất hiện ghi nhận.

Suite ablation còn trạng thái lỗi cũ từ 30/09, do pipeline CLIP tiền nhiệm bị
SIGKILL. Recovery supervisor hiện chạy queue theo thứ tự CLIP → UFM → ablation;
khi full marker hợp lệ xuất hiện, suite sẽ được gọi lại và bỏ qua trạng thái cũ
vì parent đã hoàn tất. Không khởi chạy suite thứ hai song song.

**Việc ngay tiếp theo:** để full UFM tiếp tục; theo dõi `python3 status.py` và
`runs/ufm_training_queue_v1.log`. Sau đó xác nhận production checkpoint,
validation full rồi kiểm tra ablation tự vào trạng thái `ablation_training`.

## Cập nhật 02/10/2026, 07:59 UTC+07

FITLAB `status.py` xác nhận CLIP hoàn tất **767.045/767.045** (758.692 ảnh OK) và
UFM full đã hoàn tất **12 epoch, 198.232 bước**. Best checkpoint là epoch 9 tại
bước **148.674**; ba epoch sau không cải thiện. Trên validation sampled ranking
40.000 trường hợp (1 positive + 99 negative), best NDCG@10 là overall **0,285611**,
cold macro **0,163738**, zero-shot **0,188403**, extreme-cold **0,113300**, cold
**0,189511**, warm **0,651231**.

Cùng protocol, TF-IDF có overall **0,260367**, cold macro **0,191270** và warm
**0,467661**. UFM cao hơn overall/warm nhưng thấp hơn cold macro; chưa đánh giá test
và chưa thể tuyên bố UFM vượt baseline cho mục tiêu cold-start.

Recovery supervisor và watchdog còn sống. Bảy ablation ở trạng thái
`waiting_for_full_ufm_and_idle_gpu`; GPU chia sẻ được ghi nhận **100% utilization**,
còn **8.352 MiB**. Chưa có tiến trình GPU riêng của campaign tại thời điểm snapshot;
suite đang chờ tài nguyên, không cần chạy tay.

Demo UFM đã nạp trên FITLAB CPU với 767.045 sản phẩm. Code-server forward cổng 8766
dưới `/proxy/8766/`; bản cũ trả `Loopback origin required` do chưa hiểu origin/path
reverse proxy. Đang cập nhật route để hỗ trợ origin HTTPS khớp chính xác và prefix
này, sau đó kiểm thử tìm kiếm, zero-shot, Top 5 và xuất JSON.

**Bước kế tiếp:** để supervisor tự chạy bảy ablation khi GPU chia sẻ rảnh; nghiệm
thu demo UFM qua proxy; rồi benchmark SASRec, BERT4Rec và hybrid trên cùng protocol.
Chỉ khóa cấu hình trên validation trước khi chạy một lượt test có kiểm soát.
