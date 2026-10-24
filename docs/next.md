# Bước tiếp theo — 24/10/2026 (cập nhật 12:28 UTC+07)

Source commit `faad9ed` và môi trường CUDA đã sẵn sàng trên FITLAB-02. Snapshot
12:28 báo `status.py` CLIP 0/767.045, không có queue/checkpoint. `items.csv.gz` và
`products_text.csv.gz` còn trên FITLAB, nhưng metadata nguồn và catalog URL ảnh chưa
có. Năm shard Parquet đã được tải về local và kiểm tra SHA-256; chi tiết trong
`data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/download_manifest.json`. Xem
[tiến độ có bằng chứng](../reports/progress.md) và [protocol hàng đợi](followup.md).

1. Chuyển 5 shard đã xác minh từ local vào
   `data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/` trên FITLAB, đồng thời
   đồng bộ `src/prepare_ufm_catalog.py` và `requirements-catalog-parquet.txt` đã cập
   nhật. Tệp nguồn tổng cộng **972.667.016 byte**; manifest ghi revision và SHA-256.
   Các file Parquet lớn không đưa vào Git.
2. Trong venv `/home/coder/.venv-ufm`, cài `pyarrow` theo
   `requirements-catalog-parquet.txt`. Trên FITLAB, `items.csv.gz` 7,2 MiB và
   `products_text.csv.gz` 283 MiB đã tồn tại. Gọi builder với 5 shard theo thứ tự
   (`full-00000` đến `full-00004`) và output mới
   `data/processed/toys_games_full_temporal/foundation_catalog_restore_20261024`.
   Không dùng report cũ thay catalog.

   ```bash
   /home/coder/.venv-ufm/bin/pip install -r requirements-catalog-parquet.txt
   /home/coder/.venv-ufm/bin/python src/prepare_ufm_catalog.py \
     --data data/processed/toys_games_full_temporal \
     --metadata data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/full-00000-of-00005.parquet \
                data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/full-00001-of-00005.parquet \
                data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/full-00002-of-00005.parquet \
                data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/full-00003-of-00005.parquet \
                data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/full-00004-of-00005.parquet \
     --output data/processed/toys_games_full_temporal/foundation_catalog_restore_20261024
   ```

3. Đối chiếu catalog với đúng **767.045 `parent_asin`** và mapping theo thứ tự hiện
   tại; dừng nếu thiếu/trùng ID hoặc checksum nguồn không khớp. Checkout local còn
   report cũ với catalog SHA256 dự kiến
   `65d63e7b58babb133db78f3369b3224db6a451d80c4f616a4b8c0023f73b9003`; dùng nó
   để đối chiếu sau khi tái tạo, không dùng report thay file catalog.
4. Môi trường đã kiểm tra: `/home/coder/.venv-ufm/bin/python`; PyTorch
   `2.14.1+cu132`, CUDA khả dụng. Chạy CLIP smoke nhỏ sau khi có catalog, kiểm tra
   hash/shape/norm/mask, rồi mới trích xuất đủ item và xác nhận marker hoàn tất.
5. Không resume: không có checkpoint. Sau khi cache hợp lệ, chạy UFM/ablation từ
   đầu qua queue hiện có; trước mỗi đợt xác nhận không có trainer khác và GPU đủ
   ngưỡng protocol. Không chạy đồng thời nhiều queue.
6. Chọn checkpoint/calibration bằng validation, khóa config/hash/candidates trong
   manifest, rồi chạy một đợt test. Ghi rõ kết quả benchmark lịch sử không phải kết quả
   của workspace hiện tại.
7. Đo VRAM/RAM, latency p50/p95 và throughput; cập nhật báo cáo/slide từ số đo mới.
   UFM chưa có test metric mới; cold macro lịch sử thấp hơn TF-IDF khoảng 0,027532.

Hiện các shard đã sẵn sàng ở local nhưng chưa được chuyển lên FITLAB; không coi
catalog hay CLIP là đã chạy. Xem trạng thái từng bước trong
[báo cáo tiến độ](../reports/progress.md).

Các queue/snapshot ngày 03/10 và ghi chép CLIP/UFM ngày 28/09 chỉ là lịch sử. Dùng
[hướng dẫn phục hồi](recovery.md) và [protocol queue](followup.md) khi đã có đủ
source/checkpoint; không kết luận hoàn tất chỉ từ heartbeat.
