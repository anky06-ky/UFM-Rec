# Sao lưu FITLAB về máy và khôi phục

Thư mục lưu bản sao trên máy: `D:\Nam3_ki1\UFM-Rec_Backups\fitlab01_20261028`.
Thư mục này nằm ngoài checkout để việc dọn repo không vô tình xóa bản dự phòng.

## Khi nào được coi là đã sao lưu xong?

Phải có `manifest.json` với `phase=complete`, đủ tất cả `bundle_*.zip` khớp
SHA-256, và `restore_receipt.json` xác nhận `all_files_sha256_verified=true`.
Inventory, manifest đang đóng gói, hoặc chỉ có gói đầu tiên chưa chứng minh dữ
liệu đã an toàn. Xem `download_receipt.json` để biết gói nào còn thiếu.

Gói sao lưu giữ dataset đã chia theo thời gian, toàn bộ catalog sản phẩm,
TF-IDF, CLIP features/masks, checkpoint, predictions, config, log thực nghiệm,
mã nguồn đã huấn luyện, demo và tài liệu. CLIP image embeddings là đặc trưng
ảnh; chúng không phải tập JPEG gốc. Các ảnh demo đã lưu trong repo có thể mở
offline; URL ảnh catalog thật vẫn cần mạng nếu chưa có ảnh local.

Không đưa virtualenv, Git internals, khóa đăng nhập hoặc cache tạm vào gói.
Dữ liệu thô Parquet đang có ở `data/raw/` trên máy cũng cần được giữ riêng.

## Thu hồi tệp trình duyệt đã tải

Nếu trình duyệt để tên ngẫu nhiên `.tmp` ở ổ D, công cụ sau chỉ di chuyển tệp
có kích thước và SHA-256 trùng với một bundle đã biết. Tệp khác không bị đụng tới.

```powershell
python scripts/collect_backup_downloads.py --backup D:\Nam3_ki1\UFM-Rec_Backups\fitlab01_20261028 --inbox D:\ D:\Nam3_ki1\UFM-Rec
```

## Khôi phục kiểm tra, không ghi đè checkout

```powershell
python scripts/artifact_backup.py restore --backup D:\Nam3_ki1\UFM-Rec_Backups\fitlab01_20261028 --output D:\Nam3_ki1\UFM-Rec_Backups\fitlab01_20261028\restored
```

Đích phải mới hoặc rỗng. Công cụ kiểm tra SHA archive, đường dẫn và từng tệp
sau giải nén, sau đó mới ghi receipt. Chỉ đưa `data/` và `runs/` đã kiểm tra vào
checkout; giữ mã hiện tại và mã tạo checkpoint thành hai bản có nguồn gốc rõ.

Khi chỉ tải được một phần, thêm `--available-only` và chọn thư mục đích mới.
`partial_restore_receipt.json` liệt kê tệp/gói còn thiếu; cờ
`all_files_sha256_verified=false` xác định đây là bản khôi phục chưa đầy đủ.
Tệp khôi phục có SHA được kiểm tra; tệp trong gói lỗi hash sẽ bị từ chối.
Các tên Linux chỉ khác hoa/thường được giữ riêng dưới `.restore_case_conflicts/`
và ghi đường dẫn gốc/đích trong receipt để tránh mất tệp trên Windows.

Có thể giữ dữ liệu ngoài repo, rồi truyền `--data`, `--run`, `--features` và
`--training-source` vào launcher bằng đường dẫn trong thư mục đã khôi phục.
Không cần chép thêm một bản dataset lớn vào checkout.

## Chạy UFM sau khi phục hồi

```powershell
.\Start_Demo.cmd --backend ufm --port 8766
```

Launcher kiểm tra checkpoint/cache và dùng snapshot huấn luyện đã ghi hash.
Nếu thiếu tệp hoặc hash sai, cần khôi phục đúng tệp; không bỏ kiểm tra để ép nạp.
`--backend content` mở baseline; backend và nguồn dữ liệu được ghi trong API.

Việc sao lưu chỉ hoàn tất khi có receipt thực tế; tài liệu này không tự xác nhận
rằng mọi bundle đã tải và khôi phục xong.
