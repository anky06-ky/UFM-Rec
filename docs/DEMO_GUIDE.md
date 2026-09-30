# Chạy và trình bày demo UFM Rec

## Mở demo

Trên máy hiện tại, mở `Start_Demo.cmd` trong thư mục dự án. Launcher mở lại server
đang có hoặc tải catalog và chạy server mới. Lần tải đầu có thể mất khoảng một phút.
Trình duyệt mở tại http://127.0.0.1:8765. Giữ cửa sổ server khi đang trình bày;
Ctrl+C trong cửa sổ đó để dừng. Server chỉ nghe trên máy local.

Nếu cần chạy thủ công:

```powershell
tmp/ufm_checks_env/Scripts/python.exe src/demo_recommender.py --backend content --port 8765 --open-browser
```

Máy khác cần Python 3.11+, `pip install -r requirements.txt` và các file dữ liệu
đúng mapping: `content/items.csv.gz`, `content/products_text.csv.gz`,
`content/tfidf_all.npz`. Ảnh lấy từ `foundation_catalog_v1/catalog.csv.gz` nếu có.
Các file lớn không nằm trên GitHub. Không thể clone repo trống dữ liệu rồi demo ngay.

## Kịch bản trình bày 3 phút

1. Giới thiệu catalog Amazon Toys and Games gồm 767.045 sản phẩm. Chỉ rõ tên
   backend đang hiển thị: TF-IDF content baseline.
2. Bấm LEGO, thêm một sản phẩm. Lịch sử được đánh số từ cũ đến mới; không thêm
   trùng một sản phẩm, tối đa 20. TF-IDF dùng 10 sản phẩm gần nhất.
3. Chọn Top K = 5, bấm “Tìm gợi ý cho tôi”. Trình bày ảnh, tên, mã ASIN,
   điểm xếp hạng và số tương tác train. Sản phẩm đã chọn không xuất hiện lại.
4. Đổi nhóm sang zero-shot, chạy lại. Tất cả kết quả có train count = 0.
   Tín hiệu văn bản vẫn cho phép baseline xếp hạng sản phẩm chưa có tương tác.
5. Bấm “Xuất JSON” để lưu đúng đầu vào, tên backend, bộ lọc, thời gian và kết quả.
   Khi đổi lịch sử/bộ lọc, kết quả cũ được làm mờ và tắt xuất cho đến khi chạy lại.
6. Xóa lịch sử rồi chọn tất cả nhóm, chạy lại để minh họa fallback theo độ phổ biến.

## Cách giải thích đúng

- Zero-shot là không có tương tác trong train; không có nghĩa vừa ra mắt.
- Điểm không phải xác suất mua hàng. Không so trực tiếp score của hai mô hình.
- Empty history + zero-shot không có tín hiệu cá nhân hóa hoặc popularity để
  phân biệt các sản phẩm; thứ tự khi đó là tie-break xác định.
- Thời gian hiển thị đo xử lý inference trên server; chưa gồm tải ảnh và mạng.
- Demo xếp hạng toàn catalog hoặc toàn nhóm đã lọc. Metric trong báo cáo dùng
  1 positive + 99 negative cố định, nên không phải chất lượng demo toàn catalog.
- Ảnh lỗi có placeholder; tên sản phẩm vẫn hiển thị. Lịch sử không lưu lâu dài.

## Chuyển sang UFM sau khi train xong

```powershell
python src/demo_recommender.py --backend ufm --run runs/ufm_full_v1 --features data/processed/toys_games_full_temporal/foundation_clip_b32_v1 --port 8766 --open-browser
```

Cần mang về checkpoint production `completed.json`, `config.json`, `best.pt` và
toàn bộ CLIP feature tables kèm manifest. Tham số `--features` cho phép đổi vị trí
cache từ FITLAB sang local, nhưng vẫn kiểm tra hash manifest và các bảng gốc.
Giữ nguyên config và checkpoint; không sửa fingerprint. Không dùng technical
smoke làm checkpoint demo UFM. Alpha CF/semantic và uncertainty chỉ xuất hiện khi
backend UFM thật đã được tải thành công.

## Xử lý lỗi thường gặp

- Cổng 8765 bận: nếu là demo hiện tại, launcher mở lại. Nếu là dịch vụ khác,
  chạy thủ công với `--port 8766`; không dừng tiến trình không rõ nguồn.
- Thiếu dữ liệu: khôi phục đúng bộ dữ liệu local đã xử lý; không tạo dữ liệu giả.
- Không có kết quả: đổi từ khóa tiếng Anh, nhóm sản phẩm hoặc giảm lịch sử.
- UFM chưa có completed.json: tiếp tục chờ pipeline GPU; dùng backend content.
