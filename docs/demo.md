# Cập nhật bản sửa proxy ngày 02/10/2026

Handler nhận cả đường dẫn giữ prefix và đường dẫn đã loại prefix theo hành vi
[code-server](https://coder.com/docs/code-server/guide). Năm kiểm thử demo local
PASS, bao gồm GET/POST qua hai dạng path và từ chối origin sai. Bản sửa mới chưa
được triển khai/nghiệm thu lại UFM inference trên FITLAB.
Trạng thái mới: [reports/progress.md](../reports/progress.md).

---

# Chạy và trình bày demo UFM Rec

## Demo UFM trên FITLAB

Full UFM đã hoàn tất. Trên FITLAB, backend UFM được chạy bằng CPU trên checkpoint
`runs/ufm_full_v1`, với catalog 767.045 sản phẩm. Mở cổng **8766** trong tab Ports
của code-server; đường dẫn proxy hiện tại là:

`https://coder-2474802010460.fitlab-02.is-tech.vn/proxy/8766/`

App cần được khởi động với origin FITLAB được cho phép và base path `/proxy/8766`.
Sau khi mở trang, xác nhận nhãn backend là **UFM Rec**, tìm từ khóa `LEGO`, thêm
một món vào lịch sử, chọn `Zero-shot`, chạy Top 5 rồi thử xuất JSON. Nếu trang báo
`Loopback origin required` hoặc API không tải, kiểm tra log `runs/demo_ufm_v1.log`
và tham số proxy trước khi trình bày.

**Trạng thái 02/10/2026, 18:04 UTC+07:** lần kiểm tra gần nhất, PID 72211 còn chạy
server UFM bằng tham số cũ nên proxy trả `Loopback origin required`. Một request
inference sau đó làm server thoát với `Bus error`; đồng thời workspace
`/home/coder/iDragonCloud/DA_AI` trả `EIO` và code-server báo mất kết nối. Cổng
8766 hiện không được xem là demo UFM sẵn sàng. Khôi phục mount và kiểm tra log/
checkpoint trước khi khởi động lại. Trang `127.0.0.1:8766` có thể mở được nhưng
đang dùng TF-IDF baseline, không phải UFM.

## Giao diện hiện tại

`demo/index.html` có hero xanh rừng, thẻ kết quả làm nổi bật ảnh và nhãn
cold-start, hiệu ứng hover/focus và bố cục thích ứng màn hình nhỏ. Trên mỗi thẻ,
score được ghi rõ là **Điểm xếp hạng**; backend và giới hạn điểm được hiển thị để
người xem không nhầm score với xác suất mua hàng.

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
   backend đang hiển thị và nói rõ đây là checkpoint UFM hoặc TF-IDF baseline.
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

## Khởi động backend UFM trên FITLAB

```bash
python3 src/demo_recommender.py --backend ufm --run runs/ufm_full_v1 --port 8766 \
  --allowed-origin https://coder-2474802010460.fitlab-02.is-tech.vn \
  --base-path /proxy/8766
```

Lệnh này chỉ bind loopback; truy cập từ ngoài đi qua proxy code-server. Origin phải
khớp chính xác host HTTPS của workspace và path prefix phải khớp cổng đã forward.
Khi chạy local, dùng `--backend content` hoặc bỏ hai tùy chọn proxy.

Checkpoint production phải có `completed.json`, `config.json`, `best.pt` và CLIP
feature tables kèm manifest. Tham số `--features` cho phép đổi vị trí cache nếu
chuyển môi trường; hash manifest và bảng gốc vẫn được kiểm tra. Giữ nguyên config
và checkpoint; không dùng technical smoke làm checkpoint demo UFM. Alpha CF,
semantic và uncertainty chỉ được trình bày khi backend UFM thật tải thành công.

## Xử lý lỗi thường gặp

- Cổng 8765 bận: nếu là demo hiện tại, launcher mở lại. Nếu là dịch vụ khác,
  chạy thủ công với `--port 8766`; không dừng tiến trình không rõ nguồn.
- Thiếu dữ liệu: khôi phục đúng bộ dữ liệu local đã xử lý; không tạo dữ liệu giả.
- Không có kết quả: đổi từ khóa tiếng Anh, nhóm sản phẩm hoặc giảm lịch sử.
- UFM chưa có completed.json: tiếp tục chờ pipeline GPU; dùng backend content.