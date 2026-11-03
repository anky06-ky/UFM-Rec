# Chạy đúng vòng UFM trên Colab

Cập nhật 29/10/2026. Vòng đã khóa dùng **Toys & Games full temporal 80/10/10** và 15 lượt: năm biến thể × seed 42, 7, 2026. Mã huấn luyện được ghim tại commit `4b72a2d`; notebook có thể được sửa phần vận hành mà không đổi ba file nguồn đã khóa.

## Chọn notebook

- Colab trong trình duyệt: tải [UFM_Colab_Browser_Train.ipynb](../notebooks/UFM_Colab_Browser_Train.ipynb) từ repo local rồi mở bằng **Tệp → Mở sổ tay → Tải lên**. Mặc định runtime T4; nếu tài khoản có L4, đổi `GPU_TYPE = 'L4'` ở cell đầu (phép đo throughput lưu riêng `l4_throughput.json`). Bản `UFM_Colab_L4_Train.ipynb` trước đây chỉ khác chữ T4/L4 nên đã gộp vào biến này.
- Colab trong VS Code: mở [UFM_Colab_Train.ipynb](../notebooks/UFM_Colab_Train.ipynb), chọn kernel Colab GPU T4.

Chọn một cách để chạy nối tiếp cùng kế hoạch; không chạy hai notebook đồng thời vào cùng thư mục thí nghiệm. Khi đã mở notebook trên Colab, sửa file local không tự cập nhật bản đang mở; cần dùng bản mới khi bắt đầu phiên phù hợp.

Ảnh người dùng cung cấp ngày 29/10 cho thấy `Huan_Luyen.ipynb` đang đặt `DATA_DIR` là `/content/drive/MyDrive/DA_AI/data/toys_games_70_15_15`. Đây là đường dẫn bộ chia cũ, **khác bộ đầu vào khóa của vòng 15 run**. Có GPU T4 chưa đủ để xác nhận đang chạy đúng vòng. Giữ bản notebook cũ và mở riêng bản UFM nêu trên; không sửa tên thư mục dữ liệu cũ thành `full_temporal` để ép chạy.

## Các mốc phải thấy trước khi train

1. `SOURCE_OK`: đúng source commit và kernel GPU T4. Notebook giữ PyTorch CUDA của runtime; phiên bản thư viện phải khớp khi resume.
2. `DRIVE_WRITE_OK`: Drive đã gắn, ghi được vào `MyDrive/UFM-Rec_Experiments/ufm_followup_20261028`.
3. `ZIP_AND_DRIVE_SHA_OK`: sáu ZIP trong `D:\Nam3_ki1\UFM-Rec_Backups\colab_training_inputs_20261029` đã nhận và đối chiếu. Trong bản trình duyệt, chọn một ZIP mỗi lần khi được hỏi. Phiên sau dùng bản đã lưu trên Drive.
4. `COLAB_PREFLIGHT_PASSED 9 locked hashes`: đã restore đúng dữ liệu, features, counts và validation. Receipt được lưu trên Drive ở `colab_input_receipt.json`.
5. `ORCHESTRATOR_READY 15 locked jobs`: sẵn sàng chạy kế hoạch, chưa phải đã train xong.

Không đổi split, source hash, batch, seed hay giới hạn validation để vượt qua lỗi preflight. Đối chiếu đúng dữ liệu thiếu trước khi tiếp tục.

## Train, ngắt phiên và resume

- Cell đo tốc độ dùng 100 update đầu của lượt full/seed 42, rồi tiếp tục từ checkpoint đó. Nếu checkpoint có sẵn nhưng thiếu file throughput, notebook đo một đoạn tiếp theo đủ hai mốc log; không xóa checkpoint hoặc train lại từ đầu.
- Lượt train được ngắt có kiểm soát theo đoạn và lưu optimizer/RNG/cursor để nối tiếp. `DRIVE_CHECKPOINT_SHA_OK` xác nhận file của đoạn vừa sao lưu đã được kiểm tra trên Drive.
- Nếu phiên ngắt trước khi checkpoint mới kịp sao lưu, chỉ có thể resume từ bản sao đã xác minh gần nhất. Không coi phần chưa lưu là đã được bảo toàn.
- Khi mở phiên mới, kết quả đánh giá đã có receipt hợp lệ được khôi phục; `REPORTS_FROM_DRIVE_SHA_OK` xác nhận report và predictions khớp SHA. Backup thiếu receipt, sai source hoặc hỏng file sẽ bị từ chối để đối chiếu.
- Giữ cùng source/config/phiên bản thư viện. Không chạy thêm một campaign vào cùng thư mục trong khi lượt cũ đang hoạt động.
- Ngân sách 120/150 phút trong notebook là giới hạn do bộ chạy đặt, không bảo đảm runtime Colab tồn tại đủ thời gian đó. Ước lượng throughput chưa bao gồm toàn bộ validation và sao lưu.

## Thế nào là đã hoàn thành

`completed.json` hợp lệ + best checkpoint + report v3 + predictions mới đủ bằng chứng cho một lượt. `RUN_COMPLETE_AND_BACKED_UP` là mốc đã đánh giá và sao lưu của lượt đó; forward/backward smoke, loss hoặc checkpoint giữa chừng chưa phải kết quả cuối.

Đủ 15 report mới tổng hợp bằng `scripts/summarize_locked_ufm_followup.py`. Tool kiểm tra seed, variant, source/features/counts/candidate hash và chỉ nhận validation; thiếu seed hoặc lẫn test bị từ chối. File tổng hợp hiện chứa NDCG từng seed/nhóm và mean ± sample std của cold macro; Recall/MRR và breakdown chi tiết vẫn nằm trong từng report gốc.

Tải các artifact cần bàn giao về local và đối chiếu SHA. Drive checkpoint không đồng nghĩa đã có bản tải về máy; receipt local phải xác nhận riêng.

## Giới hạn khi báo cáo kết quả

Checkpoint được chọn theo validation v1; v3 là đánh giá thăm dò trên quần thể liên quan, không phải holdout độc lập. Test v2 đã được xem và không dùng để chọn mô hình/seed. Kết quả v3 dùng 100 candidates/ca, không đại diện trực tiếp cho full-catalog ranking. Xem [giao thức](protocol_current.md), [runbook](gpu_followup_runbook.md) và [checklist còn thiếu](../reports/project_readiness_20261029.md).
