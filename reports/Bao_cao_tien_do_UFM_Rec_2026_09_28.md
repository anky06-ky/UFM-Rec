# Báo cáo tiến độ đồ án UFM Rec

## Hệ khuyến nghị sản phẩm mới trên thương mại điện tử

Môn học: Trí tuệ nhân tạo và ứng dụng · Cập nhật: 28/09/2026  
Nhóm: Trần An Kỳ, Nguyễn Thành Phong, Nguyễn Hoàng Hải  
Khoa Công nghệ Thông tin, Trường Đại học Văn Lang

### 1 Tổng quan tiến độ

Đồ án đã có dữ liệu quy mô lớn và kết quả của bốn mô hình nền. Transformer trên GPU đã hoàn thành bốn epoch, nhưng hiện dừng giữa epoch 5 và chưa đánh giá test. Việc tiếp theo là khôi phục lượt train, chốt phạm vi với giảng viên và hoàn thiện thực nghiệm.

So với proposal gốc, phần Foundation Model, ước lượng độ bất định và hợp nhất UGAF chưa được triển khai. Vì vậy, kết quả hiện tại là tiến độ xây dựng nền tảng và thử nghiệm mô hình, chưa phải kết quả hoàn chỉnh của kiến trúc UFM-Rec đề xuất.

| Hạng mục | Trạng thái | Kết quả hoặc việc còn thiếu |
|---|---|---|
| Dữ liệu và chia tập | Hoàn thành | 11.572.689 tương tác; chia theo thời gian 80/10/10 |
| Lịch sử tương tác | Hoàn thành | Lịch sử từ quá khứ cho huấn luyện và đánh giá |
| Đặc trưng nội dung | Hoàn thành | 767.045 sản phẩm; TF-IDF 20.000 đặc trưng |
| Mẫu đánh giá và baseline | Hoàn thành | Candidate cố định; có metric validation và test |
| Transformer trên GPU | Chưa hoàn tất | Có checkpoint; đã xong 4 epoch; chưa test |
| Foundation Model và UGAF | Chưa triển khai | Còn thiếu các thành phần chính của proposal |
| Báo cáo cuối kỳ và bảo vệ | Chưa hoàn tất | Cần tổng hợp thực nghiệm, biểu đồ và slide |

### 2 Quy mô dữ liệu hiện tại

Dữ liệu có 16.260.406 review gốc; giữ tương tác mua hàng đã xác minh, rating từ 4 trở lên và loại 121.726 bản ghi trùng. Có 6.158.534 người dùng sau lọc. Bộ lớn 80/10/10 thay thế bộ 5-core 70/15/15 trước đây.

| Tập dữ liệu | Số tương tác | Tỷ lệ thực tế |
|---|---:|---:|
| Train | 9.258.647 | 80,004% |
| Validation | 1.159.827 | 10,022% |
| Test | 1.154.215 | 9,974% |
| Tổng | 11.572.689 | 100% |

Tỷ lệ lệch nhẹ do giữ ranh giới trọn ngày UTC. Lượt GPU dùng 4.230.848 mẫu train có lịch sử, không phải toàn bộ 9,26 triệu tương tác train.

<!-- pagebreak -->

### 3 Giao thức dữ liệu và đánh giá

Validation bắt đầu ngày 11/08/2021 UTC; test bắt đầu ngày 16/07/2022 UTC. Lịch sử chỉ lấy tương tác trước thời điểm mục tiêu; các tương tác cùng timestamp không nhìn thấy nhau. Pipeline lưu tối đa 50 sản phẩm trước đó; thử nghiệm GPU dùng 20 sản phẩm gần nhất. Lịch sử test có thể bao gồm sự kiện train và validation xảy ra trước đó, theo cách đánh giá trực tuyến.

Với cách chia theo thời gian, cùng người dùng hoặc sản phẩm có thể xuất hiện ở nhiều tập. Điều này khác với chia tách hoàn toàn theo sản phẩm. Cần kiểm tra không trùng bản ghi tương tác và không đưa sự kiện tương lai vào lịch sử; không nên yêu cầu mọi ID sản phẩm đều khác nhau giữa train, validation và test.

TF-IDF chỉ fit vocabulary và IDF trên sản phẩm có tương tác train, sau đó biến đổi toàn bộ sản phẩm bằng bộ đặc trưng đã cố định. Văn bản sử dụng tiêu đề, đặc điểm, mô tả, cửa hàng và danh mục. Ảnh sản phẩm chưa được dùng làm đặc trưng trong mô hình hiện tại.

Mỗi tập validation và test có 40.000 trường hợp đánh giá, gồm 10.000 trường hợp cho mỗi nhóm sản phẩm: zero-shot có 0 tương tác train, extreme cold có 1–5, cold có 6–20 và warm có trên 20. Mỗi trường hợp có một sản phẩm đúng và 99 sản phẩm âm được lấy mẫu cố định, dùng chung giữa các mô hình. Đây là xếp hạng trong 100 ứng viên, chưa phải xếp hạng toàn bộ danh mục.

### 4 Kết quả các mô hình nền

Bảng dưới là kết quả trên cùng 40.000 mẫu test. Recall@10 đo khả năng đưa sản phẩm đúng vào top 10; NDCG@10 và MRR@10 phản ánh thêm vị trí của sản phẩm đúng trong danh sách.

| Mô hình | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Popularity | 0,2669 | 0,1691 | 0,1393 |
| TF-IDF content profile | 0,4013 | 0,2658 | 0,2243 |
| Collaborative TruncatedSVD | 0,2193 | 0,1310 | 0,1043 |
| Hybrid RRF | 0,4231 | 0,2657 | 0,2172 |

Hybrid RRF tăng Recall@10 khoảng 5,4% tương đối so với content baseline, nhưng NDCG gần như ngang nhau và MRR thấp hơn. Trọng số của nhánh SVD được chọn bằng 0 trên validation, nên chưa có bằng chứng SVD bổ sung giá trị trong cấu hình hybrid này.

| Recall@10 theo nhóm | Zero shot | Extreme cold | Cold | Warm |
|---|---:|---:|---:|---:|
| TF-IDF content | 0,2816 | 0,2874 | 0,3296 | 0,7065 |
| Collaborative SVD | 0,0000 | 0,0775 | 0,1498 | 0,6497 |
| Hybrid RRF | 0,2731 | 0,2575 | 0,3146 | 0,8473 |

Lợi ích Recall của hybrid chủ yếu nằm ở nhóm warm; ba nhóm sản phẩm ít hoặc chưa có tương tác có Recall thấp hơn content-only. Vì vậy chưa thể kết luận hybrid hiện tại giải quyết cold-start tốt hơn, dù chỉ số trung bình cao hơn.

<!-- pagebreak -->

### 5 Tiến độ thử nghiệm GPU trên FITLAB

Thử nghiệm dùng PyTorch 2.12.1+cu130 trên Quadro RTX 6000 có 24 GiB VRAM. Mô hình có nhánh sản phẩm kết hợp các term TF-IDF với embedding ID của sản phẩm train; nhánh người dùng là Transformer hai lớp, bốn attention head, chiều embedding 128 và lịch sử tối đa 20 sản phẩm. Đây là mô hình tự triển khai, chưa phải SASRec hoặc UFM-Rec gốc.

Lượt chạy dùng 4.230.848 mẫu train có lịch sử, batch size 256, learning rate 0,0003 và seed 42. Cấu hình giới hạn 20 epoch; dừng sớm sau ba epoch không cải thiện NDCG@10 của nhóm có lịch sử trên validation. Checkpoint định kỳ lưu mỗi 1.000 bước; test không chạy tự động trong quá trình train.

| Epoch đã xong | Train loss | Recall validation | NDCG validation | NDCG có lịch sử |
|---|---:|---:|---:|---:|
| 1 | 4,5574 | 0,399950 | 0,262510 | 0,351274 |
| 2 | 4,2416 | 0,401950 | 0,267774 | 0,361494 |
| 3 | 4,1278 | 0,406025 | 0,272334 | 0,370348 |
| 4 | 4,0553 | 0,406875 | 0,273501 | 0,372614 |

Train loss giảm và metric validation tăng qua bốn epoch. Checkpoint tốt nhất trong các epoch đã hoàn thành là epoch 4. So sánh dưới đây chỉ dùng validation, không trộn kết quả GPU validation với baseline test.

| Cùng 40.000 mẫu validation | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| TF-IDF content | 0,391925 | 0,260367 | 0,219947 |
| Hybrid RRF | 0,411975 | 0,266104 | 0,221060 |
| Transformer epoch 4 | 0,406875 | 0,273501 | 0,232476 |

Transformer epoch 4 có NDCG validation cao hơn content khoảng 5,0% tương đối và cao hơn hybrid khoảng 2,8%; Recall vẫn thấp hơn hybrid. Kết quả này khuyến khích hoàn thiện thử nghiệm, nhưng chưa đủ để xác nhận hiệu quả trên test hoặc gọi đây là mô hình tốt nhất.

### Trạng thái kiểm tra ngày 28 tháng 9

Không tìm thấy tiến trình train_gpu_recommender.py đang chạy. Log dừng ở epoch 5, bước 81.100, sau khoảng 139 phút tính từ lúc bắt đầu. Không có completed.json và chưa có test_metrics.json. Log hiện không cho biết nguyên nhân dừng; không thể kết luận là hết bộ nhớ hoặc container bị tắt chỉ từ các dấu hiệu này.

best.pt khoảng 386 MiB và latest.pt khoảng 1.158 MiB vẫn tồn tại trên iDragonCloud. Cần kiểm tra khả năng nạp checkpoint trước khi resume. Trạng thái này là lượt train chưa hoàn tất, không phải đã dừng sớm thành công theo điều kiện validation.

<!-- pagebreak -->

### 6 Đối chiếu với proposal gốc

Proposal.pdf đề xuất UFM-Rec kết hợp biểu diễn ngữ nghĩa từ bộ mã hóa tiền huấn luyện với hành vi tuần tự, ước lượng độ bất định riêng cho hai nhánh và hợp nhất UGAF. Các ngưỡng cải thiện trong proposal là mục tiêu kiểm chứng, không phải kết quả đã đạt.

| Thành phần đề xuất | Hiện trạng | Việc cần bổ sung |
|---|---|---|
| Dữ liệu và lịch sử tuần tự | Đã có Amazon theo thời gian | Kiểm thử tự động và lưu biên bản kiểm tra |
| Nhánh hành vi tuần tự | Có Transformer tự triển khai | Thêm baseline SASRec hoặc BERT4Rec nếu giữ thiết kế so sánh gốc |
| Semantic Foundation Branch | Hiện dùng TF-IDF | Bộ mã hóa văn bản tiền huấn luyện; ảnh nếu giữ phạm vi đa phương thức |
| Dual Uncertainty Estimation | Chưa có | Ước lượng độ bất định cho hai nhánh và kiểm tra độ tin cậy |
| Hợp nhất UGAF | Hiện có hybrid RRF | Gate theo độ tin cậy và thành phần căn chỉnh chéo như thiết kế đã chốt |
| Ablation và calibration | Chưa có kết quả | Bỏ từng nhánh, bỏ uncertainty, fusion cố định; ECE hoặc thước đo phù hợp |
| Thực nghiệm bổ sung | Mới có Toys and Games | Tập thứ hai hoặc thử nghiệm nhỏ nếu giữ yêu cầu trong proposal |
| Hiệu quả vận hành | Có log thời gian và VRAM | Đo tham số cập nhật, tốc độ suy diễn và bộ nhớ nhất quán |

### Hai phương án hoàn thiện phạm vi

Nếu giữ proposal gốc, cần xây dựng Foundation Model branch, uncertainty heads và UGAF, sau đó so sánh với baseline phù hợp. Mục tiêu cold-start cần kiểm tra riêng trên zero-shot và extreme cold; cải thiện trung bình do warm không đủ để chứng minh giả thuyết chính. LoRA hoặc adapter chỉ nên bổ sung khi cần và phải báo cáo rõ chiến lược đóng băng hoặc tinh chỉnh encoder.

Nếu thời hạn ngắn, có thể đề nghị giảng viên duyệt đề tài thu gọn về so sánh content-based, collaborative, hybrid và content-aware Transformer trên dữ liệu lớn. Khi đó cần sửa tên và mục tiêu nghiên cứu cho đúng mô hình đã triển khai; Foundation Model và UGAF chuyển thành hướng phát triển. Không tự coi phương án thu gọn là đã được chấp thuận.

Việc cần quyết định sớm nhất là phạm vi được giảng viên chấp nhận. Chưa nên dành thêm nhiều giờ train chỉ để đạt đủ 20 epoch khi các thành phần nghiên cứu cốt lõi của proposal vẫn còn thiếu.

<!-- pagebreak -->

### 7 Các việc cần làm theo thứ tự ưu tiên

1. Bảo toàn và khôi phục lượt GPU. Sao lưu best.pt, latest.pt, config.json và history.jsonl cùng đặc trưng, mapping sản phẩm. Kiểm tra nạp checkpoint và nguyên nhân dừng; nếu tiếp tục, dùng resume đúng cấu hình. Đầu ra cần có là lượt kết thúc có trạng thái rõ ràng và checkpoint được chọn chỉ bằng validation.

2. Chốt phạm vi và kiểm thử dữ liệu. Đối chiếu proposal với giảng viên; thống nhất thành phần bắt buộc và tiêu chí đánh giá. Bổ sung kiểm tra trùng bản ghi giữa các tập, ranh giới thời gian, lịch sử không chứa tương lai, nguồn fit TF-IDF và các candidate âm. Lưu kết quả kiểm tra thành file để đưa vào báo cáo.

3. Hoàn thiện mô hình theo phạm vi đã duyệt. Nếu giữ UFM-Rec gốc, ưu tiên encoder văn bản đóng băng, hai nhánh tách biệt, uncertainty và UGAF trước khi mở rộng ảnh hoặc LoRA. Nếu thu gọn, hoàn thiện mô hình Transformer hiện có và phân tích vì sao hybrid tăng warm nhưng giảm cold-start.

4. Chốt thực nghiệm trước khi chạy test cuối. Tuning chỉ bằng train và validation; dùng cùng candidate set, cùng nhóm cold-start và cùng chính sách fallback cho người dùng không có lịch sử. Thực hiện ablation; nếu thời gian cho phép, chạy nhiều seed và khoảng tin cậy. Sau khi chốt cấu hình, đánh giá checkpoint được chọn trên test; không tiếp tục tuning theo test.

5. Hoàn thiện đầu ra khuyến nghị và đo hiệu quả. Tạo notebook hoặc demo đơn giản nhập lịch sử người dùng và trả danh sách top 10 có tên sản phẩm. Kiểm thử lịch sử rỗng, sản phẩm mới và ID không hợp lệ. Đo thời gian suy diễn, bộ nhớ và tham số cập nhật. Demo là đầu ra ứng dụng khuyến nghị, không thay thế đánh giá offline.

6. Hoàn thiện hồ sơ nộp và bảo vệ. Viết báo cáo cuối kỳ với bảng kết quả cùng giao thức, biểu đồ loss và validation, phân tích cold-start, ablation và giới hạn. Chuẩn bị slide, hướng dẫn tái lập, nguồn dữ liệu và cách nạp checkpoint; không đưa dữ liệu lớn hoặc thông tin đăng nhập vào Git.

### 8 Kế hoạch thời gian đề xuất

Đây là ước lượng theo ngày làm việc kể từ khi chốt phạm vi, chưa phải lịch nộp đã xác nhận. Thời gian chạy GPU thực tế còn phụ thuộc quyền sử dụng FITLAB và số cấu hình thử nghiệm.

| Giai đoạn | Bản thu gọn nếu được duyệt | Giữ UFM Rec gốc |
|---|---|---|
| Khôi phục GPU và kiểm thử dữ liệu | Ngày 1–2 | Ngày 1–3 |
| Hoàn thiện mô hình và ablation | Ngày 3–5 | Ngày 4–12 |
| Đánh giá cuối và đo hiệu quả | Ngày 6–7 | Ngày 13–17 |
| Demo báo cáo slide và tái lập | Ngày 8–10 | Ngày 18–22 |

Nếu thiếu thời gian, ưu tiên một thực nghiệm có kiểm soát và báo cáo đúng phạm vi. Không cam kết đạt ngưỡng cải thiện 3–8% hoặc một số giờ train cố định trước khi kiểm chứng.

<!-- pagebreak -->

### 9 Tiêu chí hoàn thành đồ án

- Phạm vi, tên đề tài và đóng góp khớp với triển khai, được giảng viên xác nhận khi có thay đổi so với proposal.
- Dữ liệu có nguồn, checksum, manifest và biên bản kiểm tra chia tập, lịch sử và đặc trưng.
- Lượt train kết thúc có trạng thái rõ ràng; checkpoint được nạp lại và tái lập suy diễn thành công.
- Có bảng validation và test cho mô hình cuối cùng, baseline, từng nhóm cold-start và nhóm lịch sử rỗng.
- Có ablation và phép đo efficiency; nếu giữ UFM-Rec gốc, có bằng chứng hoạt động của uncertainty, UGAF và đánh giá calibration.
- Có đầu ra top 10 kiểm thử được, hướng dẫn chạy, báo cáo cuối kỳ, biểu đồ và slide bảo vệ.
- Các kết luận nêu cả cải thiện lẫn phần không cải thiện; không gán kết quả sampled ranking cho xếp hạng toàn danh mục.

### 10 Giới hạn cần nêu trong báo cáo cuối

Mẫu đánh giá cân bằng bốn nhóm không phản ánh tỷ trọng thực tế. Candidate âm lấy từ các item quan sát đến cuối mỗi tập, chưa bảo đảm có sẵn tại từng thời điểm. Cần ghi rõ đây là benchmark lấy mẫu trước khi kết luận về triển khai thực tế.

Metadata snapshot có thể cập nhật sau thời điểm tương tác. Mô hình hiện chỉ dùng văn bản; in-batch negative có thể chứa sản phẩm người dùng thích ngoài lịch sử đã biết. Người dùng chưa có lịch sử dùng popularity fallback, ảnh hưởng đến metric tổng thể.

GPU hiện mới có validation và một seed. Train loss giảm hoặc train lâu hơn không tự chứng minh chất lượng tốt hơn; cần metric test, ablation và phân tích cold-start.

### 11 Minh chứng và vị trí lưu kết quả

Minh chứng dữ liệu nằm trong data/processed/toys_games_full_temporal. Minh chứng GPU nằm trong dự án /home/coder/iDragonCloud/DA_AI trên FITLAB.

- docs/history/proposal-v1.pdf: kiến trúc, giả thuyết và tiêu chí nghiên cứu gốc.
- split_manifest.json, content/content_report.json, evaluation/sampling_report.json: dữ liệu, đặc trưng và giao thức đánh giá.
- models/content_baseline/metrics.json, models/collaborative_svd/report.json, models/hybrid_rrf/report.json: kết quả các baseline.
- FITLAB runs/content_transformer_v1/config.json và history.jsonl: cấu hình cùng bốn epoch đã hoàn thành.
- FITLAB runs/content_transformer_v1.log, best.pt và latest.pt: log cùng checkpoint; kiểm tra ngày 28/09/2026.

### Kết luận và bước tiếp theo

Ưu tiên sao lưu checkpoint, kiểm tra lượt GPU bị dừng và chốt phạm vi với giảng viên. Nếu giữ UFM-Rec gốc, cần triển khai và kiểm chứng Foundation Model, uncertainty và UGAF trước khi đánh giá cuối và hoàn thiện hồ sơ bảo vệ.
