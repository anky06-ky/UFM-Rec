# Báo cáo đồ án UFM Rec

> Nguồn nội dung của bản nộp theo mẫu Trường Đại học Văn Lang: [Word](../output/docx/UFM_Rec_Bao_Cao_Do_An_20261103.docx), [PDF](../output/pdf/UFM_Rec_Bao_Cao_Do_An_20261103.pdf), dựng bằng `python scripts/build_school_report_docx.py`. Nhóm chốt đồ án với các checkpoint và bằng chứng hiện có; số liệu chính là validation v3 (40.000 ca, 100 ứng viên/ca, seed 42), truy nguồn trong [bảng bằng chứng](../reports/nontraining_20261029/evidence.json). Vòng 15 lượt huấn luyện đa seed là hướng phát triển (mục 5.1), không phải kết quả đã có.

## UFM-Rec: Hệ Khuyến Nghị Sản Phẩm Cold-Start Đa Phương Thức Dựa Trên Độ Tin Cậy Học Được

- **Học phần**: Học máy ứng dụng
- **Giảng viên hướng dẫn**: Phan Hồ Viết Trường
- **Nhóm thực hiện**: Nguyễn Thành Phong, Trần An Kỳ, Nguyễn Hoàng Hải, Hân
- **Phiên bản**: 03/11/2026 — bản nộp, chốt với kết quả hiện có

---

### TÓM TẮT ĐỀ TÀI (ABSTRACT)

Bài toán khởi đầu lạnh (Cold-Start) và đặc biệt là cực lạnh (Extreme Cold-Start) là một trong những thách thức cốt lõi và dai dẳng nhất của các hệ thống gợi ý hiện đại (Recommender Systems). Khi một sản phẩm mới xuất hiện trên sàn thương mại điện tử hoặc một người dùng mới gia nhập nền tảng, các phương pháp lọc cộng tác truyền thống (Collaborative Filtering - CF) và mô hình chuỗi hành vi (Sequential Recommenders như SASRec, BERT4Rec) gặp hạn chế do thiếu tương tác để học biểu diễn định danh cho người dùng hoặc sản phẩm mới.

Để giải quyết bài toán này, đề tài xây dựng và thực nghiệm kiến trúc **UFM-Rec (Uncertainty-aware Fusion Model Recommender)**. Mô hình kết hợp biểu diễn đa phương thức đóng băng (Frozen Multimodal Features) từ **CLIP ViT-B/32** (bao gồm văn bản tiêu đề, mô tả và hình ảnh trực quan sản phẩm) với biểu diễn tuần tự (Causal Sequential Transformer) của lịch sử tương tác. Đóng góp thiết kế của UFM-Rec là việc tích hợp **Cơ chế Ước Lượng Độ Tin Cậy Học Được (Learned Reliability Proxy)** và **Cổng Căn Chỉnh Đa Phương Thức (Cross-Modal Alignment Gate)**, cho phép mạng tự động điều tiết trọng số giữa ngữ nghĩa nội dung (Semantic Branch) và hành vi cộng tác (Collaborative Branch) tùy theo mức độ khả dụng và sai số ước lượng của dữ liệu.

Nghiên cứu thực nghiệm trên Amazon Reviews 2023 — Toys & Games (767.045 sản phẩm, khoảng 11,57 triệu tương tác), chia thời gian 80/10/10. Vòng đánh giá lại ngày 28/10/2026 dùng validation v3 cố định: 40.000 ca, 1 positive + 99 negatives, loại positive đã biết trước target. Negative vẫn lấy theo regime của target và thống kê cuối split nên đánh giá có tính hồi cứu. Trên cùng v3, Cold macro NDCG@10 của TF-IDF là **0,195682**, semantic-only dùng CLIP **0,190036**, UFM full **0,182056**, popularity **0,042527**. Tám checkpoint UFM được đối chiếu cùng nguồn đặc trưng và candidates; chưa chứng minh fusion/uncertainty cải thiện chỉ số chính. Các checkpoint có một seed (42), chọn từ validation v1; đánh giá v3 không tạo holdout độc lập. Hiệu chỉnh thăm dò trên hai phần validation theo thời gian giảm ECE của full UFM từ **0,082061** xuống **0,037478**; đây là confidence có điều kiện trên 100 ứng viên, không phải xác suất mua. Test v2 đã được đánh giá trong lịch sử; vòng mới không mở thêm test. Chưa có kết quả đa seed UFM hoặc UFM trên All Beauty được xác minh. Số liệu lịch sử v1/v2 được giữ riêng trong Chương 4; nguồn hiện tại là [bộ bằng chứng ngày 28/10](../reports/continuation_20261028/README.md).

---

### MỤC LỤC

1. [CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ](#chương-1-giới-thiệu--đặt-vấn-đề)
2. [CHƯƠNG 2: KIẾN TRÚC MÔ HÌNH UFM-REC](#chương-2-kiến-trúc-mô-hình-ufm-rec)
3. [CHƯƠNG 3: THIẾT LẬP THỰC NGHIỆM & GIAO THỨC ĐÁNH GIÁ](#chương-3-thiết-lập-thực-nghiệm--giao-thức-đánh-giá)
4. [CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM, TÀI NGUYÊN & GIỚI HẠN NGHIÊN CỨU](#chương-4-kết-quả-thực-nghiệm-tài-nguyên--giới-hạn-nghiên-cứu)
5. [CHƯƠNG 5: CÁC ĐỀ XUẤT VÀ HƯỚNG MỞ RỘNG](#chương-5-các-đề-xuất-và-hướng-mở-rộng)
6. [CHƯƠNG 6: KẾT LUẬN](#chương-6-kết-luận)

---

### CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ

#### 1.1. Thách thức Cold-Start trong Thương mại Điện tử
Trong bài toán gợi ý sản phẩm, các item mới hoặc ít tương tác có ít dữ liệu để học biểu diễn định danh. Các sản phẩm này rơi vào các phân tầng lịch sử (Regimes) khác nhau tính theo số lượt xuất hiện trong tập huấn luyện ($count_{train}$):
- **Zero-shot**: Sản phẩm chưa từng có bất kỳ tương tác nào trong tập huấn luyện ($count_{train} = 0$).
- **Extreme Cold**: Sản phẩm chỉ có từ 1 đến 5 tương tác ($1 \le count_{train} \le 5$).
- **Cold**: Sản phẩm có từ 6 đến 20 tương tác ($6 \le count_{train} \le 20$).
- **Warm**: Sản phẩm phổ biến, có trên 20 tương tác ($count_{train} > 20$).

Các mô hình CF cổ điển (Matrix Factorization, BPR-MF, SVD) không có embedding đã học cho item zero-shot và rất dễ bị nhiễu trên các item ít tương tác, dẫn đến sai lệch khuyến nghị nghiêm trọng.

#### 1.2. Câu hỏi nghiên cứu và phạm vi đóng góp

TF-IDF tận dụng từ khóa trong mô tả và là đối chứng nội dung; nó không dùng ảnh. Nhánh ID và Transformer học từ chuỗi tương tác, trong khi CLIP cung cấp đặc trưng văn bản/ảnh đã được tiền huấn luyện. Nghiên cứu kiểm tra việc phối hợp hai nguồn có cải thiện chất lượng theo từng mức train count hay không.

Đóng góp triển khai của nhóm gồm adapter đa phương thức, cơ chế availability mask, hai head proxy độ bất định và fusion có cross-alignment; CLIP, Transformer và BPR là các thành phần kế thừa. Đây là thiết kế thực nghiệm trong đồ án, chưa có cơ sở tuyên bố một phương pháp mới vượt các mô hình hiện có. Giả thuyết ID ít tương tác làm giảm chất lượng cần đối chứng để kiểm chứng, không suy ra chỉ từ việc dùng ID embedding.

Mục tiêu thực nghiệm là Cold macro NDCG@10, đồng thời báo ảnh hưởng ở warm, overall, calibration và tài nguyên. Người dùng không history dùng fallback; chưa tuyên bố cá nhân hóa người dùng mới.

---

### CHƯƠNG 2: KIẾN TRÚC MÔ HÌNH UFM-REC

Mô hình UFM-Rec được thiết kế gồm 4 khối thành phần chính:

Luồng dữ liệu đúng theo `src/ufm_model.py`:

1. Văn bản và ảnh của history/candidate qua CLIP đóng băng, sau đó adapter và chuẩn hóa L2.
2. Hồ sơ ngữ nghĩa người dùng là trung bình các vector ngữ nghĩa history khả dụng, rồi chuẩn hóa; nhánh này không đi qua Transformer.
3. ID history hợp lệ cộng positional embedding đi qua Transformer causal; lấy trạng thái ở vị trí quan sát cuối, cùng ID candidate để tạo CF pair.
4. Semantic pair kết hợp hồ sơ ngữ nghĩa với candidate và modality masks; CF pair kết hợp vector người dùng/candidate với độ dài history và train count.
5. Hai head Softplus tạo reliability proxy; masked fusion và cross-alignment đưa vào recommendation head. Nếu cả hai nhánh thiếu tín hiệu, dùng popularity train-only.


#### 2.1. Nhánh Biểu Diễn Ngữ Nghĩa (Semantic Branch)
- Đặc trưng trích xuất từ **CLIP ViT-B/32** (khóa cố định commit revision): vector văn bản $e_{text} \in \mathbb{R}^{512}$ và hình ảnh $e_{img} \in \mathbb{R}^{512}$ được chuẩn hóa $\ell_2$.
- Khi một phương thức bị thiếu (ví dụ sản phẩm không có ảnh), vector đó được zero-fill trước khi qua adapter và được đánh dấu bằng cờ nhị phân $m \in \{0, 1\}^2$ thay vì chèn nhiễu ngẫu nhiên.
- Bộ điều hợp Semantic Adapter đưa đặc trưng về không gian tiềm ẩn $d = 128$:
  $$v_{sem} = \text{LayerNorm}(\text{Linear}(\text{GELU}(\text{Linear}([e_{text} \odot m_{text}, e_{img} \odot m_{img}, m]))))$$

#### 2.2. Nhánh Lịch Sử Chuỗi Hành Vi (Collaborative Branch)
- Chuỗi $H = [i_1, i_2, \dots, i_T]$ (với $T = 20$) được đưa qua mạng **Causal Transformer Encoder** 2 lớp, 4 attention heads với mặt nạ nhân quả (causal mask) và đệm phải (right padding).
- Điểm mấu chốt: Các item ID chưa từng xuất hiện trong tập Train ($count_{train} = 0$) được gán trực tiếp về chỉ số Padding ($idx=0$) **trước khi lookup** để tránh dùng embedding ID chưa được học và ngăn gradient vào ID không khả dụng.

#### 2.3. Cổng Ước Lượng Độ Tin Cậy (Uncertainty-aware Fusion Gate)
Thay vì dùng cổng tĩnh hoặc softmax tự do, UFM-Rec định nghĩa đại lượng proxy độ bất định học được qua hàm kích hoạt dương:
$$u_{cf} = \text{Softplus}(\text{MLP}_{cf}(h_{cf})), \quad u_{sem} = \text{Softplus}(\text{MLP}_{sem}(h_{sem}))$$

Trong đó mỗi nhánh học khớp phần dư bình phương có điều kiện trên tập ứng viên lấy mẫu ($\text{MSE}(u_k, (\sigma(s_k)_{\text{detached}} - y)^2)$). Trọng số độ tin cậy được tính theo công thức:
$$w_k = \frac{\mathbb{I}(k\in A)\exp(-u_k)}{\sum_{j\in A}\exp(-u_j)},\quad A=\{j:\text{branch }j\text{ available}\},\ A\ne\varnothing$$

Nếu cả hai nhánh không khả dụng, mô hình dùng fallback popularity train-only. Khi nhánh Semantic khả dụng và candidate là item zero-shot ($count_{train}=0$), nhánh CF không khả dụng ($\mathbb{I}_{cf}=0$), cổng tự động triệt tiêu $w_{cf} = 0$ và chuyển toàn bộ 100% trọng số sang $w_{sem} = 1,0$.

#### 2.4. Căn Chỉnh Đa Phương Thức (Cross-Modal Alignment Gate)
Khi cả hai nhánh đều khả dụng, thành phần Alignment đóng vai trò cầu nối liên phương thức:
$$g_{align} = \sigma(\text{MLP}_{gate}([h_{cf}, h_{sem}])) \odot \text{Linear}([h_{cf}, h_{sem}]) \cdot \mathbb{I}_{cf \wedge sem}$$
Vector tổng hợp cuối cùng:
$$h_{fused} = w_{cf} h_{cf} + w_{sem} h_{sem} + g_{align}$$
$$\hat{y} = \text{MLP}_{out}(h_{fused})$$

---

### CHƯƠNG 3: THIẾT LẬP THỰC NGHIỆM & GIAO THỨC ĐÁNH GIÁ

#### 3.1. Tập Dữ Liệu
Nghiên cứu sử dụng tập dữ liệu thực tế quy mô lớn từ Amazon Reviews 2023:
- Danh mục chính: **Toys & Games** ([`data/processed/toys_games_full_temporal/split_manifest.json`](../data/processed/toys_games_full_temporal/split_manifest.json))
- Tổng số sản phẩm (Items): **767.045**
- Số sản phẩm có ảnh hợp lệ trích xuất thành công: **758.692** (98,91%)
- Tổng số tương tác sau làm sạch: **11.572.689**
- Chiến lược phân chia: **Global Temporal Split (80% Train, 10% Validation, 10% Test)** theo mốc thời gian. Chia split theo thời gian không tự bảo đảm mọi bước lấy mẫu và metadata chỉ dùng thông tin có sẵn tại thời điểm dự đoán; giới hạn được nêu ở mục 3.2.
- Danh mục thứ hai (chuẩn bị cho kiểm chứng đa miền): **All Beauty** gồm **86.014 sản phẩm, 445.239 tương tác** ([`reports/second_dataset_preparation.json`](../reports/second_dataset_preparation.json)).

#### 3.2. Giao thức Đánh giá Phân tầng
Mỗi bộ mẫu Toys có 40.000 ca, mỗi ca gồm 1 target ở cột 0 và 99 negative duy nhất, được chia thành 10.000 ca cho mỗi regime theo **số tương tác train của target**: zero-shot = 0; extreme-cold = 1–5; cold = 6–20; warm >20. Các ca có thể thuộc cùng người dùng, vì vậy không gọi chúng là các quan sát độc lập. Cold macro là trung bình NDCG@10 của ba nhóm lạnh; Overall là trung bình trên bộ mẫu cân bằng, không đại diện trực tiếp cho phân bố lưu lượng tự nhiên.

**Cần phân biệt ba phiên bản ứng viên**:
- **Bộ lịch sử** dùng trong audit `ufm_validation_audit_v2`: hash mẫu validation `8efa116508fddd6b5227b69d6c65e3b635de634dd542046463fda84e78b47162`. Tên “audit v2” là phiên bản báo cáo kiểm toán, không có nghĩa nó dùng bộ mẫu mới `evaluation_v2`. Số liệu được giữ trong audit lịch sử, không trộn vào bảng v3 ở mục 4.1.
- **Bộ mẫu `evaluation_v2`** nhập local ngày 27/10: hash validation `c71906e65ce4067b405a94c13e22e8a25a17f46cbee5b1315f07d01d4d1e59df`, test `a4422c2cfe420e7ffc072915b16ff88a218d640054fb107c937d867dca20b109`. Candidate audit báo replay và tính nhất quán regime đạt. Code hiện tại lấy 99 negative **cùng regime với target**, loại target và các item trong history được ghi nhận, lấy trọng số popularity bằng `train_count + validation_count` khi đánh giá validation, và cộng thêm `test_count` khi đánh giá test. Đây là benchmark hồi cứu có dùng thống kê đến cuối split để dựng tập ứng viên, **không phải negative sampling chỉ từ train** và không phải đánh giá full-catalog.

- **Validation v3 hiện dùng để phân tích**: hash `9387e66e9f1905c44d02b65c3809f5fa8b6c022f6abf0c1a37d52cb57767a60e`, loại mọi positive có timestamp trước target, kể cả ngoài history window. Pool/weights vẫn hồi cứu theo regime. Tám checkpoint UFM, popularity và TF-IDF dùng cùng candidates. Checkpoint chọn bằng validation v1; v3 không tạo holdout độc lập.

Việc dùng thống kê cuối split và biết regime của target để chọn negative giới hạn khả năng diễn giải như đánh giá online. History được ghi nhận có cửa sổ hữu hạn; audit cũ ghi 3 trường hợp negative trùng positive train ở validation và 7 ở test. Audit validation mới trên FITLAB-01 xét mọi sự kiện trước target, bao gồm cùng split, ghi 7 negative trong 6 ca của v2. Validation v3 tạo riêng đã loại các known positive và audit đạt; tám checkpoint UFM cùng popularity/TF-IDF đã được đánh giá trên đúng hash v3. Không diễn giải `passed=true` thành “không có false negative”. Mỗi bảng so sánh phải dùng cùng bộ ứng viên.

Metric báo cáo: NDCG@10, Recall@10, MRR@10; **Cold macro NDCG@10 là chỉ số chính**. Nguồn: [`src/build_recommender_samples.py`](../src/build_recommender_samples.py), [artifact và hash v2](../reports/ufm_evaluation_v2_20261027/README.md).

---

### CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM, TÀI NGUYÊN & GIỚI HẠN NGHIÊN CỨU

#### 4.1. Kết quả chính trên validation v3

Cùng 40.000 ca, 100 ứng viên/ca và candidate hash nêu ở mục 3.2. Tám checkpoint UFM thuộc một seed 42; chỉ số mean/std qua seed chưa có. Các cột Overall, Recall và MRR tính trên 40.000 ca cân bằng. Cold macro là trung bình NDCG của ba nhóm lạnh.

| Mô hình | Cold macro NDCG@10 | Overall NDCG@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0,195682 | 0,181505 | 0,281275 | 0,151050 |
| Semantic-only | 0,190036 | 0,166404 | 0,268500 | 0,135315 |
| No cross-align | 0,183393 | 0,164897 | 0,262100 | 0,135189 |
| No uncertainty | 0,182253 | 0,163858 | 0,262900 | 0,133661 |
| UFM full | 0,182056 | 0,164136 | 0,262300 | 0,134202 |
| Text-only | 0,181982 | 0,158219 | 0,255275 | 0,128593 |
| Fixed fusion | 0,181305 | 0,163285 | 0,262475 | 0,133013 |
| Image-only | 0,149632 | 0,135772 | 0,231375 | 0,106708 |
| Collaborative-only | 0,070003 | 0,069681 | 0,142450 | 0,047936 |
| Popularity | 0,042527 | 0,040904 | 0,090900 | 0,026106 |

Nguồn: [dự đoán và metric v3](../reports/continuation_20261028/validation_v3_results.json), [bản tổng hợp có hash nguồn](../reports/nontraining_20261029/evidence.json). TF-IDF dùng 10 history item, UFM dùng 20; đây là so sánh hệ thống. Các ablation UFM dùng cùng CLIP/history và ngân sách lịch sử đã được audit.

![Cold macro NDCG trên validation v3](../output/figures/current_v3_20261029/cold_macro.png)

Full thấp hơn TF-IDF 0,013626 và semantic-only 0,007980 Cold macro NDCG@10. No uncertainty và no cross-align cũng có cold macro nhỉnh hơn full ở seed này; chưa xác nhận lợi ích của hai thành phần đó đối với chỉ số chính. Full chỉ cao hơn fixed fusion khoảng 0,000751; một seed chưa cho biết chênh lệch này ổn định hay không.

SASRec/BERT4Rec đã có kết quả lịch sử v1 được audit, chưa có điểm v3 đã xác minh, nên không ghép vào bảng này. Bảng v1 và BPR-MF đa seed được giữ trong [audit lịch sử](../reports/ufm_validation_audit_v2/report.md) và [báo cáo BPR-MF](../reports/BPR_MF_multiseed_2026_09_30.json).

#### 4.2. Phân tích theo số tương tác train

| Train count | Số ca | Full | Semantic-only | Full trừ semantic |
| --- | ---: | ---: | ---: | ---: |
| 0 | 10,000 | 0,156655 | 0,159080 | -0,002424 |
| 1-5 | 10,000 | 0,191567 | 0,215668 | -0,024101 |
| 6-20 | 10,000 | 0,197946 | 0,195360 | 0,002586 |
| 21+ | 10,000 | 0,110375 | 0,095509 | 0,014865 |

![Chênh lệch full và semantic-only theo train count](../output/figures/current_v3_20261029/count_effect.png)

Nhóm 1–5 có mức giảm lớn nhất. Trong 5.247 ca của nhóm này có history, full đạt 0,329435, semantic-only 0,375367; chênh lệch là −0,045933. Với 4.753 ca không history, hai checkpoint cho cùng điểm do fallback. Nguồn: [phân tích predictions đã lưu](../reports/continuation_20261028/count_effect_v3.json).

Kết quả hỗ trợ việc ưu tiên kiểm chứng fusion ở nhóm cực lạnh, nhưng chưa xác định ID embedding hay gate là nguyên nhân. Còn thiếu phân phối uncertainty/sai số trên checkpoint thật, nhiều seed và kết quả các can thiệp được train riêng. Trọng số nhánh trung bình trên toàn bộ ca cũng bị ảnh hưởng bởi các ca thiếu tín hiệu; không đọc nó như mức tin cậy có điều kiện khi cả hai nhánh khả dụng.

Ưu thế warm của full so với semantic-only trên v3 là 0,014865. Điều này không đồng nghĩa full vượt TF-IDF ở warm v3: full 0,110375, TF-IDF 0,138976. Kết luận warm cao trên v1 không được chuyển sang v3.

#### 4.3. Calibration thăm dò trên validation v3

Temperature scaling dùng T = 2,012405, fit trên 20.000 ca sớm và audit trên 20.000 ca muộn, giữ các timestamp bằng nhau cùng phía.

| Chỉ số trên phần audit | Trước | Sau |
| --- | ---: | ---: |
| ECE | 0,082061 | 0,037478 |
| NLL | 4,497667 | 4,258086 |
| Multiclass Brier | 0,969342 | 0,960605 |

Nguồn: [summary v3](../reports/continuation_20261028/summary.json) và [reliability diagram](../reports/continuation_20261028/reliability_full_v3.png). Các số trên cùng thuộc v3; số Brier 0,963379 → 0,927849 của audit cũ không được dùng ở đây.

Đây là confidence Top-1 có điều kiện trên 100 ứng viên, không phải xác suất mua hàng. Checkpoint đã dùng validation để chọn và hai phần có thể chung user nên kết quả mang tính thăm dò. T > 0 giữ thứ hạng, không làm tăng NDCG; ECE giảm chưa chứng minh hai uncertainty head cải thiện ranking. Phương pháp temperature scaling tham khảo Guo và cộng sự (2017), xem tài liệu tham khảo.

#### 4.4. Demo và tài nguyên vận hành

Backend UFM thật trên Windows đã nạp checkpoint full epoch 9, catalog 767.045 sản phẩm và counts thật. Ngày 29/10, sáu ca API đạt 6/6 và bốn ca UI Chrome headless đạt 4/4, gồm zero-shot, tải JSON, ảnh lỗi và mất mạng; [receipt UI](../reports/nontraining_20261029/demo_ui_end_to_end_20261029.json) lưu request, model và kết quả. Thiếu text hoặc thiếu ảnh vẫn được xử lý bằng modality mask. Ảnh JPEG trên UI phục vụ hiển thị; suy luận dùng CLIP embedding đã cache, nên ảnh hiển thị lỗi không tự đổi điểm. Cần thử lại trên máy bảo vệ.

Bản đo HTTP Windows ngày 28/10 dùng hai warm-up và 10 request, history một item, K=5: p50 **21,25 giây**, p95 **26,35 giây**, RSS cao nhất quan sát **2.551,5 MiB**; QPS tuần tự khoảng **0,0457**. Hai request đồng thời trả một 200 và một 429 vì server cho một suy luận mỗi lần. Đây là số của một máy/một cấu hình; RSS quan sát không phải peak RAM train. Nguồn: [benchmark Windows](../reports/continuation_20261028/windows_ufm_benchmark.json).

Mẫu 10 request chưa đủ mô tả p99 ổn định. Số 2,89–7,27 giây trên FITLAB ngày 03/10 là từng request đơn; QPS trên catalog content 30 item thuộc demo mẫu. Không dùng hai loại số đó làm benchmark hiện hành của UFM toàn catalog.

Lượt train full lịch sử mất khoảng 4,13 giờ trên cấu hình FITLAB đã ghi nhận, không phải dự báo thời gian Colab. Chưa có bản đo peak VRAM của lượt train đủ để chứng minh cam kết GPU 16 GiB. Đồ án không huấn luyện thêm sau các checkpoint này; bằng chứng nghiệm thu suy luận được lưu trong [hồ sơ ngày 29/10](../reports/nontraining_20261029/status.md).

#### 4.5. Kết quả baseline sơ bộ trên All Beauty
Tập thứ hai đã được chuẩn bị độc lập theo split thời gian 80/10/10. Manifest ghi nhận **445.239 tương tác dương**, trong đó 356.402 train, 44.396 validation và 44.441 test; catalog có **86.014 sản phẩm**. Test chưa được đánh giá. Kết quả dưới đây là baseline TF-IDF và popularity trên validation, không phải kết quả UFM hoặc đánh giá chuyển miền của mô hình đề xuất.

| Nhóm item | Số ca | Popularity NDCG@10 | TF-IDF NDCG@10 | Popularity Recall@10 | TF-IDF Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 10.000 | 0,000000 | 0,027645 | 0,000000 | 0,039500 |
| Extreme cold | 10.000 | 0,001102 | 0,020921 | 0,003700 | 0,030300 |
| Cold | 7.381 | 0,246404 | 0,246271 | 0,673621 | 0,652486 |
| Warm | 10.000 | 0,770103 | 0,742460 | 1,000000 | 0,967500 |
| **Overall** | **37.381** | **0,254963** | **0,260239** | **0,401514** | **0,406329** |

TF-IDF tăng overall NDCG@10 thêm 0,005276 so với popularity; mức tăng tập trung ở zero-shot và extreme-cold, trong khi cold gần như ngang nhau và warm thấp hơn popularity. Chỉ **2.627/37.381** ca validation có history, còn 34.754 ca dùng history rỗng; trên nhóm có history, TF-IDF đạt NDCG@10 0,278014 so với popularity 0,202934. Đây là baseline tham chiếu hữu ích cho các lượt chạy tiếp theo, nhưng chưa đủ để kết luận UFM khái quát sang miền All Beauty.

Giao thức lấy mẫu All Beauty có 100 ứng viên/ca, lấy negative uniform không hoàn lại; nhóm cold gồm toàn bộ 7.381 ca sẵn có, còn các nhóm khác lấy tối đa 10.000 ca. Giao thức này khác Toys (40.000 ca cân bằng và popularity negative sampling), vì vậy không so sánh trực tiếp metric giữa hai dataset. Nguồn: [`sampling_report.json`](../data/processed/all_beauty_full_temporal/evaluation/sampling_report.json), [`metrics.json`](../data/processed/all_beauty_full_temporal/models/content_baseline/metrics.json), [`split_manifest.json`](../data/processed/all_beauty_full_temporal/split_manifest.json).

#### 4.6. Giới hạn của nghiên cứu

1. UFM hiện có một seed đã xác minh. Bootstrap theo user không thay thế biến thiên giữa các lượt huấn luyện.
2. Test v2 đã được xem; v3 không phải holdout độc lập. Kết luận trên v3 mang tính thăm dò.
3. Negative sampling có tính hồi cứu theo regime target, khác full-catalog/online ranking. Metadata là snapshot; chưa chứng minh mọi text/ảnh tồn tại đúng tại thời điểm tương tác. CLIP có giới hạn text 77 token.
4. Hai can thiệp theo train count và UFM trên All Beauty chưa được huấn luyện trong phạm vi đồ án. Không khẳng định fusion, uncertainty hoặc adapter vượt đối chứng hay tiết kiệm tài nguyên ở mức đã đo.
5. Bộ kiểm thử tự động đạt 119 pass, 3 skip và 10 subtests (03/11/2026). Test phần mềm kiểm tra tính đúng của pipeline, không đo chất lượng recommendation.
6. Launcher tự chọn UFM khi có checkpoint hoàn tất hoặc content khi thiếu checkpoint. Demo UFM đạt API 6/6 và UI tự động 4/4 trên một máy Windows, gồm export JSON; độ trễ trên máy khác có thể khác. Bằng chứng nằm trong [hồ sơ nghiệm thu](../reports/nontraining_20261029/status.md).

#### 4.7. Kết quả UFM trên bộ ứng viên evaluation_v2
Nguồn: [`reeval_ufm_full_v1_v2.json`](../reports/ufm_evaluation_v2_20261027/reeval_ufm_full_v1_v2.json), checkpoint epoch 9, SHA-256 `5630013d0cbb6db633c4f4cf5bcd840fb703dbfc64a13c0352278624e03e11c8`. Đã kiểm tra hash mẫu/metadata và xác minh trực tiếp checkpoint, nguồn huấn luyện, cache trên FITLAB-01. Không chạy lại toàn bộ suy luận UFM trong lần cập nhật này.

| Nhóm | Số ca mỗi split | Validation NDCG@10 | Test NDCG@10 |
| --- | ---: | ---: | ---: |
| Zero-shot (0) | 10.000 | 0,156641 | 0,157281 |
| Extreme cold (1–5) | 10.000 | 0,191567 | 0,195261 |
| Cold (6–20) | 10.000 | 0,197946 | 0,198373 |
| Warm (>20) | 10.000 | 0,110375 | 0,113859 |
| **Cold macro** | 30.000 | **0,182051** | **0,183638** |
| **Overall** | 40.000 | **0,164132** | **0,166193** |

V2 có giao thức và hash bộ mẫu khác bảng 4.1, vì vậy **không so trực tiếp điểm v2 với TF-IDF/BPR-MF/SVD lịch sử**, không kết luận extreme-cold đã được sửa, và không gán mức tăng/giảm giữa hai bộ mẫu cho thay đổi mô hình. Baseline mới trên cùng validation v2 đã có: **TF-IDF Overall/Cold macro 0,181498 / 0,195672; popularity 0,040904 / 0,042527**. UFM tương ứng **0,164132 / 0,182051**, thấp hơn TF-IDF. TF-IDF dùng 10 history item, UFM dùng 20; đây chưa phải ablation kiểm soát chỉ một thành phần. Nguồn: [kết quả sửa mục 1–15](../reports/fix_1_15_20261027/README.md). Test đã được đánh giá; không tiếp tục dùng cách diễn đạt “chưa mở test”.

---

### CHƯƠNG 5: CÁC ĐỀ XUẤT VÀ HƯỚNG MỞ RỘNG

#### 5.1. Hai can thiệp theo train count

`count_aware_fusion` bổ sung penalty vào proxy CF:

$$u_{cf} \leftarrow u_{cf} + \exp(-count_{train}/5)$$

Biến thể này giữ ID dropout cố định như full. Nó giảm trọng số CF tương đối khi cả hai nhánh khả dụng, không đặt một ngưỡng cứng ép semantic thắng ở mọi item có dưới năm tương tác.

`adaptive_dropout` là biến thể riêng: thay xác suất bỏ ID trong train theo tần suất, không cộng count penalty:

$$p_{drop}(i) = p_0 \exp(-count_{train}(i)/10)$$

Hai thay đổi đã có code và kiểm thử invariant nhưng chưa được huấn luyện trong phạm vi đồ án, nên chưa có kết quả chất lượng. Kế hoạch kiểm chứng đã được khóa trong [manifest](../configs/multimodal_followup_20261028.json): năm biến thể (full, semantic-only, fixed-fusion và hai can thiệp) × ba seed 42, 7, 2026, cùng ngân sách và quy tắc chọn checkpoint; [notebook Colab](../notebooks/UFM_Colab_Browser_Train.ipynb) hỗ trợ lưu và resume theo đoạn. Kết quả không cải thiện vẫn cần công bố.

#### 5.2. Hiệu năng và dataset thứ hai

Nếu độ trễ chưa đáp ứng kịch bản, có thể khảo sát retrieval rồi UFM reranking. Phải đo recall của retrieval, NDCG đầu cuối và latency; scoring UFM phụ thuộc candidate nên ANN không thay tương đương toàn bộ model. Đây là đề xuất chưa triển khai/nghiệm thu.

All Beauty đã có dữ liệu và baseline tham chiếu, chưa có UFM xác minh. Chỉ ưu tiên campaign bổ sung khi giảng viên giữ yêu cầu hai dataset. Train riêng từng dataset là kiểm chứng trên hai miền, chưa chứng minh transfer learning. Encoder khác, LoRA, diversity/novelty và cá nhân hóa người dùng mới là các hướng có điều kiện theo phạm vi.

---

### CHƯƠNG 6: KẾT LUẬN

Nhóm đã triển khai hệ gợi ý dùng lịch sử hành vi, CLIP text/image đóng băng và fusion có reliability proxy; có checkpoint, đối chứng/ablation một seed, đánh giá cùng candidates, calibration thăm dò và demo UFM thật.

Trên validation v3, full đạt Cold macro NDCG@10 0,182056, thấp hơn TF-IDF 0,195682 và semantic-only 0,190036. Nhóm 1–5 tương tác là nơi full giảm nhiều nhất so với semantic-only. Kết quả chưa chứng minh cải tiến ranking từ fusion/uncertainty; đây là phát hiện cần kiểm chứng tiếp, không được thay bằng kết luận thắng trên một giao thức khác.

Calibration v3 giảm ECE từ 0,082061 xuống 0,037478 và Brier từ 0,969342 xuống 0,960605; không thay thứ hạng và không phải xác suất mua. Demo toàn catalog chạy được trên CPU nhưng độ trễ còn cao. Các tuyên bố về nhiều seed, cải tiến mới, peak VRAM 16 GiB và UFM All Beauty vẫn chờ bằng chứng.

Đồ án chốt kết quả với các checkpoint và bằng chứng hiện có. Hướng phát triển theo thứ tự ưu tiên: chạy đủ 15 lượt đa seed đã khóa để đo độ ổn định và tác dụng của hai can thiệp theo train count; phân tích gate/uncertainty trên nhóm 1–5 tương tác; đánh giá trên một holdout độc lập được thống nhất trước; giảm độ trễ demo bằng retrieval rồi rerank; mở rộng UFM sang All Beauty nếu cần kiểm chứng trên miền thứ hai.

### Tài liệu tham khảo

1. Radford và cộng sự (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). Nguồn CLIP; đề tài dùng encoder đóng băng, không tuyên bố đã fine-tune CLIP.
2. Kang và McAuley (2018). [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). Tham chiếu SASRec; bản repo là thích nghi, không gọi là reproduction chuẩn.
3. Sun và cộng sự (2019). [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690). Tham chiếu Cloze/bidirectional; nhánh UFM dùng causal Transformer khác mục tiêu này.
4. Rendle và cộng sự (UAI 2009; bản arXiv 2012). [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618). Tham chiếu objective ranking từ implicit feedback.
5. Guo và cộng sự (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). Tham chiếu temperature scaling; phạm vi áp dụng trong đồ án là sampled candidates.
6. McAuley Lab. [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), gắn với Hou và cộng sự (2024), Bridging Language and Items for Retrieval and Recommendation. Quy mô sau xử lý trong báo cáo lấy từ manifest của repo, không đồng nhất với tổng số dòng thô trên trang dataset.
