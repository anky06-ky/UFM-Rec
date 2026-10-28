# Checklist hoàn thiện UFM Rec

<!-- current-evidence-20261028 -->
> Cập nhật 28/10/2026: validation v3 đã hoàn tất (40.000 ca, 100 ứng viên/ca, seed 42). Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only 0,190036; UFM full 0,182056; popularity 0,042527**. Tám checkpoint UFM dùng cùng candidate hash. Đây là đánh giá lại checkpoint chọn bằng validation v1, chưa phải holdout độc lập. Test v2 đã được xem; vòng v3 này không đánh giá test. Chi tiết và giới hạn: [bằng chứng mới](../reports/continuation_20261028/README.md). Các số liệu ở phần lịch sử phải đọc theo phiên bản giao thức ghi kèm. Word/PDF cũ chưa đồng bộ cập nhật này.
<!-- /current-evidence-20261028 -->


Đối chiếu ngày **28/10/2026**: bộ sửa 1–15 trước đây đạt Linux **87 pass** / Windows **84 pass, 3 skip**. Mã bổ sung calibration/source loader đạt Linux **94 pass** / Windows **91 pass, 3 skip**; các bổ sung restore sau đó có kiểm thử riêng. Xem báo cáo sửa mục 1–15 và log để phân biệt bản mã hiện tại với bản mã đã tạo checkpoint.

Cập nhật sau khi sửa mục 1–15 ngày 27/10/2026 trên FITLAB-01. Mô tả lỗi bên dưới giữ lại làm tiêu chí đối chiếu; checkbox phản ánh trạng thái sau sửa. Xem [bằng chứng và trạng thái từng mục](../reports/fix_1_15_20261027/README.md).

## Trạng thái hiện tại

- Checkpoint full và cache CLIP thực còn trên FITLAB-01, đã kiểm tra hash, mapping, mask, finite forward và metric từ predictions lưu.
- Đã chạy popularity/TF-IDF trên đúng validation v2. Cold macro NDCG@10: UFM **0,182051**, TF-IDF **0,195672**, popularity **0,042527**; chưa chứng minh UFM cải thiện chỉ số chính.
- V2 giữ nguyên; audit mới ghi 7 prior-positive negative trong 6 ca. V3 đã được tạo riêng và audit đạt chính sách loại mọi positive trước target; đã có điểm tám checkpoint UFM và baseline v3.
- Test local **84 passed, 3 skipped**, 10 subtests passed; ba skip dành cho Linux. Bản mã hiện tại được kiểm thử riêng trên FITLAB-01 để giữ nguyên nguồn tạo checkpoint.
- Demo TF-IDF có 30 ảnh local, synthetic counts được gắn nhãn trong UI/API/export; ảnh chỉ hiển thị ở backend này.
- **Mục 8 còn chờ GitHub Actions; mục 9 còn chờ giảng viên xác nhận phạm vi.** Không tích hoàn thành hai phần cần bằng chứng bên ngoài này. Word/PDF chưa xuất lại.

## P0 — Sửa lỗi và xác minh tài nguyên trước khi chạy thí nghiệm dài

- [x] **P0-01 — Kiểm kê workspace FITLAB hiện tại.** Kiểm tra file thực, heartbeat, tiến trình, GPU/RAM/disk và nơi lưu backup; phân biệt snapshot 03/10, bản ghi khôi phục 24/10 và artifact v2.
  Hoàn thành khi có một snapshot mới kèm thời điểm kiểm tra và danh sách file/tài nguyên, không chỉ lời mô tả từ log cũ.

- [x] **P0-02 — Khôi phục checkpoint và CLIP cache có nguồn gốc.** Tìm `best.pt`, `config.json`, `completed.json`, predictions, text/image tables, masks, train counts và code snapshot đúng lượt chạy. Đối chiếu SHA-256, mapping, shapes, padding và tensor hữu hạn. Chỉ lập kế hoạch train lại sau khi xác nhận không có bản khôi phục dùng được.
  Hoàn thành khi nạp được model đúng fingerprint và tái tạo được dự đoán/metric kiểm tra trên validation; không bỏ guard hash để ép nạp checkpoint khác code.

- [x] **P0-03 — Sửa hai lỗi fixture trong cold-start tests.** `setup_model` hiện không khả dụng cho lớp `TestBugConditionDocumentation` trong [test_cold_start_bug_condition.py](../tests/test_cold_start_bug_condition.py).
  Hoàn thành khi test chạy tới logic cần kiểm tra, không lỗi collection/setup.

- [x] **P0-04 — Viết lại năm test luôn gọi `pytest.fail()`.** Bỏ kết luận về chất lượng NDCG dựa trên mô hình chưa huấn luyện; thay bằng kiểm tra invariant có ý nghĩa. Các giả thuyết về trọng số CF phải được kiểm chứng trên validation/checkpoint thật. Chỉ dùng xfail nếu có bug xác định, điều kiện tái hiện và lý do cụ thể; không dùng để che lỗi.
  Hoàn thành khi test phản ánh hành vi mong muốn đã thống nhất, có thể phân biệt code đúng/sai và không tự báo “BUG CONFIRMED” bất kể đầu ra.

- [x] **P0-05 — Sửa hai test/sample-builder lỗi đường dẫn.** `build_split(..., output=OUTPUT)` giữ default từ lúc định nghĩa hàm, trong khi fixture chỉ monkeypatch `OUTPUT`. Truyền output rõ ràng hoặc giải quyết default tại lúc gọi; bảo đảm test chỉ ghi vào thư mục tạm.
  Hoàn thành khi hai test hồi quy chạy được và không ghi đè bộ mẫu thật hoặc để lại output hoàn tất một phần.

- [x] **P0-06 — Sửa và xác minh candidate audit.** Trong [test_candidate_validity.py](../src/test_candidate_validity.py), `past_positive_check` lấy start/end từ cùng vị trí nên slice history luôn rỗng; rà lại mốc thời gian khi gọi “past positive”, việc dùng `--data`, và replay khi số ca thấp hơn cap hoặc có lọc min-interactions. Việc sửa slice không tự chứng minh các chỉ số disclosure cũ sẽ thay đổi; phải chạy đối chiếu.
  Hoàn thành khi audit có kiểm tra bằng fixture tính tay, đọc đúng dataset truyền vào và công bố lại kết quả từ logic đã kiểm tra.

- [x] **P0-07 — Sửa ghi nhận lịch sử đánh giá test.** [run_final_test.py](../src/run_final_test.py) kiểm tra marker cũ và file output, nhưng chưa có sổ ghi nhận đầy đủ lượt test v2; đổi output có thể tránh guard hiện tại. Ghi theo dataset/candidate hash/checkpoint/code/scorer/temperature và thời điểm đánh giá; không mặc định `first_evaluation=true` khi chưa có bằng chứng.
  Hoàn thành khi trạng thái test trong tool và tài liệu cùng phản ánh lượt đã xảy ra, không gọi test đã xem là test chưa mở.

- [ ] **P0-08 — Ổn định môi trường và CI.** Xác định Python/dependency thực sự hỗ trợ, phân biệt môi trường CPU/GPU; cấu hình discovery để không thu nhầm `tmp/transfer/verify/test_*.py`. Cập nhật tên job/số suite đã cũ; chạy CI với dữ liệu nhỏ độc lập, công bố lý do skip.
  Hoàn thành khi `pytest tests/` và CI đạt với các test cần thiết; không bỏ test hợp lệ để làm xanh CI. Lưu lại phiên bản môi trường của từng lượt chạy.

## P1 — Chốt dữ liệu và cách đánh giá

- [x] **P1-01 — Chốt phạm vi với giảng viên.** Theo xác nhận của nhóm ngày 28/10, phạm vi là hệ gợi ý sản phẩm kết hợp hành vi, hình ảnh và tên/mô tả. Cold-start sản phẩm là phân tích trọng tâm; no-history hiện là fallback. Nhóm đã khóa Cold macro NDCG@10, ba seed và đối chứng trong manifest; đây là lựa chọn thí nghiệm của nhóm, chưa gán cho giảng viên xác nhận metric/dataset thứ hai.
  Hoàn thành khi có phạm vi nghiệm thu cụ thể, không đồng thời hứa giải quyết mọi dạng cold-start.

- [x] **P1-02 — Chốt candidate protocol.** Hiện v2 lấy negative cùng regime với target, popularity đến cuối split; đây là đánh giá hồi cứu có giới hạn. Thống nhất giữ protocol này với disclosure hay bổ sung đánh giá bám thời điểm dự đoán. Làm rõ catalog khả dụng và negative của item zero-shot; không đơn giản loại mọi item có train count bằng 0.
  Hoàn thành khi có mô tả và config cố định về positive, negative pool, weights, availability, exclusion và seeds; tất cả mô hình dùng đúng các file/hash đã chốt.

- [x] **P1-03 — Kiểm tra lịch sử và positive đã biết.** Loại negative trùng positive đã biết trước thời điểm dự đoán, kể cả ngoài cửa sổ history; kiểm tra repeat purchase, sự kiện cùng timestamp và history trước target. Future positive có thể là nhiễu nhãn của implicit feedback: đo và công bố, không dùng nhãn test tương lai để tune/lọc validation mà vẫn gọi là đánh giá online.
  Hoàn thành khi có audit theo thời gian và chính sách xử lý false negative rõ ràng, không dùng `passed=true` thay cho disclosure.

- [x] **P1-04 — Kiểm tra dữ liệu đa phương thức.** Đối chiếu ASIN/row index giữa interactions, text, ảnh, features và counts; thống kê missing text/image, URL hỏng, download/decode lỗi theo từng regime. Nêu giới hạn metadata snapshot và cắt text 77 token.
  Hoàn thành khi các bảng khớp mapping, có mask modality đúng và coverage report của chính cache được dùng.

- [x] **P1-05 — Phân biệt dữ liệu demo và benchmark.** Giữ 30 sản phẩm/ảnh mẫu làm luồng minh họa; dữ liệu thực nghiệm dùng train counts và split thật. Không đưa synthetic counts, ảnh minh họa hoặc trạng thái người dùng chọn trong demo vào báo cáo benchmark.
  Hoàn thành khi UI, API export và báo cáo xác định được chế độ dữ liệu; mỗi số liệu truy được đúng nguồn.

- [x] **P1-06 — Khóa manifest và cách xử lý test đã xem.** Lưu dataset/mapping/features/candidates/code/checkpoint/calibration hash cùng selection rule. Test v2 đã có điểm; công bố điều đó và xin góp ý thiết kế đánh giá tiếp theo. Nếu cần tập kiểm tra mới, phải có cơ sở độc lập phù hợp; chia lại dữ liệu đã xem không tự tạo ra holdout độc lập.
  Hoàn thành khi có protocol được chốt trước vòng so sánh tiếp theo, không chọn cấu hình theo điểm test đã biết và không ghép baseline cũ vào v2.

## P2 — Hoàn thiện bằng chứng mô hình

- [x] **P2-01 — Chạy baseline trên cùng validation đã chốt.** Ít nhất popularity và TF-IDF; bổ sung BPR-MF/SVD theo phạm vi. Kiểm tra predictions/ranks và hash candidates để so công bằng với UFM.
  Hoàn thành khi có một bảng so sánh cùng dữ liệu, metric và protocol; không chỉ đặt cạnh nhau các JSON từ hai bộ ứng viên khác nhau.


<!-- verified-P2-02-20261028 -->
Bằng chứng ngày 28/10: Semantic-only và fixed-fusion cùng CLIP; audit v1 xác minh code/budget; tám UFM checkpoint đã so trên v3, một seed.

- [x] **P2-02 — Có đối chứng dùng cùng CLIP features.** Thêm semantic-only, fixed fusion hoặc concat để xác định lợi ích của gate thay vì lợi ích từ encoder pretrained. Kiểm soát objective, budget, history và tuning; concat hiện dùng objective khác UFM nên phải nêu và kiểm tra confound này.
  Hoàn thành khi có ít nhất một đối chứng đủ sát kiến trúc để đánh giá phần đề xuất.


<!-- verified-P2-03-20261028 -->
Bằng chứng ngày 28/10: SASRec/BERT4Rec đã đối chiếu predictions và candidate v1; chưa có kết quả tuần tự v3.

- [x] **P2-03 — Đối chứng tuần tự theo cam kết.** Hoàn thiện SASRec và BERT4Rec thích nghi nếu còn trong phạm vi báo cáo. Công bố khác biệt với bản gốc, sampled-softmax và cách xử lý unseen ID.
  Hoàn thành khi có run/config/predictions kiểm tra được, không coi custom Transformer là reproduction chuẩn chỉ vì dùng self-attention.

- [ ] **P2-04 — Chạy nhiều seed.** Kế hoạch hiện dùng `42, 7, 2026`; ưu tiên full UFM và baseline chính, cùng selection rule và budget. Chốt số run trước khi xem kết quả, không chọn seed đẹp.
  Hoàn thành khi báo mean ± sample std và từng seed; bootstrap user không thay thế training-seed variability.

<!-- verified-P2-05-20261028 -->
Bằng chứng ngày 28/10: JSON v3 có đủ regime/history/repeat, số ca và ba metric cho tám UFM + hai baseline. Audit v1 có paired user bootstrap.

- [x] **P2-05 — Báo đầy đủ các nhóm đánh giá.** NDCG@10, Recall@10, MRR@10 theo zero-shot/extreme-cold/cold/warm; có/không history; repeat/new purchase. Báo cold macro, overall trên mẫu cân bằng và nêu khác phân bố tự nhiên. Dùng paired bootstrap theo user khi phù hợp và ghi rõ phạm vi suy luận.
  Hoàn thành khi người đọc thấy mô hình tăng/giảm ở đâu và số ca từng nhóm, không chỉ có Overall.

<!-- verified-P2-06-20261028 -->
Bằng chứng ngày 28/10: Tám calibration fit/audit theo thời gian trên v3, ECE/NLL/Brier, biểu đồ reliability full và giới hạn thăm dò đã lưu.

- [x] **P2-06 — Kiểm chứng calibration trên đúng bộ ứng viên.** Fit temperature bằng phần validation đã định trước, vẽ reliability và báo ECE/NLL/Brier. Nhiệt độ cũ chỉ thuộc candidates cũ. Nếu validation đã dùng chọn checkpoint, gọi kết quả là thăm dò; không gọi xác suất trên 100 candidates là xác suất mua.
  Hoàn thành khi có config fit/audit, trước/sau và giới hạn rõ; không hứa calibration làm tăng NDCG.

## P3 — Cải tiến mô hình sau khi có đối chứng đáng tin

- [ ] **P3-01 — Kiểm chứng giả thuyết nhánh ID gây giảm extreme-cold.** Phân tích phân phối trọng số CF/semantic và chất lượng dự đoán theo train count trên checkpoint thật. So semantic-only với full trên cùng cases; tách can thiệp tại inference để chẩn đoán khỏi ablation được train lại.
  Hoàn thành khi có bằng chứng ủng hộ hoặc bác bỏ giả thuyết; không đồng nhất TF-IDF với nhánh Semantic CLIP của UFM.

- [ ] **P3-02 — Đánh giá `count_aware_fusion`.** Biến thể đã có code; chọn hệ số/hàm phạt theo validation, kiểm tra đồng thời cold và warm. Mức CF nhỏ hơn 0,1 không tự là tiêu chí đúng cho mọi sản phẩm.
  Hoàn thành khi có so sánh cùng seed/budget với full và fixed fusion, kể cả trường hợp không cải thiện.

- [ ] **P3-03 — Đánh giá `adaptive_dropout`.** So với ID dropout cố định; theo dõi sự phụ thuộc vào nội dung, cold macro và độ biến thiên qua seed. Tách tác dụng dropout khỏi count penalty.
  Hoàn thành khi có run đối chứng riêng cho từng thay đổi, không gộp nhiều thay đổi rồi quy công cho một cơ chế.

<!-- verified-P3-04-20261028 -->
Bằng chứng ngày 28/10: Bảy ablation đã train thực tế, source/cache/budget được audit, đánh giá cùng v3. Hoàn tất mức một seed; nhiều seed theo dõi tại P2-04.

- [x] **P3-04 — Ablation các thành phần đã đề xuất.** Các variant hiện có gồm `no_uncertainty`, `fixed_fusion`, `no_cross_align`, `semantic_only`, `collaborative_only`, `text_only`, `image_only`. Chốt ma trận tối thiểu với giảng viên, train lại các ablation cần đánh giá đóng góp học được.
  Hoàn thành khi chứng minh được vai trò hoặc giới hạn của ảnh, text, hành vi, uncertainty và alignment bằng kết quả thực, không chỉ forward/unit test.

- [ ] **P3-05 — Xử lý rõ các trường hợp thiếu tín hiệu.** Kiểm tra item zero-shot, history toàn unknown ID, không history, thiếu ảnh, thiếu text và thiếu cả hai. Xác nhận mask trước lookup, fallback train-only và finite output. Nếu muốn cá nhân hóa người dùng mới, cần thêm tín hiệu người dùng cung cấp và đánh giá riêng.
  Hoàn thành khi cơ chế fallback có kiểm thử và demo đúng ý nghĩa; chưa có dữ liệu sở thích thì không hứa cá nhân hóa.

## P4 — Đưa UFM thật vào demo và kiểm tra vận hành

Bằng chứng ngày 28/10: launcher chọn backend/readiness; sáu request UFM thật trả weights/uncertainty và checkpoint/cache provenance. UI zero-shot đã nghiệm thu trên FITLAB-01, catalog 767.045 item.

Thêm bằng chứng Windows ngày 28/10: nạp checkpoint từ bản sao local đã kiểm tra SHA, 6/6 ca API thật đạt; xem [receipt](../reports/continuation_20261028/windows_ufm_verification.json). Chưa tích P4-03 vì còn ca thiếu modality/ảnh lỗi và nghiệm thu giao diện local.

- [x] **P4-01 — Cho phép chọn backend UFM có kiểm tra readiness.** Nạp đúng checkpoint/cache đã xác minh; thêm lựa chọn content/UFM vào cách khởi động. Nếu thiếu tài nguyên, báo rõ lý do thay vì hiển thị nhãn UFM cho kết quả baseline.
  Hoàn thành khi một request thật chạy qua UFM, trả weights/uncertainty và có log tên model, dữ liệu, feature version.

- [x] **P4-02 — Demo trên catalog có train counts thật.** Dùng toàn catalog nếu tài nguyên đủ, hoặc subset có mapping/features đồng bộ. Giữ chế độ 30 sản phẩm minh họa riêng, có nhãn rõ và vẫn mở được offline.
  Hoàn thành khi chế độ dữ liệu kiểm tra được và hình ảnh/tên/ASIN khớp sản phẩm. Không dùng cache thiếu item để giả làm full-catalog.

- [ ] **P4-03 — Nghiệm thu các kịch bản đầu cuối.** Có history, không history, zero-shot, extreme-cold, cold, warm, thiếu modality, ảnh lỗi và loại item đã chọn. Kiểm tra tìm kiếm, Top-K, proxy và JSON export trên backend thật.
  Hoàn thành khi lưu được đầu vào/kết quả/model version cho từng ca; kiểm thử API không tự chứng minh chất lượng recommendation.

- [ ] **P4-04 — Đo hiệu năng trên dữ liệu đúng quy mô.** Đo nhiều request có warm-up, p50/p95/p99, throughput theo mức đồng thời, RAM RSS/peak, peak VRAM, thời gian train và số tham số trainable/frozen.
  Hoàn thành khi ghi cả phần cứng, backend, catalog size, history/K/batch và phạm vi đo; không suy peak từ một snapshot, không lấy QPS 30 item làm QPS catalog thật.

- [ ] **P4-05 — Hoàn thiện cách giải thích kết quả.** Hiển thị model, nguồn dữ liệu, train-count regime, điểm xếp hạng và trọng số nhánh khi UFM thật có trả về. Giải thích weights là tín hiệu mô hình, không phải bằng chứng nhân quả hoặc xác suất mua.
  Hoàn thành khi người xem hiểu ảnh đang để hiển thị hay tham gia suy luận và demo nào đang chạy.

## P5 — Báo cáo và bàn giao

- [ ] **P5-01 — Đồng bộ mọi con số và trạng thái từ nguồn duy nhất.** Báo cáo, slide, README, overview/proposal đang dùng, notebook và progress phải thống nhất dataset, candidates, seed, metric và thời điểm. Phân biệt lịch sử với trạng thái kiểm tra mới; cập nhật số test sau khi sửa lỗi.
  Hoàn thành khi mỗi số có đường dẫn/config/hash nguồn, không còn tuyên bố nguyên nhân hoặc kết quả vượt quá bằng chứng.

- [ ] **P5-02 — Xuất lại Word/PDF và kiểm tra trực quan.** Script Word đã được sửa một phần nhưng file Word/PDF cũ chưa đồng bộ. Rà cả các đoạn viết riêng trong script, công thức, bảng, mục lục, tên thành viên và tài liệu tham khảo; render và kiểm tra trang trước khi gửi.
  Hoàn thành khi bản gửi thầy cùng phiên bản với Markdown/slide, không chỉ đúng nội dung nguồn.

- [ ] **P5-03 — Làm rõ đóng góp và đối chiếu tài liệu liên quan.** Phân biệt thành phần kế thừa CLIP/Transformer/BPR với thiết kế fusion của nhóm. Đối chiếu nghiên cứu liên quan trước khi gọi là đóng góp mới; ghi count-aware/adaptive là đề xuất đang kiểm chứng cho tới khi có kết quả.
  Hoàn thành khi trả lời được mô hình thay đổi chỗ nào, tại sao cần, đối chứng gì và bằng chứng nào xác nhận hiệu quả.

- [ ] **P5-04 — Chuẩn bị buổi gặp/bảo vệ.** Trình bày bài toán, kiến trúc, kết quả cùng protocol, hạn chế và ba câu xin góp ý. Chuẩn bị demo UFM nếu sẵn sàng, demo baseline có nhãn và ảnh/video dự phòng đúng backend.
  Hoàn thành khi cả nhóm giải thích được cold-start item/user, sampled/full-catalog và trường hợp mô hình thua baseline. Tham khảo [lời trình bày](trao_doi_gvhd_2026_10_27.md).

- [ ] **P5-05 — Bàn giao repo có thể chạy lại.** Rà diff đang có, commit có chủ đích và đồng bộ remote theo yêu cầu; không đưa credentials, dataset/checkpoint lớn hoặc file tạm vào Git. Giữ manifest và hướng dẫn tải/khôi phục, checksum cho artifact lớn; xác nhận khởi động demo trên môi trường sạch.
  Hoàn thành khi người khác biết cài gì, tải gì, chạy lệnh nào và nhận kết quả nào, không phụ thuộc vào thư mục `tmp/transfer` không được bàn giao.

## P6 — Mở rộng khi phạm vi và tài nguyên cho phép

- [ ] **P6-01 — UFM trên All Beauty nếu giữ cam kết đa dataset.** Hiện mới có TF-IDF; cần CLIP coverage, các mô hình/seed và protocol phù hợp trên dataset này. Không so trực tiếp điểm khác protocol hoặc gọi train/evaluate riêng từng dataset là transfer chưa fine-tune.
  Hoàn thành khi có bằng chứng ở dataset thứ hai và cách diễn giải đúng phạm vi; ưu tiên sau khi Toys có benchmark đáng tin.

- [ ] **P6-02 — Retrieval rồi UFM reranking nếu catalog lớn chậm.** Thiết kế retrieval phù hợp với biểu diễn, thử ANN/FAISS, đo Recall@100/200 và NDCG đầu cuối cùng latency. UFM có scoring điều kiện theo candidate nên không giả định toàn bộ score là tích vô hướng có thể thay bằng ANN trực tiếp.
  Hoàn thành khi mức tăng tốc và mất mát chất lượng được đo; không hứa độ trễ dưới 20 ms khi chưa benchmark.

- [ ] **P6-03 — Kiểm tra độ bền và chất lượng danh sách.** Thử thiếu ảnh/text có kiểm soát, history ngắn, coverage/diversity/novelty và ảnh hưởng popularity bias. Xác định trước cách đo và phạm vi, không biến thành yêu cầu nghiệm thu mới khi chưa thống nhất.
  Hoàn thành khi có phân tích giúp giải thích cold-start và trải nghiệm người dùng, thay vì thêm metric cho đủ số lượng.

- [ ] **P6-04 — Encoder/fine-tuning hoặc người dùng mới sâu hơn.** Chỉ cân nhắc encoder khác, text dài, adapter/LoRA hoặc onboarding sở thích sau khi đã có baseline/ablation và ngân sách. Đo lợi ích, chi phí, overfitting và khả năng xử lý unseen item.
  Hoàn thành khi có câu hỏi nghiên cứu cụ thể và đối chứng; chưa cần thêm chatbot/LLM để chứng minh đồ án thuộc AI ứng dụng.

## Thứ tự triển khai đề nghị

Làm P0 và chốt P1 trước; sau đó phục hồi một lượt UFM để kiểm tra nạp/suy luận và chạy baseline cùng giao thức. Có mốc so sánh tin cậy mới chọn thí nghiệm P3. Hoàn thiện demo UFM và báo cáo song song với các kết quả đã được xác minh. P6 phụ thuộc cam kết với giảng viên và tài nguyên thực tế.

Nếu thời gian hạn chế, ưu tiên một dataset có protocol rõ, baseline công bằng, full UFM và ablation chủ chốt, nhiều seed theo phạm vi đã chốt, demo đúng model và báo cáo trung thực. Không cần chạy toàn bộ mở rộng trước khi giải quyết lỗi nền tảng.
