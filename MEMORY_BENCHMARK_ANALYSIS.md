# Phân tích benchmark và bonus: Conflict handling

Đo lại ngày 03/10/2026 bằng `python src/benchmark.py`. Output và hướng dẫn chạy nằm trong [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). Dùng tên `MEMORY_BENCHMARK_ANALYSIS.md` cho phần phân tích trong bài nộp; README, Guide và Rubric không quy định tên file bắt buộc. Nếu giảng viên có quy định riêng thì đổi tên theo quy định đó.

## Điều kiện đo và kết quả

Chế độ offline, không gọi API hoặc judge LLM; ngưỡng compact 900 token, giữ 6 message gần nhất. Mỗi suite dùng agent mới và thư mục memory tạm sạch. Hai agent nhận cùng input, thứ tự lượt và câu hỏi; sau mỗi hội thoại, recall được hỏi trong một thread mới trước khi chạy hội thoại tiếp theo.

| Dataset | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | Baseline | 727 | 11,181 | 0% | 0% | 0 | 0 |
| Standard | Advanced | 774 | 17,509 | 100% | 100% | 328 | 0 |
| Stress | Baseline | 127 | 21,194 | 0% | 0% | 0 | 0 |
| Stress | Advanced | 653 | 12,074 | 100% | 100% | 246 | 12 |

Token và compaction chỉ tính các lượt hội thoại đầu vào, không cộng lượt hỏi recall. `estimate_tokens()` dùng `len(text.strip()) // 4` với tối thiểu 1 cho chuỗi không rỗng, nên các con số là ước lượng ngữ cảnh và đầu ra, không phải hóa đơn API. `Recall` là trung bình điểm 0/0.5/1 theo câu hỏi; `quality` là trung bình tỷ lệ fact kỳ vọng có trong câu trả lời. Quality 100% chỉ chứng minh khớp fact theo heuristic, chưa đánh giá tính hữu ích, suy luận, phủ định hoặc mâu thuẫn.

## 1. Vì sao Advanced có recall tốt hơn

**Số liệu:** Baseline đạt 0% recall ở cả Standard và Stress, Advanced đạt 100% ở cả hai; memory của Baseline tăng 0 byte, Advanced tăng lần lượt 328 và 246 byte.

**Cơ chế:** `BaselineAgent.sessions` khóa theo `thread_id`; thread recall mới không có lịch sử khai báo. Advanced gọi `extract_profile_updates()` trong `_prepare_turn()`, ghi fact qua `UserProfileStore.upsert_fact()`, rồi `_offline_response()` đọc `facts()` từ `User.md` khi hỏi ở thread mới. Profile được khóa theo user, còn short-term và summary được khóa theo thread. Test `test_cross_session_recall` kiểm chứng cả Advanced nhớ, Baseline quên và Advanced mới khởi tạo vẫn đọc được fact đã lưu.

**Giới hạn:** Kết quả chỉ áp dụng cho các kiểu fact và cách diễn đạt mà extractor heuristic nhận diện trong hai dataset. Không chứng minh recall trên câu hỏi tùy ý hoặc môi trường live; substring scoring cũng có thể cho điểm một câu nhắc đúng từ nhưng sai ý.

## 2. Vì sao Advanced tốn hơn ở hội thoại ngắn

**Số liệu:** Standard: output của Advanced là 774 so với 727 token của Baseline, tăng 47 token (6.46%); prompt là 17,509 so với 11,181, tăng 6,328 token (56.60%). Cả hai có 0 compaction.

**Cơ chế:** `_estimate_prompt_context_tokens()` cộng profile, summary và message gần nhất trong mỗi lượt. Với thread ngắn chưa kích hoạt compact, Advanced vẫn giữ lịch sử và thêm hồ sơ tích lũy qua các thread. Phần output khác nhau do câu trả lời dựa trên profile và style đã lưu. Ghi file tạo thêm thao tác I/O, nhưng benchmark không tính I/O vào token, nên không dùng việc ghi file để giải thích trực tiếp chênh lệch token.

**Giới hạn:** Đây là tổng trên cùng bộ input, không phải giá tiền hoặc thời gian thực thi. Không thể suy ra Advanced luôn đắt hơn với mọi hội thoại ngắn; kết quả còn phụ thuộc độ dài profile, style và threshold.

## 3. Vì sao compact có lợi ở hội thoại dài

**Số liệu:** Stress: Advanced xử lý 12,074 prompt token so với 21,194 của Baseline, giảm 9,120 token (43.03%), với 12 compaction. Output của Advanced lại cao hơn: 653 so với 127 token. Vì vậy lợi ích đo được nằm ở `Prompt tokens processed`, không phải `Agent tokens only`.

**Cơ chế:** Baseline cộng toàn bộ lịch sử trước mỗi phản hồi, nên message dài được xử lý lại qua nhiều lượt. `CompactMemoryManager.append()` thay phần lịch sử cũ bằng summary và giữ message gần nhất khi vượt ngưỡng; `summarize_messages()` giới hạn số đoạn và độ dài trích đoạn. Advanced cộng ngữ cảnh sau khi compact, giảm phần lịch sử bị lặp. Stress chỉ có 16 lượt nhưng Baseline xử lý 21,194 prompt token: các đoạn tin dài tạo tải ngữ cảnh lớn; không nên so tổng này với Standard để kết luận chỉ từ số lượt, vì độ dài input khác nhau.

**Kiểm tra thêm để tách tác động compact:** Chạy Advanced trên chính stress input với threshold 100,000,000 (không kích hoạt compact) cho 25,819 prompt token, 653 output token, recall 100%, memory 246 byte và 0 compaction. Bản có compact dùng 12,074 prompt token, giảm 53.24% so với đối chứng này; output, recall và kích thước profile giống nhau. Đây là phép đo bổ sung, không thay hàng Baseline trong bảng bắt buộc.

**Giới hạn:** Summary là trích đoạn có mất thông tin; recall tốt ở đây dựa vào profile, không chứng minh summary giữ được mọi chi tiết. Threshold là điều kiện kích hoạt, không phải trần cứng nếu các message gần nhất vẫn rất dài. Bộ đếm token chưa bao gồm chi phí model tóm tắt vì bản offline dùng heuristic.

## 4. Memory tăng trưởng và rủi ro

**Số liệu:** Stress tăng 246 byte profile và compact 12 lần; Standard tăng 328 byte profile và compact 0 lần. Hai suite dùng user và state riêng, nên 246 byte không phải sự thu nhỏ từ 328 byte.

**Cơ chế:** `memory_growth_bytes` tính tổng kích thước file cuối trừ đầu, mỗi user một lần. `upsert_fact()` thay giá trị cùng key thay vì nối thêm bản cũ; compact tác động lịch sử trong RAM, không nén `User.md`. Hai cột memory và compaction đo hai lớp khác nhau: 12 compaction không có nghĩa file profile đã được nén 12 lần.

**Quan sát và giới hạn:** Không quan sát thấy profile phình nhanh hoặc fact nhiễu làm sai recall trên dataset hiện tại; không nên bịa lỗi từ bảng 100%. Rủi ro có thể tái hiện khi lưu cả giá trị cũ và mới của cùng key được kiểm tra riêng ở phần bonus. Trong triển khai dài hạn, giá trị sở thích dài hơn, nhiều user hoặc fact trích sai vẫn có thể làm tăng dung lượng và tải prompt; benchmark này không đo tốc độ tăng theo tháng, số byte ghi tích lũy hay latency I/O.

## Bonus đã chọn: Conflict handling

### Vấn đề giải quyết và bằng chứng

Correction về nơi ở/nghề có trong input. Một phép kiểm tra tạm cho user riêng cung cấp `Mình ở Huế.` rồi `Giờ mình đang ở Đà Nẵng chứ không còn ở Huế nữa.` cho thấy profile hiện tại chỉ còn một dòng `location: Đà Nẵng`, và recall ở thread mới trả `Đà Nẵng`.

Để tái hiện điểm yếu của cách lưu nối tiếp, lấy cùng các fact extractor đã trích và nối từng dòng vào một profile đối chứng. Đối chứng giữ hai dòng location, trong khi bản upsert chỉ giữ một:

| Cách lưu trong phép kiểm tra riêng | Dòng location | Byte profile | Token profile ước lượng |
| --- | ---: | ---: | ---: |
| Nối fact cũ và mới | 2 | 57 | 12 |
| `upsert_fact()` hiện tại | 1 | 39 | 8 |

Đây là đối chứng lưu profile đơn giản, không phải Baseline Agent và không phải kết quả toàn bộ dataset. Nó cho thấy rủi ro giữ fact lỗi thời và ngữ cảnh dư thừa; không khẳng định đối chứng luôn trả lời sai, vì một parser vẫn có thể chọn dòng cuối.

### Cải thiện recall hoặc token cost như thế nào

Bonus đã triển khai trong `src/memory_store.py`: `upsert_fact()` thay dòng cùng key và bỏ dòng trùng; extractor nhận correction, `_prepare_turn()` ghi cập nhật trước khi trả lời. `_offline_response()` đọc giá trị hiện tại từ profile. Prompt live cũng hướng dẫn ưu tiên giá trị đã correction, nhưng chưa kiểm chứng hiệu quả bằng API thật.

Trong probe, profile giảm 18 byte và 4 token so với cách nối dòng; vì profile được mang theo mỗi lượt, nó tránh lặp phần ngữ cảnh dư thừa đó. Test `test_corrections_survive_compaction_and_restart` xác nhận chỉ còn một key location/profession, không còn giá trị cũ và vẫn recall đúng sau compact rồi khởi động lại. Benchmark đạt recall 100% với cơ chế này, nhưng chưa có phép ablation chứng minh bỏ upsert sẽ làm recall của toàn dataset giảm bao nhiêu; không gán một mức tăng recall chưa đo được cho bonus.

### Rủi ro mới

Chính sách giá trị mới nhất thắng có thể ghi đè fact đúng bằng một câu trích sai; xóa giá trị cũ cũng làm mất lịch sử để truy vết. Regex lọc câu hỏi, câu đùa và chuyến đi chỉ phủ các cách diễn đạt đã dự liệu. Hiện chưa có confidence score, timestamp/provenance, xác nhận correction mơ hồ hoặc bảo vệ cập nhật đồng thời. Guardrail tiếp theo nên ưu tiên độ tin cậy trước khi ghi và nguồn của fact; không tự động coi mọi câu đến sau là chính xác hơn.

## Đối chiếu Rubric

| Mốc | Bằng chứng hiện có | Đánh giá |
| --- | --- | --- |
| 0–60 | Baseline thread-only, Advanced có profile, stress kích hoạt 12 compaction, đủ dataset và cấu trúc repo | Đáp ứng yêu cầu cơ bản |
| 60–75 | Hai agent cùng input; đủ sáu chỉ số; bốn test tại `src/test_agents.py` pass | Đáp ứng yêu cầu benchmark và test cốt lõi |
| 75–90 | Hai suite; Standard prompt tăng 56.60%, Stress prompt giảm 43.03%; phân biệt output với prompt và phân tích memory | Có bằng chứng cho trade-off và tác dụng compact |
| 90–100 | Conflict handling đã có code, test correction/restart, probe 57→39 byte và 12→8 token, phân tích rủi ro | Có cơ sở xét bonus; điểm cuối do reviewer quyết định |

Kiểm tra ngày 03/10/2026: `python -m pytest src/test_agents.py -v` thu thập đúng 4 test và pass; `python -m pytest -q` có 59 test pass. Test cốt lõi dựng đủ `LabConfig` trong `tmp_path`, threshold 80, giữ 2 message và ép offline. Không cần API key để tái lập kết quả benchmark hoặc các test này.
