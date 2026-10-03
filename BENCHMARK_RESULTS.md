# Benchmark results

Chạy ở chế độ offline, threshold 900 token, giữ 6 message gần nhất. Mỗi bộ dữ liệu dùng agent mới và thư mục memory tạm riêng; không đọc hoặc sửa hồ sơ đã có trong `state/profiles`. Chạy lại cho kết quả giống nhau.

## Cách chạy

```powershell
.venv\Scripts\python.exe src/benchmark.py
```

Kiểm tra toàn bộ test (59 test đã pass):

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Cách đo

- Hai agent nhận cùng input, cùng thứ tự lượt và cùng câu hỏi. Sau mỗi hội thoại, hỏi các câu recall trong một thread mới riêng cho hội thoại đó, trước khi chạy hội thoại tiếp theo. ID thread giống nhau cho hai agent; chỉ số thứ tự tránh trùng khi dataset có ID lặp.
- Hai cột token cộng số liệu từng lượt của các hội thoại đầu vào; không cộng lượt hỏi recall. Token là ước lượng heuristic, không phải token API thực tế.
- Recall chấm từng câu: không khớp fact = 0; khớp một phần = 0.5; khớp đủ = 1. Điểm bảng là trung bình các câu hỏi, không phải tỷ lệ tổng số fact.
- Quality là tỷ lệ fact kỳ vọng xuất hiện trong câu trả lời, rồi lấy trung bình theo câu hỏi. Cùng công thức cho hai agent. Đây là proxy offline: điểm 100% không chứng minh khả năng viết, tính hữu ích hoặc suy luận; substring matching cũng không phát hiện phủ định hay mâu thuẫn. So khớp không phân biệt hoa/thường và chuẩn hóa Unicode NFC, vẫn giữ dấu tiếng Việt.
- Memory growth là tổng kích thước file cuối trừ đầu, tính mỗi user một lần; không phải tổng số byte ghi ra đĩa. Với state sạch, growth bằng kích thước file cuối.
- Compactions chỉ cộng trên các thread hội thoại, không tính thread recall. Baseline luôn bằng 0.

## Kết quả

Mode: deterministic offline; token counts are heuristic estimates.
Tokens/compactions: conversation turns only (recall probes excluded).
Recall: mean 0/0.5/1 per question; quality: mean expected-fact coverage.
Memory growth: final minus initial profile bytes, counted once per user.
Compaction: threshold=900 tokens, keep=6 messages.

## Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 727 | 11181 | 0.00% | 0.00% | 0 | 0 |
| Advanced | 774 | 17509 | 100.00% | 100.00% | 328 | 0 |

## Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 127 | 21194 | 0.00% | 0.00% | 0 | 0 |
| Advanced | 653 | 12074 | 100.00% | 100.00% | 246 | 12 |


## Nhận xét

Phân tích đầy đủ cơ chế, giới hạn phép đo, bonus Conflict handling và đối chiếu Rubric: [MEMORY_BENCHMARK_ANALYSIS.md](MEMORY_BENCHMARK_ANALYSIS.md).

Ở Standard Benchmark, Advanced xử lý 17,509 prompt token so với 11,181 của Baseline: hồ sơ bền vững làm tăng ngữ cảnh ở các thread ngắn, chưa cần compact.

Ở Long-Context Stress Benchmark, Advanced xử lý 12,074 prompt token so với 21,194 của Baseline, giảm khoảng 43.03%, với 12 lần compact. Advanced sinh nhiều token hơn vì tuân theo style 3 bullet; lợi ích compact thể hiện ở lượng prompt phải xử lý.

Recall của Baseline là 0% vì câu hỏi được đưa sang thread mới. Advanced đạt 100% theo thang substring nhờ hồ sơ bền vững, kể cả correction và nhiễu trong dataset. Kết quả này kiểm chứng hành vi offline trên hai bộ dữ liệu đã cho; chưa có đánh giá bằng model thật hoặc judge LLM.
