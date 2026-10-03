# Lab 17 — Memory Systems for AI Agent

> **Course:** `K04-L34-P2-T3`  
> **Lesson:** `Day17-Track3-MemorySystems4Agent`  
> **Version:** `4`

## Mục lục

1. [Chuẩn bị repo và môi trường chạy](#1-chuẩn-bị-repo-và-môi-trường-chạy)
2. [Đọc scaffold trước khi viết dòng nào](#2-đọc-scaffold-trước-khi-viết-dòng-nào)
3. [Chốt cấu hình chung trong `config.py`](#3-chốt-cấu-hình-chung-trong-configpy)
4. [Làm lớp memory trong `memory_store.py`](#4-làm-lớp-memory-trong-memory_storepy)
5. [Hoàn thiện Baseline Agent](#5-hoàn-thiện-baseline-agent)
6. [Hoàn thiện Advanced Agent](#6-hoàn-thiện-advanced-agent)
7. [Viết benchmark cho hai bộ dữ liệu](#7-viết-benchmark-cho-hai-bộ-dữ-liệu)
8. [Viết test cho hành vi memory](#8-viết-test-cho-hành-vi-memory)
9. [Chạy benchmark và đọc đúng sáu cột](#9-chạy-benchmark-và-đọc-đúng-sáu-cột)
10. [Viết phân tích kết quả và chọn bonus](#10-viết-phân-tích-kết-quả-và-chọn-bonus)
11. [Nộp bài](#11-nộp-bài)

## Tài liệu bài học

- **Slide:** Bài 15
- **Phạm vi:** trang 1–73

## 1. Chuẩn bị repo và môi trường chạy

> **Về bài lab này**
>
> Hoàn thiện memory layer trong `src/`: Baseline Agent chỉ nhớ trong thread, Advanced Agent thêm `User.md` bền vững và compact memory, rồi benchmark hai agent trên hai bộ dữ liệu tiếng Việt để đọc trade-off recall, token và độ phức tạp.

Bạn sẽ hoàn thiện memory layer cho một AI agent trong repo `src/`, rồi chứng minh bằng số liệu rằng lớp memory đó thay đổi điều gì. Kết quả cuối cùng không phải một agent "nhớ nhiều hơn", mà là hai agent chạy trên cùng một benchmark để bạn đọc được trade-off giữa độ nhớ dài hạn, chất lượng phản hồi, chi phí token và độ phức tạp của hệ thống.

Đích đến cụ thể: `python src/benchmark.py` in ra **Standard Benchmark** và **Long-Context Stress Benchmark**, mỗi bảng so sánh Baseline với Advanced theo đủ sáu cột; `pytest src/test_agents.py -v` xanh; và một file phân tích riêng trong repo giải thích vì sao các con số đó lại như vậy.

### Bạn làm được gì sau bài này

- Phân biệt short-term memory, persistent memory và compact memory bằng đúng chỗ chúng xuất hiện trong `src/`.
- Cài đặt `LabConfig` và `load_config()` dùng chung cho sáu provider: openai, custom, gemini, anthropic, ollama, openrouter.
- Cài đặt `estimate_tokens()`, `UserProfileStore`, `extract_profile_updates()`, `summarize_messages()` và `CompactMemoryManager`.
- Hoàn thiện Baseline Agent và Advanced Agent chạy offline, tất định, không cần API key.
- Chạy `python src/benchmark.py` ra hai bảng Standard và Long-Context Stress với đủ sáu cột chỉ số.
- Viết test cho `User.md`, compact trigger, cross-session recall và mức giảm prompt load.

### Cần chuẩn bị

- Python >= 3.11 và quyền tạo virtual environment trên máy.
- Đã đọc `README.md`, `Guide.md`, `Rubric.md` của repo Day 17.
- Biết đọc traceback Python và chạy lệnh trong terminal tại thư mục gốc repo.
- langchain, langgraph và provider SDK: langchain-openai, langchain-google-genai, langchain-anthropic, langchain-ollama, langchain-openrouter
- python-dotenv, tabulate, pytest

> **Lỗi thường gặp**
>
> - Baseline vẫn nhớ fact qua thread mới → session đang được lưu theo `user_id` thay vì `thread_id`.
> - Advanced không sinh `User.md` → `extract_profile_updates()` trả rỗng hoặc `_reply_offline()` không gọi `write_text()`.
> - Cột `Compactions` luôn bằng 0 → chưa kiểm tra ngưỡng token sau mỗi lần `CompactMemoryManager.append()`.
> - `ModuleNotFoundError: No module named 'config'` → chạy `python src/benchmark.py` từ thư mục gốc thay vì `python -m src.benchmark`.
> - Advanced giữ đồng thời fact cũ và fact mới sau correction → chưa cập nhật lại `User.md` qua `edit_text()`.

Mục tiêu: có repo Day 17 trên máy, có môi trường Python sạch, và biết chính xác thư mục nào chứa phần bạn phải viết.

Repo nguồn của bài là nơi chứa scaffold, hướng dẫn và dữ liệu benchmark:

[Repo Day 17 — Memory Systems for AI Agent](https://github.com/VinUni-AI20k/day17-cohort4-MemorySystems4Agent)

Trong repo đó, `src/` là phần bạn phải hoàn thiện, `data/` là input benchmark dùng chung và không được sửa. Khi agent chạy, nó ghi trạng thái vào `state/` — thư mục này đã nằm trong `.gitignore`, nên bạn có thể xóa và chạy lại bất cứ lúc nào để có kết quả từ đầu.

Bước đầu tiên là tạo virtual environment và cài đủ package cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate` và `pytest`.

- **1.** Mở terminal tại thư mục gốc của repo, ngay cạnh `README.md`.
- **2.** Tạo virtual environment và kích hoạt nó bằng đúng lệnh cho hệ điều hành của bạn.
- **3.** Cài các package mà `README.md` liệt kê.

#### Tạo môi trường Python

##### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

##### Windows PowerShell

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Trên Windows không có lệnh `python3`, nên dùng `py -3`. Sau khi kích hoạt, dấu nhắc lệnh sẽ hiện tên môi trường `(.venv)`; đó là dấu hiệu bạn đang cài đúng chỗ, và cũng là điều kiện để `pytest` về sau nhìn thấy các package này.

> **Không commit bí mật**
>
> Repo chỉ cần API key khi bạn chạy chế độ live. `.env` đã nằm trong `.gitignore` — hãy giữ nguyên như vậy, không dán key vào mã nguồn, vào ảnh chụp màn hình hay vào file nộp. Toàn bộ benchmark của bài chạy được ở chế độ offline, không cần key.

#### Trước khi vào code

- [ ] Đã clone repo Day 17 và đang đứng ở thư mục gốc.
- [ ] Đã tạo `.venv` và kích hoạt thành công.
- [ ] `pip install` chạy xong không lỗi.
- [ ] Nhìn thấy hai thư mục `src/` và `data/`.

## 2. Đọc scaffold trước khi viết dòng nào

Mục tiêu: xác định mỗi file chịu trách nhiệm gì, và tự tay nhìn thấy scaffold đang dở dang ở đâu.

`Guide.md` (Bước 1) yêu cầu đọc `src/README.md` và sáu file nguồn theo thứ tự triển khai. Lý do không phải thủ tục: các file này gọi nhau theo một chiều phụ thuộc, nên đọc sai thứ tự sẽ khiến bạn viết code cho một hàm mà đầu vào của nó chưa tồn tại.

| File | Vai trò trong bài |
| --- | --- |
| `src/config.py` | Đường dẫn repo, `data/`, `state/`; ngưỡng compact; cấu hình provider cho model chính và model judge |
| `src/model_provider.py` | Khởi tạo chat model cho từng provider: `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `src/memory_store.py` | Lõi memory: ước lượng token, `User.md`, trích fact, compact hội thoại dài |
| `src/agent_baseline.py` | Agent A — chỉ nhớ trong cùng thread |
| `src/agent_advanced.py` | Agent B — short-term + `User.md` + compact |
| `src/benchmark.py` | So sánh hai agent trên hai bộ dữ liệu |
| `src/test_agents.py` | Bốn test kiểm chứng hành vi memory |

Đọc xong, hãy chạy thử lệnh mà `README.md` quy định cho bài nộp, ngay cả khi mọi thứ còn `TODO`:

```bash
python src/benchmark.py
```

Bạn sẽ nhận `NotImplementedError` phát ra từ `load_config()`. Đây là checkpoint đầu tiên và nó có nghĩa cụ thể: luồng benchmark đã nối đúng thứ tự, chỉ thiếu phần thân của các hàm. Khi làm xong bài, chính lệnh này phải in ra hai bảng thay vì traceback.

Hai chi tiết dễ làm bạn mất thời gian về sau:

- Import trong `src/` là import phẳng (`from config import LabConfig`), không phải import theo package. Vì vậy hãy luôn chạy `python src/benchmark.py` từ thư mục gốc; `python -m src.benchmark` sẽ báo `ModuleNotFoundError: No module named 'config'`.
- `data/conversations.json` gồm 10 hội thoại khoảng 10 lượt của user `dungct`; `data/advanced_long_context.json` gồm 1 hội thoại 16 lượt của user `dungct_stress`. Cả hai là dữ liệu cố định, bạn không chỉnh sửa mà chỉ đọc.

#### Đọc scaffold

- [ ] Đã chạy `python src/benchmark.py` và thấy `NotImplementedError`.
- [ ] Biết `src/` là phần phải viết, `data/` là input dùng chung.
- [ ] Biết import phẳng nghĩa là phải chạy lệnh từ thư mục gốc repo.

## 3. Chốt cấu hình chung trong `config.py`

Mục tiêu: có một `LabConfig` đầy đủ và `load_config()` trả về config dùng được, để mọi file sau đó không phải tự đoán đường dẫn hay ngưỡng.

#### Vì sao cấu hình phải chốt trước?

`LabConfig` là hợp đồng giữa ba tầng: agent đọc `state_dir` để ghi và đọc memory, benchmark đọc `data_dir` để nạp dữ liệu, và cả hai đọc `compact_threshold_tokens` để biết khi nào compact được phép kích hoạt. Nếu mỗi file tự tính đường dẫn riêng, bạn sẽ gặp tình huống benchmark chạy trên một bộ dữ liệu còn agent ghi memory vào chỗ khác, và cột `Memory growth (bytes)` sẽ luôn bằng 0 dù logic đúng.

Dataclass trong scaffold đã khai báo sẵn bảy trường: `base_dir`, `data_dir`, `state_dir`, `compact_threshold_tokens`, `compact_keep_messages`, `model`, `judge_model`. Giữ nguyên tên và kiểu của chúng, vì `agent_baseline.py`, `agent_advanced.py`, `benchmark.py` và `test_agents.py` đều đọc đúng những tên này.

#### Hai tham số quyết định hành vi compact

`compact_threshold_tokens` và `compact_keep_messages` là hai nút vặn của cả bài. Thứ nhất là ngưỡng token để một thread bị coi là "quá dài" và phải nén bớt lịch sử. Thứ hai là số message gần nhất được giữ nguyên văn sau khi nén.

Điều cần tránh: đặt ngưỡng quá cao. Khi đó compact gần như không bao giờ chạy, `Compactions` bằng 0, và lợi thế của Advanced ở hội thoại dài biến mất — đúng lỗi mà `Rubric.md` liệt vào nhóm trừ điểm mạnh ("compact memory không thực sự kích hoạt"). Ở phần test bạn sẽ đặt ngưỡng nhỏ hơn nhiều để ép compact xảy ra nhanh; còn ở benchmark, ngưỡng phải đủ lớn để hội thoại ngắn không bị nén, nhưng đủ nhỏ để hội thoại 16 lượt trong stress test bị nén vài lần.

#### Provider và chế độ offline

Repo yêu cầu hỗ trợ sáu provider: `openai`, `custom` (OpenAI-compatible base URL), `gemini`, `anthropic`, `ollama`, `openrouter`. `ProviderConfig` đã được khai báo sẵn; phần bạn viết là `normalize_provider()` và `build_chat_model()` trong `src/model_provider.py`. `README.md` nêu lý do yêu cầu này: memory system không nên bị khóa vào một provider duy nhất.

| Provider | Model cần dựng trong `build_chat_model()` |
| --- | --- |
| `openai` | `ChatOpenAI` |
| `custom` | `ChatOpenAI` kèm `base_url` |
| `gemini` | `ChatGoogleGenerativeAI` |
| `anthropic` | `ChatAnthropic` |
| `ollama` | `ChatOllama` |
| `openrouter` | `ChatOpenRouter` |

`normalize_provider()` tồn tại để chuẩn hóa alias viết sai, ví dụ `anthorpic` → `anthropic`. Hãy trả về chuỗi provider chuẩn và xử lý trường hợp không nhận diện được, thay vì để lỗi mơ hồ phát sinh tận lúc dựng model.

Tên biến môi trường do bạn quyết định khi viết `load_config()`; scaffold gợi ý các nút `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL`, `OPENROUTER_API_KEY`, `CUSTOM_BASE_URL`, `CUSTOM_API_KEY`. Dù chọn tên nào, hãy giữ một quy ước và ghi nó vào file phân tích để người chấm chạy lại được.

Việc quan trọng nhất ở đây: **đường offline phải chạy được mà không cần API key**. `_maybe_build_langchain_agent()` chỉ được gọi ở chế độ live, còn `reply()` phải rơi về nhánh tất định khi không có model thật. Nhờ vậy benchmark và test cho ra kết quả lặp lại được — điều kiện để bạn so sánh Baseline với Advanced một cách công bằng. Chế độ live (LangChain/LangGraph) là phần mở rộng, không phải điều kiện để bài chạy.

- **1.** Trong `src/config.py`, điền giá trị mặc định cho `LabConfig`: đường dẫn repo, `data/`, `state/`, ngưỡng compact và số message giữ lại.
- **2.** Trong `load_config()`, resolve `root` rồi tạo `state/` nếu chưa có, đọc `.env` qua `python-dotenv`, và trả về `LabConfig` đã điền đủ bảy trường.
- **3.** Trong `src/model_provider.py`, viết `normalize_provider()` và `build_chat_model()` theo bảng provider ở trên.
- **4.** Kiểm tra nhanh rằng config nạp được và `state/` đã tồn tại.

```bash
python -c "import sys; sys.path.insert(0, 'src'); from config import load_config; cfg = load_config(); print(cfg.data_dir, '|', cfg.state_dir, '|', cfg.compact_threshold_tokens)"
```

Kết quả mong đợi: một dòng in ra đường dẫn `data/`, đường dẫn `state/` và ngưỡng compact của bạn, không có traceback. Nếu `state/` chưa xuất hiện trên đĩa, `load_config()` chưa tạo thư mục — và mọi bước sau sẽ đổ vỡ ở chỗ ghi `User.md`.

## 4. Làm lớp memory trong `memory_store.py`

Mục tiêu: memory layer chạy đúng ở mức đơn vị — đo được token, ghi và sửa được `User.md`, trích được fact ổn định, và nén được lịch sử dài.

#### Ba loại memory trong Lab này

Đây là file trả lời câu hỏi trung tâm của cả track, nên hãy tách nghĩa ba thuật ngữ theo đúng cách bài dùng chúng:

- **Short-term memory** là danh sách message trong một `thread_id`. Nó chỉ sống trong phiên đó; sang thread mới là mất. Cả Baseline và Advanced đều có lớp này.
- **Persistent memory** là `User.md` trên đĩa: fact ổn định về người dùng được ghi lại và còn nguyên khi thread đóng. Chỉ Advanced có.
- **Compact memory** là cơ chế nén: message cũ được thay bằng một bản tóm tắt, còn vài message gần nhất vẫn giữ nguyên văn. Chỉ Advanced có.

Ba lớp này ánh xạ thẳng vào ba cột benchmark: short-term quyết định `Prompt tokens processed`, persistent quyết định `Cross-session recall` và `Memory growth (bytes)`, compact quyết định `Compactions`.

#### `estimate_tokens()` — cây thước dùng chung

Toàn bộ cơ chế compact và cả cột `Prompt tokens processed` đều dựa trên hàm này, nên nó phải **tất định**: cùng một chuỗi luôn cho cùng một số. Scaffold gợi ý cách đơn giản: bỏ khoảng trắng, trả 0 cho chuỗi rỗng, còn lại ước lượng theo độ dài ký tự (ví dụ `len(text) / 4`).

Đừng cố khớp tokenizer thật. `README.md` nói rõ một estimator heuristic ổn định là đủ cho benchmark offline. Cái bạn cần tránh là trả về số ngẫu nhiên hoặc phụ thuộc thời gian — khi đó ngưỡng compact sẽ kích hoạt thất thường và test sẽ chập chờn.

Sau khi tự viết bản đầu tiên, kiểm tra ba tính chất: `estimate_tokens("")` phải trả `0`; hai lần gọi trên cùng một chuỗi phải cho cùng kết quả; và chuỗi dài hơn phải cho số lớn hơn hoặc bằng. Nếu một trong ba điều kiện sai, sửa trước khi viết tiếp. Mẫu tối thiểu bám theo gợi ý trong scaffold:

#### `UserProfileStore` — nơi `User.md` sống

Lớp này là cầu nối giữa agent và đĩa. Nó nhận `root_dir` (trong `agent_advanced.py`, scaffold truyền `config.state_dir / "profiles"`), rồi ánh xạ mỗi `user_id` sang một file markdown. `README.md` mô tả dạng đường dẫn `state/profiles/<user>/User.md`; hãy chọn một dạng cụ thể và giữ nhất quán giữa `path_for()`, `read_text()` và `write_text()`.

Năm hàm cần hoàn thiện, mỗi hàm có một hợp đồng rõ ràng:

- `path_for(user_id)` — chuẩn hóa `user_id` trước khi ghép đường dẫn. Đây là bước an toàn, không phải trang trí: `user_id` đến từ file dữ liệu, nên đừng để nó tạo ra đường dẫn nằm ngoài `root_dir`.
- `read_text(user_id)` — trả nội dung file, hoặc một profile markdown rỗng khi file chưa tồn tại. Trả chuỗi rỗng thay vì ném lỗi, vì lượt chat đầu tiên của mọi user luôn đọc file chưa có.
- `write_text(user_id, content)` — ghi markdown và trả về đường dẫn file để nơi gọi kiểm tra được.
- `edit_text(user_id, search_text, replacement)` — thay **một** lần xuất hiện và trả `True`/`False` cho biết có thay đổi hay không. Hàm này tồn tại cho tình huống correction: người dùng đổi nơi ở, bạn phải sửa dòng cũ chứ không thêm dòng mới.
- `file_size(user_id)` — số byte hiện tại của file. Đây chính là nguồn số cho cột `Memory growth (bytes)`.

Tùy chọn nhưng hữu ích: `facts()` và `upsert_fact()`. Chúng giúp bạn ghi fact theo khóa thay vì nối văn bản, và nhờ đó việc xử lý correction sạch hơn.

#### `extract_profile_updates()` và cái bẫy nhiễu

Hàm này biến câu người dùng thành các fact ổn định: tên, nơi ở, nghề nghiệp, phong cách trả lời mong muốn, sở thích hoặc mối quan tâm kỹ thuật. Scaffold gợi ý dùng vài regex rồi chỉ trả về những fact thực sự có trong message.

Phần khó không nằm ở regex mà ở chỗ chọn đúng. Dữ liệu benchmark cố tình chứa hai loại bẫy mà `README.md` nêu:

- **Correction**: nơi ở đổi giữa Đà Nẵng và Huế. Trong stress test, người dùng bắt đầu ở Huế rồi chuyển sang Đà Nẵng. Nếu bạn chỉ `upsert` mà không thay giá trị cũ, `User.md` sẽ giữ cả hai và agent trả lời sai ở câu hỏi "nơi ở hiện tại".
- **Nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa. Những chuỗi này trông giống fact nhưng không phải. Một bộ trích xuất quá tham sẽ ghi chúng vào `User.md` và làm recall giảm.

Thêm một quy tắc đáng làm ngay từ đầu: bỏ qua những lượt chỉ có câu hỏi. Người dùng hỏi "mình tên gì?" không cung cấp fact mới; ghi vào `User.md` từ lượt đó chỉ tạo nhiễu.

#### `CompactMemoryManager` — compact hoạt động thế nào

Đây là lớp thứ ba và là thứ phân biệt Advanced với Baseline về chi phí ngữ cảnh. Dataclass đã có sẵn ba trường: `threshold_tokens`, `keep_messages`, và `state` — một dict theo `thread_id`.

- `append(thread_id, role, content)`: tạo state cho thread nếu chưa có, thêm message mới, rồi kiểm tra ngưỡng token. Khi vượt ngưỡng, message cũ được đưa vào `summarize_messages()`, summary được lưu lại, và bộ đếm compaction tăng lên. Sau khi nén, thread vẫn giữ `keep_messages` message gần nhất nguyên văn.
- `context(thread_id)`: trả về state của thread với các khóa `messages`, `summary`, `compactions`. Advanced Agent sẽ ghép ba thứ này với `User.md` để dựng prompt.
- `compaction_count(thread_id)`: trả số lần đã nén. Cột `Compactions` lấy trực tiếp từ đây.

Hãy để `summarize_messages()` là heuristic văn bản trước (scaffold cho phép), rồi nâng lên tóm tắt bằng LLM nếu muốn. Điều cần giữ là tính đơn điệu: số message giữ nguyên văn không được tăng sau mỗi lần nén, nếu không compact chỉ đổi chỗ chứa.

> **Checkpoint của lớp memory**
>
> Sau khi Advanced Agent chạy được một hội thoại dài, trên đĩa phải có `User.md` với nội dung thật, và `compaction_count(thread_id)` phải lớn hơn 0. Hai điều kiện này tương ứng trực tiếp với hai lỗi bị trừ điểm mạnh trong `Rubric.md`: "advanced không lưu được `User.md`" và "compact memory không thực sự kích hoạt".

### Code gợi ý: Làm lớp memory trong `memory_store.py`

Code gợi ý của Codelab — hãy tự viết trước rồi mới đối chiếu.

#### `estimate_tokens.py`

```python
def estimate_tokens(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)
```

## 5. Hoàn thiện Baseline Agent

Mục tiêu: Baseline trở thành mốc so sánh trung thực — nhớ trong thread, quên sạch khi sang thread mới.

#### Baseline phải "ngây thơ" một cách có chủ đích

Baseline là nhánh đối chứng của thí nghiệm, nên giá trị của nó nằm ở việc nó **không** có gì: không `User.md`, không compact, không nhớ dài hạn. `Guide.md` (Bước 4) yêu cầu Baseline "phải thật sự ngây thơ ở mức hợp lý" và phải là mốc so sánh công bằng cho Advanced.

Rủi ro lớn nhất ở đây là vô tình làm Baseline thông minh lên. `Rubric.md` xếp "baseline vẫn nhớ được qua session mới theo cách không mong muốn" vào nhóm lỗi trừ điểm mạnh, vì khi đó toàn bộ phần so sánh phía sau mất ý nghĩa: recall cao của Baseline không còn chứng minh được điều gì.

#### `SessionState` khóa theo `thread_id`

Scaffold đã khai báo `self.sessions: dict[str, SessionState]` và `SessionState` gồm `messages`, `token_usage`, `prompt_tokens_processed`. Khóa của dict này là quyết định quan trọng nhất trong file: khóa theo `thread_id`, không theo `user_id`.

Nếu bạn khóa theo `user_id`, hai thread của cùng một người dùng sẽ dùng chung danh sách message, và Baseline lập tức "nhớ" xuyên phiên. Lỗi này không lộ ra khi bạn chỉ chạy một thread, nó chỉ lộ ở câu hỏi recall hỏi trong thread mới.

#### Điền các hàm còn thiếu

- **1.** `reply(user_id, thread_id, message)` — định tuyến: nếu `self.langchain_agent` tồn tại thì đi nhánh live, ngược lại gọi `_reply_offline()`. Trả về dict chứa câu trả lời và số liệu token cho lượt đó.
- **2.** `_reply_offline(thread_id, message)` — lấy hoặc tạo `SessionState` theo `thread_id`, thêm message người dùng, sinh câu trả lời tất định, cộng số liệu token, rồi lưu lại message của agent.
- **3.** `token_usage(thread_id)` — trả tổng token agent đã sinh trong thread. Đây là nguồn cho cột `Agent tokens only`.
- **4.** `prompt_token_usage(thread_id)` — trả ước lượng ngữ cảnh agent phải kéo theo qua các lượt. Đây là nguồn cho cột `Prompt tokens processed`.
- **5.** `_maybe_build_langchain_agent()` — tùy chọn. Khi làm, dùng `build_chat_model(self.config.model)` để Baseline chạy được với mọi provider đã hỗ trợ.

`compaction_count()` đã được viết sẵn và luôn trả `0` — đó là hợp đồng, không phải TODO. Đừng sửa nó để "cho công bằng"; chính con số 0 này là bằng chứng Baseline không nén lịch sử.

Phân biệt hai cột token rất quan trọng cho phần phân tích sau này. `Agent tokens only` đo phần agent tự sinh ra; `Prompt tokens processed` đo phần ngữ cảnh agent phải mang theo. Baseline không nén, nên khi thread dài ra, `Prompt tokens processed` của nó tăng gần như tuyến tính — và đó chính là thứ `Long-Context Stress Benchmark` được thiết kế để làm lộ.

Một chi tiết dễ sai: hai bộ đếm phải được cộng dồn **theo từng lượt**, ngay trong `_reply_offline()`, chứ không phải tính lại một lần ở cuối thread. Nếu bạn chỉ tính ở cuối, các lượt trung gian biến mất và cột `Prompt tokens processed` sẽ thấp giả tạo ở cả hai agent, khiến phần so sánh mất ý nghĩa.

> **Kiểm tra tính trung thực của Baseline**
>
> Tạo hai thread cho cùng một `user_id`, cung cấp tên ở thread thứ nhất rồi hỏi lại ở thread thứ hai. Baseline phải không biết. Nếu nó trả lời đúng, bạn đang lưu session theo `user_id` hoặc vô tình đọc `User.md` — cả hai đều làm hỏng phép so sánh.

## 6. Hoàn thiện Advanced Agent

Mục tiêu: Advanced ghép đúng ba lớp memory và trả lời được câu hỏi recall trong thread mới.

#### Ba lớp memory nối với nhau ra sao

Khác Baseline, Advanced không chỉ giữ message: mỗi lượt đi qua một chuỗi xử lý, trong đó fact ổn định tách khỏi hội thoại và được ghi xuống đĩa.

#### Luồng một lượt của Advanced Agent

```mermaid
flowchart TD
  A[message người dùng] --> B[extract_profile_updates]
  B --> C[ghi User.md]
  B --> D[CompactMemoryManager.append]
  D --> E{vượt ngưỡng token?}
  E -- có --> F[summarize_messages, tăng compactions]
  E -- không --> G[giữ recent messages]
  F --> H[prompt = User.md + summary + recent]
  G --> H
  H --> I[sinh câu trả lời, cập nhật token]
```

Điều cần chú ý trong sơ đồ: `User.md` và compact memory là hai đường độc lập. Fact ổn định đi vào file markdown; phần hội thoại đi vào bộ nhớ nén. Nếu bạn gộp cả hai vào một chỗ, câu hỏi recall xuyên phiên sẽ phụ thuộc vào việc bản tóm tắt có giữ được tên và nghề nghiệp hay không — một canh bạc không cần thiết.

#### Sáu bước của `_reply_offline()`

Scaffold đã viết sẵn pseudocode; việc của bạn là biến nó thành code có kiểm tra:

- **1.** Trích fact ổn định từ message người dùng bằng `extract_profile_updates()`.
- **2.** Ghi những fact đó vào `User.md` qua `profile_store`.
- **3.** Đẩy message vào `compact_memory` để lớp compact tự quyết định có nén hay không.
- **4.** Ước lượng ngữ cảnh sẽ mang vào lượt này bằng `_estimate_prompt_context_tokens()`.
- **5.** Sinh câu trả lời bằng `_offline_response()`.
- **6.** Đẩy câu trả lời của agent vào `compact_memory` và cập nhật bộ đếm token.

Bước 2 là chỗ xử lý correction. Khi người dùng đổi nơi ở, `User.md` phải còn đúng một giá trị cho khóa đó — giá trị mới nhất. `edit_text()` sinh ra chính cho việc này; nếu bạn chỉ ghi thêm, file sẽ phình ra và câu hỏi "nơi ở hiện tại" sẽ có hai đáp án.

#### `_estimate_prompt_context_tokens()` — đo đúng thứ benchmark cần

Hàm này phải cộng ba thành phần: nội dung `User.md`, phần summary của compact memory, và các message gần nhất còn giữ nguyên văn. Scaffold ghi rõ ba thành phần này trong docstring.

Lý do phải đủ ba: `Prompt tokens processed` là cột chứng minh compact có tác dụng. Nếu bạn quên `User.md`, Advanced trông rẻ hơn thực tế ở hội thoại ngắn và bạn sẽ kết luận sai. Nếu bạn quên summary, compact không bao giờ thể hiện được lợi ích. Chỉ khi đủ ba thành phần thì phép so sánh với Baseline mới có nghĩa.

#### `_offline_response()` — trả lời được câu hỏi recall

Ở chế độ offline, Advanced phải tất định và phải dùng memory đã lưu. Scaffold nêu các dạng câu hỏi tối thiểu:

- "Mình tên gì?"
- "Hiện tại mình làm nghề gì?"
- "Nhắc lại style trả lời mình thích"
- các câu hỏi trong bộ dữ liệu stress dài

Cách làm gọn nhất: đọc `User.md` đã lưu, tra theo khóa, rồi ghép câu trả lời. Với style trả lời, hãy giữ đúng yêu cầu mà người dùng đã nêu — stress test yêu cầu trả lời ngắn gọn thành 3 bullet và ưu tiên trade-off; câu trả lời của bạn phải phản ánh được yêu cầu đó, vì dữ liệu chấm recall tìm chuỗi "3 bullet" trong câu trả lời.

`_maybe_build_langchain_agent()` là phần mở rộng: dùng `build_chat_model(self.config.model)`, `InMemorySaver` cho short-term, một tool đọc `User.md`, một tool ghi/sửa `User.md`, prompt động chèn profile, và middleware tóm tắt cho thread dài. Làm phần này là điểm cộng, không phải điều kiện để benchmark chạy.

> **Hai con số phải dương**
>
> Sau một hội thoại có fact mới, `memory_file_size(user_id)` phải lớn hơn 0 và `compaction_count(thread_id)` phải tăng khi thread đủ dài. Cột `Memory growth (bytes)` trong benchmark lấy từ con số thứ nhất; `Compactions` lấy từ con số thứ hai.

## 7. Viết benchmark cho hai bộ dữ liệu

Mục tiêu: một lệnh chạy ra hai bảng, mỗi bảng so sánh Baseline và Advanced trên cùng input với đủ sáu cột.

#### Vì sao phải có hai bộ dữ liệu

Hai bộ dữ liệu không phải để chạy cho nhiều, mà để trả lời hai câu hỏi khác nhau.

`data/conversations.json` là **Standard Benchmark**: 10 hội thoại khoảng 10 lượt của user `dungct`, kèm `recall_questions`. Bộ này đo khả năng nhớ qua nhiều phiên bình thường. Ở đây hội thoại chưa đủ dài để compact có lợi, nên đây là nơi bạn sẽ thấy Advanced có thể **tốn hơn** Baseline về token.

`data/advanced_long_context.json` là **Long-Context Stress Benchmark**: một hội thoại 16 lượt của user `dungct_stress`, trong đó có nhiều đoạn tin tức dài được cố tình nhét vào. Mục tiêu của bộ này là làm lộ chi phí ngữ cảnh của Baseline, và cho thấy compact chủ yếu tối ưu ở `Prompt tokens processed`.

Điểm chung bắt buộc: `recall_questions` luôn được hỏi ở **thread mới**. Đây là điều làm nên "cross-session": nếu bạn hỏi trong chính thread vừa chat, Baseline cũng trả lời đúng và bạn không đo được gì.

#### Sáu hàm cần hoàn thiện

- **1.** `load_conversations(path)` — đọc JSON và trả về danh sách hội thoại. Mỗi hội thoại có `id`, `user_id`, `turns`, `recall_questions`; mỗi câu hỏi có `question` và `expected_contains`.
- **2.** `recall_points(answer, expected)` — trả `0`, `0.5` hoặc `1` tùy số chuỗi trong `expected` xuất hiện trong câu trả lời. Thang ba mức này giữ được thông tin "đúng một nửa" thay vì gộp thành sai hoàn toàn.
- **3.** `heuristic_quality(answer, expected)` — điểm chất lượng nhẹ cho chế độ offline, dùng khi không có judge model thật.
- **4.** `run_agent_benchmark(agent_name, agent, conversations, config)` — đưa toàn bộ lượt vào agent, cộng dồn token, hỏi recall trong thread mới, rồi tính trung bình recall và quality; ghi thêm mức tăng file memory và số compaction.
- **5.** `format_rows(rows)` — in bảng markdown hoặc dùng `tabulate`.
- **6.** `main()` — nạp cả hai dataset, khởi tạo Baseline và Advanced, chạy và in hai bảng.

Điểm công bằng cần giữ: **cùng một bộ dữ liệu, cùng thứ tự lượt, cùng câu hỏi recall cho cả hai agent**. Nếu bạn cho Advanced chạy trên dữ liệu đã lọc hoặc hỏi recall ở thread khác nhau, bảng so sánh mất giá trị.

Với vòng hỏi recall, hãy cấp một `thread_id` **mới** cho mỗi hội thoại trước khi hỏi, và dùng cùng cách đặt id đó cho cả hai agent. Cách này giữ đúng yêu cầu của `README.md` — `recall_questions` được hỏi ở thread mới — đồng thời bảo đảm bạn không vô tình hỏi trong chính thread vừa chat, nơi Baseline cũng trả lời đúng.

Về `recall_points()`, thang ba mức tồn tại để một câu trả lời đúng một nửa số fact không bị tính như sai hoàn toàn. `heuristic_quality()` thì phải dùng cùng một công thức cho cả hai agent; nếu mỗi agent được chấm một kiểu, cột `Response quality` không còn so sánh được nữa.

#### Sáu cột và ý nghĩa của từng cột

| Cột | Đo điều gì |
| --- | --- |
| `Agent tokens only` | Token agent sinh ra trực tiếp trong hội thoại |
| `Prompt tokens processed` | Lượng ngữ cảnh agent phải kéo theo qua các lượt |
| `Cross-session recall` | Khả năng nhớ fact qua thread hoặc session mới |
| `Response quality` | Chất lượng phản hồi |
| `Memory growth (bytes)` | Tốc độ phình của file memory |
| `Compactions` | Số lần compact đã nén lịch sử cũ |

`BenchmarkRow` trong scaffold đã có đúng sáu trường tương ứng; giữ nguyên tên trường để `format_rows()` in ra không lệch cột.

Kết quả mong đợi của lệnh `python src/benchmark.py`: hai bảng **Standard Benchmark** và **Long-Context Stress Benchmark**, mỗi bảng hai dòng (Baseline, Advanced) và đủ sáu cột. Nếu một trong hai bảng thiếu, hoặc chỉ có một agent, phần `main()` chưa nạp đủ dataset — hãy kiểm tra lại trước khi viết phân tích.

## 8. Viết test cho hành vi memory

Mục tiêu: bốn test chứng minh hành vi memory, chạy trên `tmp_path` và không phụ thuộc API key.

#### Vì sao test ở đây quan trọng hơn bình thường

`Guide.md` (Bước 7) nói rõ: phần test là thứ phân biệt giữa "trông có vẻ chạy được" và "đã kiểm chứng được hành vi memory". Với bài này điều đó đúng theo nghĩa kỹ thuật: một agent có thể in ra câu trả lời hợp lý mà `User.md` chưa từng được ghi, hoặc compact chưa từng kích hoạt. Test là chỗ duy nhất bạn phát hiện hai tình huống đó mà không phải đọc lại toàn bộ log.

`Rubric.md` (mốc 60–75) yêu cầu tối thiểu ba bài test: một cho `User.md`, một cho compact trigger, một cho cross-session recall. Scaffold đã khai báo bốn hàm, trong đó test thứ tư đo mức giảm prompt load — bài test này chính là bằng chứng định lượng cho luận điểm của cả track.

#### `make_config(tmp_path)` — nền của mọi test

Test phải chạy trong thư mục tạm, không ghi vào `state/` thật của repo, và phải ép compact xảy ra nhanh. Scaffold gợi ý đúng hai điều đó: trỏ `state_dir` vào `tmp_path` và giảm ngưỡng compact.

Lưu ý một chi tiết của scaffold: `LabConfig` không có giá trị mặc định cho các trường, nên `make_config()` phải điền đủ cả bảy trường — kể cả `model` và `judge_model`. File test sẽ cần import thêm `LabConfig` từ `config` và `ProviderConfig` từ `model_provider`. Nếu test dừng ngay dòng đầu với `TypeError: missing required positional argument`, bạn đang thiếu một trường trong danh sách này; hãy đối chiếu lại trước khi nghi ngờ phần memory. Mẫu tối thiểu để bạn so với bản của mình:

#### Bốn test và điều mỗi test phải khẳng định

- **1.** `test_user_markdown_read_write_edit` — ghi một profile, đọc lại thấy đúng nội dung, sửa một giá trị và kiểm tra `edit_text()` trả `True` cùng nội dung mới. Đây là test cho lớp persistent.
- **2.** `test_compact_trigger` — đẩy đủ message vào `CompactMemoryManager` với ngưỡng nhỏ trong `make_config()` để `compaction_count()` lớn hơn 0. Nếu test này xanh mà benchmark vẫn báo `Compactions = 0`, ngưỡng benchmark đang quá cao.
- **3.** `test_cross_session_recall` — cung cấp fact cho Advanced ở thread thứ nhất, hỏi lại ở thread thứ hai và khẳng định có nhớ; làm điều tương tự với Baseline và khẳng định **không** nhớ. Vế thứ hai quan trọng ngang vế thứ nhất.
- **4.** `test_compact_reduces_prompt_load_on_long_thread` — chạy một thread đủ dài cho cả hai agent và khẳng định prompt load của Advanced thấp hơn Baseline. Test này là bản thu nhỏ của luận điểm `README.md` nêu: ở hội thoại rất dài, compact giúp Advanced xử lý ngữ cảnh hiệu quả hơn.

Hãy viết test theo hành vi quan sát được, không theo cài đặt bên trong. Ví dụ ở test cross-session, hãy khẳng định câu trả lời của Advanced chứa tên đã cung cấp — đừng khẳng định `write_text` đã được gọi, vì như vậy test sẽ xanh ngay cả khi `User.md` rỗng.

Chạy toàn bộ:

```bash
pytest src/test_agents.py -v
```

Kết quả mong đợi: cả bốn test được thu thập và pass, không test nào cần API key. Nếu một test báo `TypeError` khi dựng config, quay lại `make_config()`; nếu `test_cross_session_recall` fail ở vế Baseline, Baseline đang lưu session sai khóa.

### Code gợi ý: Viết test cho hành vi memory

Code gợi ý của Codelab — hãy tự viết trước rồi mới đối chiếu.

#### `make_config.py`

```python
def make_config(tmp_path: Path):
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,   # nhỏ để compact kích hoạt sớm
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )
```

## 9. Chạy benchmark và đọc đúng sáu cột

Mục tiêu: hai bảng số liệu hiện ra, và bạn đọc được chúng theo đúng bốn kết luận mà `Rubric.md` gọi là "câu chuyện rõ ràng".

#### Chạy lại trên trạng thái sạch

Xóa `state/` trước khi chạy để chắc chắn không còn `User.md` từ lần thử trước, rồi chạy benchmark:

```bash
rm -rf state
python src/benchmark.py
```

Trên Windows PowerShell, xóa bằng `Remove-Item -Recurse -Force state`. Lý do phải làm sạch: `User.md` là file bền vững, nên nếu lần chạy trước đã ghi fact, lần này Advanced có thể recall đúng ngay từ lượt đầu và bạn không còn đo được đường cong tăng trưởng memory.

Vì đường offline là tất định, hai lần chạy liên tiếp trên cùng trạng thái sạch phải cho ra cùng bộ số. Nếu chúng lệch nhau, có gì đó không tất định đã lọt vào nhánh offline — ví dụ câu trả lời phụ thuộc thời gian, hoặc thứ tự message không ổn định — và phần phân tích phía sau sẽ không tái lập được.

Bạn cần thấy hai bảng, và trong mỗi bảng, hai dòng Baseline/Advanced đủ sáu cột. Số cụ thể phụ thuộc ngưỡng compact và estimator token bạn chọn, nên đừng so với một bảng mẫu; hãy đọc theo quan hệ giữa các dòng.

#### Bốn kết luận cần đọc ra từ bảng

`Rubric.md` mô tả chuỗi logic mà người chấm muốn thấy. Hãy đối chiếu từng mắt xích với số liệu của bạn:

- **1.** **Baseline không nhớ dài hạn** — `Cross-session recall` của Baseline thấp, và `Memory growth (bytes)` bằng 0 vì nó không ghi file nào.
- **2.** **Advanced thêm `User.md` nên recall tăng** — `Cross-session recall` của Advanced cao hơn rõ rệt, kèm `Memory growth (bytes)` dương.
- **3.** **Hội thoại dài làm prompt cost tăng mạnh** — trong bảng stress, `Prompt tokens processed` của Baseline tăng theo số lượt.
- **4.** **Compact kéo chi phí ngữ cảnh xuống** — ở cùng bảng stress, `Prompt tokens processed` của Advanced thấp hơn Baseline, và `Compactions` lớn hơn 0 chứng minh cơ chế đã thực sự chạy.

Mắt xích thứ năm của chuỗi đó nằm ở phần phân tích, không nằm trong bảng: hệ thống mạnh hơn nhưng cũng phức tạp hơn và cần guardrail tốt hơn.

#### Phép kiểm chứng bổ sung: tắt từng lớp memory

Một bảng chỉ thuyết phục khi bạn chỉ ra được lớp memory nào tạo ra khác biệt nào. Cách kiểm tra rẻ nhất là tắt một lớp rồi đo lại. Đặt `compact_threshold_tokens` lên rất cao để compact không bao giờ chạy, chạy lại benchmark, và so `Prompt tokens processed` của Advanced ở bảng stress. Nếu con số này tiến gần về mức của Baseline, compact đúng là thứ tạo ra lợi thế ở cột đó; nếu nó không đổi, compact của bạn chưa từng kích hoạt ngay cả ở hội thoại 16 lượt.

Cách đọc ngược lại cũng hữu ích: nếu `Agent tokens only` của Advanced không khác Baseline ở bảng Standard, nhiều khả năng nhánh ghi `User.md` đang bị bỏ qua, vì mỗi lượt Advanced còn phải mang profile vào ngữ cảnh.

Sau khi đo xong, nhớ trả ngưỡng compact về giá trị ban đầu. Để lại ngưỡng thử nghiệm nghĩa là `Compactions` bằng 0 trong bài nộp, và mốc 75–90 mất đi bằng chứng quan trọng nhất.

#### Bảng chẩn đoán khi số liệu vô lý

| Triệu chứng | Nguyên nhân thường gặp | Cách sửa |
| --- | --- | --- |
| `Compactions = 0` ở cả hai bảng | Ngưỡng compact quá cao, hoặc `append()` không kiểm tra ngưỡng | Giảm `compact_threshold_tokens`, kiểm tra lại nhánh nén trong `append()` |
| `Memory growth (bytes) = 0` cho Advanced | `extract_profile_updates()` trả rỗng, hoặc `_reply_offline()` không gọi `write_text()` | Kiểm tra regex trích fact và bước ghi trong `_reply_offline()` |
| Recall của hai agent bằng nhau | Câu hỏi recall đang được hỏi trong chính thread vừa chat | Hỏi recall ở thread mới, như dữ liệu quy định |
| Baseline recall cao bất thường | Session khóa theo `user_id` | Đổi khóa `self.sessions` sang `thread_id` |
| `Prompt tokens processed` của Advanced không thấp hơn | `_estimate_prompt_context_tokens()` thiếu thành phần summary, hoặc compact chưa chạy | Cộng đủ `User.md` + summary + recent messages |

> **Đừng chỉnh dữ liệu để bảng đẹp**
>
> `data/` là input dùng chung và không được sửa. Nếu bảng cho kết quả không như mong đợi, sửa `src/` hoặc ngưỡng compact — không lọc bớt hội thoại, không đổi câu hỏi recall, không xóa các lượt nhiễu.

## 10. Viết phân tích kết quả và chọn bonus

Mục tiêu: file phân tích trong repo giải thích được chính các con số bạn vừa đo, không phải mô tả lại tính năng.

`Rubric.md` mô tả rõ luồng logic mà reviewer phải nhìn thấy được trong bài: Baseline không nhớ dài hạn; Advanced thêm `User.md` nên recall tăng; hội thoại dài làm prompt cost tăng mạnh; compact kéo chi phí ngữ cảnh xuống; và cuối cùng hệ thống mạnh hơn nhưng cũng phức tạp hơn, cần guardrail tốt hơn.

Năm mắt xích đó không nằm rời nhau. Mỗi mắt xích phải được chống đỡ bằng một con số lấy từ bảng benchmark của bạn, cộng với cơ chế trong `src/` đã tạo ra con số đó. Một câu như "Advanced nhớ tốt hơn nhờ có memory" không chứng minh được gì; câu "recall của Advanced cao hơn Baseline ở cả hai bảng, trong khi `Memory growth (bytes)` của Baseline bằng 0 vì nó không ghi file nào" thì chứng minh được, vì nó chỉ vào đúng cột.

#### Bốn câu hỏi của Bước 8 và bằng chứng cho từng câu

`Guide.md` (Bước 8) yêu cầu bốn điểm. Với mỗi điểm, hãy nêu số liệu trước rồi mới giải thích:

- **Vì sao Advanced có recall tốt hơn Baseline** — dẫn `Cross-session recall` của hai agent ở cả hai bảng, rồi chỉ vào đường đi của fact: `extract_profile_updates()` trích fact, ghi vào `User.md`, và `_offline_response()` đọc lại file đó trong thread mới. Nếu recall của bạn không cao hơn, đừng viết phần này trước khi sửa lỗi.
- **Vì sao Advanced có thể tốn hơn ở hội thoại ngắn** — dẫn `Agent tokens only` và `Prompt tokens processed` ở bảng Standard. Ở đây bạn giải thích bằng chính những gì bạn đã code: mỗi lượt Advanced còn phải mang theo profile và có thể phải ghi file, trong khi hội thoại ngắn chưa đủ dài để compact bù lại phần chi phí đó.
- **Vì sao compact có lợi thế ở hội thoại dài** — dẫn bảng stress, và chỉ vào đúng cột `Prompt tokens processed`. `Rubric.md` yêu cầu nói rõ compact tối ưu cột này chứ không phải `Agent tokens only`; một bài nói "compact giúp tiết kiệm token" mà không tách hai cột sẽ không đạt mốc 75–90.
- **File memory tăng trưởng ra sao và rủi ro gì** — dẫn `Memory growth (bytes)` và `Compactions` trong bảng stress, rồi nêu rủi ro cụ thể mà bạn quan sát được: file phình theo thời gian, hoặc fact sai bị giữ lại sau một lượt nhiễu.

Một cách viết đủ chặt cho từng luận điểm là ba câu: số liệu nào, cơ chế nào trong code tạo ra nó, và giới hạn nào đi kèm. Cách này cũng buộc bạn nhìn lại bảng chẩn đoán ở phần trước — nếu một luận điểm không có số liệu chống đỡ, khả năng cao đó là lỗi cài đặt chứ không phải đặc điểm của thiết kế.

#### Chọn bonus theo tiêu chí của Rubric

`Rubric.md` mốc 90–100 chỉ nhận bốn hướng mở rộng: `Confidence threshold` trước khi ghi vào `User.md`, `Memory decay`, `Entity extraction` có cấu trúc hơn, và `Conflict handling` khi có correction mới. `Guide.md` (Bước 9) mô tả cùng nhóm hướng này, trong đó có việc tránh lưu sai khi người dùng đặt câu hỏi thay vì cung cấp fact.

Chọn một hướng gắn với điểm yếu bạn thực sự quan sát trong kết quả của mình. Nếu `User.md` phình nhanh, `Memory decay` hoặc `Confidence threshold` là lựa chọn có bằng chứng; nếu recall sai ở câu hỏi về nơi ở hiện tại, `Conflict handling` là lựa chọn có bằng chứng. Điều `Rubric.md` yêu cầu khi viết: bonus đó giải quyết vấn đề gì, nó cải thiện recall hoặc token cost thế nào, và nó tạo thêm rủi ro gì cho hệ thống. Thiếu mặt thứ ba — rủi ro — thì bonus không được tính là có giá trị, vì mọi cơ chế memory thêm vào đều đánh đổi bằng độ phức tạp.

#### Kiểm tra mốc điểm

Đối chiếu bài của bạn theo thứ tự bốn mốc trong `Rubric.md`:

- **1.** **0–60**: đủ Baseline, Advanced có `User.md`, có compact, có benchmark, cấu trúc repo rõ ràng.
- **2.** **60–75**: benchmark chạy cùng input cho cả hai agent; có test cho `User.md`, compact trigger và cross-session recall; bảng đủ sáu cột.
- **3.** **75–90**: có cả Standard và Long-Context Stress; stress đủ dài để lộ chi phí ngữ cảnh của Baseline; phân tích được vì sao compact không phải lúc nào cũng thắng ở hội thoại ngắn và vì sao nó chủ yếu tối ưu `Prompt tokens processed`.
- **4.** **90–100**: có bonus kèm đủ ba câu trả lời như trên.

Đặt phần phân tích vào **một file riêng trong repo bài làm** cùng với output benchmark. Tên file cụ thể chưa được nguồn quy định: `TODO — cần xác nhận` với giảng viên hoặc lab coach; nếu không có quy định riêng, dùng tên mô tả rõ nội dung và nêu tên đó ở bài nộp.

## 11. Nộp bài

Mục tiêu: repo bài làm đúng định dạng, chạy lại được từ trạng thái sạch, và link đã được nộp.

Bài này làm **cá nhân** và nộp bằng **link repo GitHub trên VLearn**. Kết quả được kiểm bằng chính hai lệnh mà `README.md` quy định cho bài, nên repo phải chạy được ở máy khác khi chỉ có `src/` và `data/`.

Đặt tên repo bài làm theo đúng mẫu; đây là thư mục gốc của bài nộp:

```text
KX-DAY17-HoVaTen-MSSV/
├── src/             # các file .py đã hoàn thiện
├── data/            # giữ nguyên input benchmark
├── <STEP8.md>  # Câu trả lời cho bước 8 trong phần GUIDE.md
└── README.md
```

Trước khi lấy link nộp, chạy lại hai lệnh kiểm tra từ thư mục gốc:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Lệnh thứ nhất phải in hai bảng đủ sáu cột; lệnh thứ hai phải pass toàn bộ bốn test. Repo nộp không được chứa `.env`, API key hay thư mục `state/` — `state/` là trạng thái sinh ra khi chạy, không phải phần bài làm. Một repo không chạy được ở máy khác vì thiếu `data/` hoặc vì phụ thuộc file còn sót trong `state/` sẽ không qua được lần kiểm đó, nên hãy thử lại hai lệnh sau khi đã xóa `state/`.

#### Trước khi nộp

- [ ] `python src/benchmark.py` in đủ Standard và Long-Context Stress.
- [ ] Mỗi bảng có hai dòng Baseline/Advanced và đủ sáu cột.
- [ ] `pytest src/test_agents.py -v` pass cả bốn test.
- [ ] Repo đặt tên `KX-DAY17-HoVaTen-MSSV` và không chứa `.env` hay `state/`.
- [ ] Có file phân tích riêng trả lời bốn câu hỏi của Bước 8.
- [ ] Đã dán link repo vào bài nộp trên VLearn.

## Nộp bài và đánh giá Lab

- **Deadline:** `2026-10-03T16:59:00+00:00`
- **Đánh giá:** Đánh giá Lab này
- **Link bài đã nộp:** `https://github.com/…`
- **Nút xác nhận:** Xác nhận đã nộp bài
