# LifeMate: ASR + Qwen3.5-4B tự gọi tool

```
audio .wav ─ASR (PhoWhisper)─► văn bản ─ITN─► Qwen3.5-4B + LoRA ◄──► zaic_tools (env.call)
                                                │ mỗi bước: sinh 1 tool_call
                                                │ ► kiểm schema ► kiểm ID ► call ► đọc kết quả
                                                ▼
                                submission.jsonl: id, tool_calls, state_hash
```

## 1. File

| File | Việc | Chạy ở đâu |
|---|---|---|
| `common.py` | Prompt (system + bảng ngày + tools + history), parse tool call dạng JSON và XML | — |
| `audit_data.py` | Kiểm tra chất lượng dữ liệu BTC (xem mục 3) | Local |
| `make_gold.py` | Đáp án train → `data/gold_train.jsonl` (có state_hash thật) | Local |
| `build_sft.py` | train.jsonl → dữ liệu SFT (format chuẩn TRL: `messages` + `tools`), loại nhãn lỗi, chia `train / val_seen / val_unseen` | Local |
| `score.py` | Chấm local: replay tool_calls → so hash, F1 có so gần đúng free-text | Local |
| `agent.py` | Vòng lặp agent: `oracle` / `hf` / `openai` (vLLM, llama.cpp) | GPU |
| **`train_unsloth.py`** | **SFT chính**: Unsloth + TRL `SFTTrainer`, xuất được bản merged hoặc GGUF | GPU ≥ 24GB |
| `train_sft.py` | SFT **dự phòng** (tự viết bằng HF Trainer), dùng khi Unsloth lỗi với model | GPU ≥ 24GB |
| `asr.py` | Chuyển audio thành văn bản (cả các lượt trong history) | GPU |

## 2. Phần cứng: dùng máy nào cho việc gì

| Việc | Máy | Lý do |
|---|---|---|
| Train Qwen3.5-4B | **vast.ai**, RTX 4090 / 3090 24GB (rẻ) hoặc A6000 48GB | LoRA bf16 cho 4B cần khoảng 10GB ở ngữ cảnh ngắn. Mẫu của mình dài khoảng 10–12k token nên cần hơn nhiều |
| Đánh giá, debug agent | **RTX 3070 8GB** ở nhà, chạy GGUF **Q8_0** (~4.5GB) bằng llama.cpp | bf16 (~8–9GB) không vừa 8GB |
| ASR PhoWhisper-small | 3070 hoặc vast.ai | Nhẹ |

**Không dùng QLoRA 4-bit cho Qwen3.5.** Hướng dẫn chính thức của Unsloth khuyên tránh, vì sai số lượng tử hóa của họ model này lớn bất thường. Vì vậy 3070 8GB không đủ để train bản 4B. Train trên vast.ai, rồi lượng tử hóa **sau khi train** để chạy trên 3070. Nếu muốn train trên 3070, chỉ có thể dùng Qwen3.5-2B (LoRA bf16 khoảng 5GB), nhưng với ngữ cảnh 12k vẫn rất chật.

**WSL hiện chưa thấy GPU** (không có `nvidia-smi`). Cài NVIDIA driver bản mới trên Windows rồi mở lại WSL, khi đó `nvidia-smi` phải hiện RTX 3070.

## 3. Làm sạch dữ liệu

`python3 lifemate/audit_data.py --split train` kiểm tra: thiếu audio, tool không có trong catalog, replay đáp án (call bị từ chối, `tool_results` khác simulator, `final_state` khác), call lặp, `ask_missing_info` hỏi sai tham số, câu trùng hoặc mâu thuẫn, lỗi Unicode và khoảng trắng.

**Kết quả (bản v3.2.0): dữ liệu sạch hơn dự đoán.**

| Phát hiện | Kết luận | Xử lý |
|---|---|---|
| `train_1696` gửi tin tới `chat_6142` không tồn tại | **Nhãn lỗi thật** | Loại trong `build_sft.py` (`EXCLUDE`) |
| `train_0922`, `train_1447` gọi `loyalty.redeem` 2 lần y hệt | **Không phải lỗi**: người dùng muốn đổi 2 vé | Giữ. Agent cho phép call ghi lặp lại |
| 0 call sai schema, 0 `tool_results` lệch, 1545/1545 `final_state` khớp, 0 file thiếu, public_test 0 lỗi | Cấu trúc sạch | — |

"Bẩn" thật sự nằm ở **lời nói**: phương ngữ ("rứa", "bữa", "tui", "nghe"), tên tiếng Anh xen vào ("Sarah Kim", "Northwind Trading"), số đọc bằng chữ, và lỗi ASR ở public/private test. Phần này **không được xóa**, vì test cũng y như vậy. Cách xử lý là:
- **Chuẩn hóa sau ASR (ITN):** số, tiền, giờ, số điện thoại → dạng chuẩn trước khi đưa vào model.
- **Train trên transcript ASR thật** (thay cho `audio_text` sạch) để model quen lỗi nhận dạng.
- **Không "sửa" quy ước của đáp án** (ví dụ khi nào gọi `contacts.search`). Đáp án test theo cùng quy ước, model phải học đúng như vậy.

## 4. Những gì đáng biết về Qwen3.5-4B

| Đặc điểm | Ảnh hưởng tới mình |
|---|---|
| **Kiến trúc lai**: 8 khối × (3 lớp Gated DeltaNet + 1 lớp attention) | Ngữ cảnh dài rẻ hơn hẳn transformer thường, hợp với prompt 10–12k token. Cần `pip install flash-linear-attention causal-conv1d` để chạy nhanh |
| LoRA phải phủ cả lớp DeltaNet | Unsloth tự xử lý. `train_sft.py` dự phòng dùng `target_modules="all-linear"` (danh sách `q_proj…` cũ bỏ sót 3/4 số lớp) |
| Ngữ cảnh 262k, tool call dạng **XML** (`qwen3_coder`) | `common.py` đọc được cả JSON lẫn XML. vLLM dùng `--tool-call-parser qwen3_coder` |
| **Thinking bật mặc định** | Mặc định mình **tắt** cho nhanh và gọn (`enable_thinking=False` khi train và khi chạy). Thêm `--think` để thử bật xem có tăng điểm với câu điều kiện hoặc nhiều bước không |
| Có vision encoder (đa phương thức) | `load_model` tự chọn đúng lớp model |
| Vocab ~250k | Logits cho cả chuỗi 12k token chiếm khoảng 12GB. Unsloth tự tối ưu; `train_sft.py` dùng `LeanTrainer` chỉ tính logits ở các vị trí cần học |

## 5. Quy trình

### 5.0 Local (đã chạy sẵn)
```bash
python3 lifemate/make_gold.py
python3 lifemate/build_sft.py      # val_unseen = 6 app bị giấu khỏi train → giống public test có app mới
```

### 5.1 Cài môi trường trên vast.ai
Chọn image PyTorch (CUDA 12.x), copy thư mục `zalo-ai-challenge/` lên, rồi:
```bash
pip install -U unsloth trl transformers peft accelerate openai jiwer librosa flash-linear-attention causal-conv1d vllm
```

### 5.2 Train và đánh giá
```bash
# SFT chính (Unsloth + TRL); --export merged gguf: xuất luôn bản cho vLLM và bản Q8_0 cho 3070
python lifemate/train_unsloth.py --model Qwen/Qwen3.5-4B --out lifemate/ckpt/q35-4b --export merged gguf
# Nếu Unsloth lỗi với model: dùng bản dự phòng
#   python lifemate/train_sft.py --model Qwen/Qwen3.5-4B --out lifemate/ckpt/q35-4b
#   python lifemate/train_sft.py --merge --model Qwen/Qwen3.5-4B --out lifemate/ckpt/q35-4b

vllm serve lifemate/ckpt/q35-4b-merged --served-model-name lifemate --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 32768 &
python lifemate/agent.py --split train --ids lifemate/data/ids_val_unseen.txt --backend openai --model lifemate \
    --out lifemate/out/val_unseen.jsonl
python lifemate/score.py --pred lifemate/out/val_unseen.jsonl --gold lifemate/data/gold_train.jsonl --show-errors 30
```
Lần đầu nên chạy `Qwen/Qwen3.5-4B` gốc (chưa fine-tune) làm mốc so sánh. Cả hai script train **chưa được chạy thử thật** (máy local không có GPU):
- `train_unsloth.py`: dùng `FastModel` (bộ nạp chung của Unsloth). Nếu nó không nhận Qwen3.5, đổi sang `FastVisionModel` theo notebook Qwen3.5 của Unsloth.
- `train_sft.py`: nếu lỗi `logits_to_keep`, thêm `--full-logits` (tốn VRAM hơn).

### 5.3 Mang về 3070: GGUF Q8_0
`train_unsloth.py --export gguf` đã xuất sẵn vào `lifemate/ckpt/q35-4b-gguf/`. Nếu dùng `train_sft.py` thì tự chuyển:
```bash
git clone https://github.com/ggml-org/llama.cpp && pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py lifemate/ckpt/q35-4b-merged --outtype q8_0 --outfile q35-4b-Q8_0.gguf
# trên máy 3070:
llama-server -m <file .gguf> --jinja -c 32768 -ngl 99 --port 8000
python lifemate/agent.py --split train --ids lifemate/data/ids_val_unseen.txt --backend openai --model lifemate --out lifemate/out/q8.jsonl
```
Dùng Q8_0, gần như không mất độ chính xác. Q4 tiết kiệm VRAM hơn nhưng dễ sai ID và số, mà ở bài này sai một ký tự là hash sai. Nhớ chấm lại sau khi lượng tử hóa để biết mất bao nhiêu điểm.

### 5.4 ASR và bài nộp
```bash
python lifemate/asr.py --split train --model vinai/PhoWhisper-small --eval      # đo WER
python lifemate/asr.py --split public_test --model vinai/PhoWhisper-small
python lifemate/agent.py --split public_test --backend openai --model lifemate \
    --transcripts lifemate/data/asr_public_test.json --out submission.jsonl
cd train+public_test/public_test && PYTHONPATH=. python -m zaic_tools --split public_test --root .. --validate ../../submission.jsonl
```
Khi đã chốt cấu hình, train lại trên toàn bộ dữ liệu: `build_sft.py --full`.

## 6. Agent đã có gì

| Cơ chế | Vì sao |
|---|---|
| Mỗi bước 1 call, đọc kết quả rồi mới đi tiếp | ID và điều kiện ("nếu mai mưa") chỉ biết được sau khi tra |
| **Kiểm schema trước khi call** (`env.validate` không bị ghi vào bài nộp) | Call bị từ chối vẫn bị tính vào F1. Sai thì trả lỗi để model tự sửa |
| **Chặn ID chưa tra** (tham số `x-ref` phải xuất hiện trong kết quả tool trước đó) | Trên train, 1277/1278 ID trong đáp án thỏa điều này. Chỉ chặn nhầm 3/1545 mẫu |
| Ép kiểu theo schema | Tool call dạng XML đọc `"123456"` thành số |
| **Bảng ngày 15 ngày** trong system prompt | 27% mẫu có tham số ngày giờ, SLM tính lịch rất kém |
| Dừng khi **call đọc** lặp y hệt, hoặc sau `ask_missing_info`. Call ghi được lặp (tối đa 4 lần) | Đổi 2 vé = `redeem` 2 lần là hợp lệ. Mô phỏng agent trên đáp án train: tái tạo đúng 1544/1545 (mẫu còn lại là nhãn lỗi) |

## 7. Kỹ thuật mới nên làm tiếp (theo thứ tự)

1. **Đọc lỗi trên val**: `score.py --show-errors 30`, phân loại lỗi (sai ID, sai giờ, thiếu hoặc thừa call đọc, quên hỏi lại) trước khi làm gì khác.
2. **Constrained decoding**: ép model chỉ sinh tool name trong số app đã cài và tham số đúng `type`/`enum`, dựng lại ràng buộc ở mỗi bước. vLLM hỗ trợ `guided_json` / xgrammar, llama.cpp hỗ trợ grammar.
3. **Chuẩn hóa sau ASR (ITN)**: "hai triệu tám" → 2800000, "không chín năm…" → 095…, "ba giờ chiều" → 15:00. Train SFT trên transcript ASR (thay cho `audio_text`) để model quen lỗi nhận dạng.
4. **Tăng dữ liệu bằng simulator**:
   - bỏ app cần dùng → đáp án `[]`;
   - xóa thông tin bắt buộc → `ask_missing_info`;
   - lật điều kiện mưa ↔ không mưa;
   - **đổi tên app/tool/tham số giả** để model buộc phải đọc schema (vì public test có 11 app mới).
5. **RL với phần thưởng kiểm chứng được (GRPO nhiều lượt)**, **dùng framework có sẵn** (verl / VerlTool cho agent nhiều lượt, hoặc TRL `GRPOTrainer`). Phần tự viết chỉ có hàm reward: chạy lời giải trong `zaic_tools` rồi tính hash × F1. Theo arXiv 2604.13515, tập câu hỏi cho GRPO nên **tách riêng** khỏi tập SFT (dùng phần train giữ lại hoặc dữ liệu tăng thêm). Đây là hướng mới nhất cho agent gọi tool cỡ nhỏ (EigenData, VerlTool, RC-GRPO, mtGRPO). Bài này hợp RL một cách hiếm có: `zaic_tools` cho phần thưởng **chính xác bằng công thức chấm** (đúng hash × F1), không cần reward model. Cách làm: SFT trước, sau đó GRPO, chỉ tính gradient trên token của model và che phần kết quả tool. Cần GPU 48–80GB và làm sau khi SFT đã ổn.
6. **Bỏ phiếu ở từng bước** cho call ghi: sinh 5–8 ứng viên, chọn call được nhiều phiếu nhất. Không tốn call trong env.
7. **Thử bật thinking** (`--think`) cho câu điều kiện hoặc nhiều bước, so điểm với thời gian chạy.

**Không được làm:** đọc thẳng `state.json` hoặc tạo lại `ToolEnv` để thử trước. README của BTC cấm việc này.

## Nguồn
- Qwen3.5-4B model card: https://huggingface.co/Qwen/Qwen3.5-4B
- Unsloth Qwen3.5 fine-tune (VRAM, không dùng QLoRA): https://unsloth.ai/docs/models/qwen3.5/fine-tune
- EigenData, synthetic data + verifiable-reward RL cho agent nhiều lượt: https://arxiv.org/abs/2601.22607
- VerlTool, agentic RL với tool: https://arxiv.org/pdf/2509.01055
- SFT-GRPO data overlap (tách dữ liệu SFT và GRPO): https://arxiv.org/abs/2604.13515
