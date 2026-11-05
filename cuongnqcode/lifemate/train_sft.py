"""Fine-tune SLM (mặc định Qwen3.5-4B) bằng LoRA bf16 trên dữ liệu từ build_sft.py. Chạy trên GPU >= 24GB (vast.ai).
Không dùng QLoRA 4-bit cho Qwen3.5: Unsloth khuyến cáo vì sai số lượng tử hoá lớn bất thường ở họ model này.

    pip install -U transformers peft accelerate flash-linear-attention causal-conv1d
    python lifemate/train_sft.py --model Qwen/Qwen3.5-4B --out lifemate/ckpt/q35-4b
    python lifemate/train_sft.py --merge --model Qwen/Qwen3.5-4B --out lifemate/ckpt/q35-4b   # gộp LoRA -> <out>-merged

Loss chỉ tính trên các lượt assistant của yêu cầu cuối (tool_call + câu kết), không tính system/tools/history/tool result.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)


def load_model(name, dtype, **kw):
    """Qwen3.5 là model đa phương thức (có vision encoder): thử CausalLM trước, không được thì ImageTextToText."""
    try:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(name, torch_dtype=dtype, **kw)


class LeanTrainer(Trainer):
    """Chỉ tính logits ở các vị trí có nhãn (logits_to_keep).
    Vocab Qwen3.5 ~250k: 12k token x 250k x 4 byte = ~12GB chỉ riêng logits nếu tính cả chuỗi -> OOM.
    Mỗi mẫu chỉ vài trăm token cần học nên cắt bớt tiết kiệm phần lớn VRAM. Yêu cầu batch size = 1."""

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        tgt = labels[0, 1:]
        keep = (tgt != -100).nonzero().squeeze(-1)      # vị trí t dự đoán token t+1
        out = model(**inputs, logits_to_keep=keep)
        loss = F.cross_entropy(out.logits[0].float(), tgt[keep])
        return (loss, out) if return_outputs else loss

DATA = Path(__file__).parent / "data"
A_START, A_END = "<|im_start|>assistant\n", "<|im_end|>"
# Tắt "thinking" của Qwen3/3.5 cho khớp lúc suy luận (agent.py mặc định cũng tắt). Template khác bỏ qua kwarg này.
TEMPLATE_KW = {"enable_thinking": False}


def assistant_spans(ex, tok, text):
    """Vị trí ký tự của các lượt assistant cần học. Cách chung cho mọi chat template (Qwen, Gemma, Llama...):
    render(msgs[:i], generation_prompt) là tiền tố, render(msgs[:i+1]) kết thúc lượt i -> phần chênh lệch là lượt i."""
    msgs, tools, spans = ex["messages"], ex["tools"], []
    for i in range(ex["train_from"], len(msgs)):
        if msgs[i]["role"] != "assistant":
            continue
        p0 = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)
        p1 = tok.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False, **TEMPLATE_KW)
        if not (text.startswith(p0) and text.startswith(p1) and len(p1) > len(p0)):
            return None                               # template không nhất quán theo tiền tố -> dùng cách dự phòng
        spans.append((len(p0), len(p1)))
    return spans


def qwen_spans(ex, text):
    """Dự phòng cho template kiểu ChatML (Qwen): tìm theo marker <|im_start|>assistant."""
    skip = sum(m["role"] == "assistant" for m in ex["messages"][:ex["train_from"]])
    spans, pos, k = [], 0, 0
    while (s := text.find(A_START, pos)) != -1:
        e = text.find(A_END, s) + len(A_END)
        if k >= skip:
            spans.append((s + len(A_START), e))
        pos, k = e, k + 1
    return spans


def encode(ex, tok, max_len):
    text = tok.apply_chat_template(ex["messages"], tools=ex["tools"], tokenize=False, **TEMPLATE_KW)
    spans = assistant_spans(ex, tok, text)
    if spans is None:
        spans = qwen_spans(ex, text)
        if not spans:
            raise ValueError("Không xác định được lượt assistant trong chat template của model này")
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    if len(enc["input_ids"]) > max_len:
        return None
    labels = [tid if any(a <= o0 < b for a, b in spans) else -100
              for tid, (o0, _) in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"], "labels": labels}


def load(path, tok, max_len):
    rows, dropped = [], 0
    for line in open(path, encoding="utf-8"):
        r = encode(json.loads(line), tok, max_len)
        if r is None:
            dropped += 1
        else:
            rows.append(r)
    lens = sorted(len(r["input_ids"]) for r in rows)
    print(f"{path.name}: {len(rows)} mẫu, bỏ {dropped} mẫu > {max_len} token, độ dài p50={lens[len(lens)//2]} max={lens[-1]}")
    return rows


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L], lab[i, :L], att[i, :L] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def merge(a):
    from peft import PeftModel
    base = load_model(a.model, torch.bfloat16)
    m = PeftModel.from_pretrained(base, a.out).merge_and_unload()
    dst = a.out.rstrip("/") + "-merged"
    m.save_pretrained(dst)
    AutoTokenizer.from_pretrained(a.out).save_pretrained(dst)
    print("->", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train", default=str(DATA / "sft_train.jsonl"))
    ap.add_argument("--val", default=str(DATA / "sft_val_unseen.jsonl"))
    ap.add_argument("--out", default="lifemate/ckpt/lora")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--full-logits", action="store_true", help="tắt LeanTrainer nếu model không hỗ trợ logits_to_keep")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge(a)

    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = load_model(a.model, torch.bfloat16 if bf16 else torch.float16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        # "all-linear": phủ cả lớp attention thường lẫn lớp Gated DeltaNet của Qwen3.5 (tên module khác q/k/v/o)
        target_modules="all-linear"))
    model.print_trainable_parameters()

    train = load(Path(a.train), tok, a.max_len)
    val = load(Path(a.val), tok, a.max_len) if a.val and Path(a.val).exists() else None
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_accum, bf16=bf16, fp16=not bf16, logging_steps=10,
        eval_strategy="epoch" if val else "no", save_strategy="epoch", save_total_limit=2,
        report_to="none", remove_unused_columns=False, group_by_length=True, prediction_loss_only=True)
    (Trainer if a.full_logits else LeanTrainer)(model=model, args=args, train_dataset=train, eval_dataset=val,
            data_collator=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("LoRA ->", a.out)


if __name__ == "__main__":
    main()
