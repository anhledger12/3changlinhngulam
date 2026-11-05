"""SFT bằng framework có sẵn: Unsloth + TRL SFTTrainer (cách CHÍNH). train_sft.py là phương án dự phòng.

    pip install -U unsloth trl flash-linear-attention causal-conv1d
    python lifemate/train_unsloth.py --model Qwen/Qwen3.5-4B --out lifemate/ckpt/q35-4b
    python lifemate/train_unsloth.py --model Qwen/Qwen3.5-4B --out lifemate/ckpt/q35-4b --export merged gguf

Framework lo: nạp model, LoRA, tiết kiệm VRAM (gồm cả logits vocab ~250k), vòng train, gộp LoRA, xuất GGUF.
Phần riêng của bài chỉ còn: render dữ liệu build_sft.py bằng chat template của model.
Không dùng QLoRA 4-bit cho Qwen3.5 (Unsloth khuyến cáo) -> nạp bf16.
"""
import argparse
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"


def render(path, tok, max_len, think):
    """messages + tools -> 1 chuỗi text theo chat template. Bỏ mẫu dài quá max_len token."""
    rows, dropped = [], 0
    for line in open(path, encoding="utf-8"):
        ex = json.loads(line)
        text = tok.apply_chat_template(ex["messages"], tools=ex["tools"], tokenize=False, enable_thinking=think)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > max_len:
            dropped += 1
            continue
        rows.append({"text": text})
    print(f"{Path(path).name}: {len(rows)} mẫu, bỏ {dropped} mẫu > {max_len} token")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train", default=str(DATA / "sft_train.jsonl"))
    ap.add_argument("--val", default=str(DATA / "sft_val_unseen.jsonl"))
    ap.add_argument("--out", default="lifemate/ckpt/q35-4b")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--think", action="store_true", help="giữ thinking khi render (mặc định tắt, khớp agent.py)")
    ap.add_argument("--export", nargs="*", default=[], choices=["merged", "gguf"],
                    help="merged = model bf16 cho vLLM; gguf = Q8_0 cho llama.cpp (RTX 3070)")
    a = ap.parse_args()

    from unsloth import FastModel                      # import unsloth TRƯỚC transformers/trl để nó vá tối ưu
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastModel.from_pretrained(
        a.model, max_seq_length=a.max_len, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    model = FastModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none",
        finetune_vision_layers=False, finetune_language_layers=True,   # chỉ train phần ngôn ngữ, bỏ vision encoder
        finetune_attention_modules=True, finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth", random_state=3407)

    text_tok = getattr(tok, "tokenizer", tok)          # Qwen3.5 có vision -> tok có thể là processor
    train = Dataset.from_list(render(a.train, text_tok, a.max_len, a.think))
    val = Dataset.from_list(render(a.val, text_tok, a.max_len, a.think)) if a.val and Path(a.val).exists() else None

    trainer = SFTTrainer(
        model=model, tokenizer=text_tok, train_dataset=train, eval_dataset=val,
        args=SFTConfig(
            output_dir=a.out, dataset_text_field="text", max_length=a.max_len, packing=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
            num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
            optim="adamw_8bit", weight_decay=0.01, logging_steps=10, bf16=True,
            eval_strategy="epoch" if val else "no", save_strategy="epoch", save_total_limit=2,
            report_to="none", seed=3407))
    # Chỉ tính loss trên lượt assistant (tool_call + câu kết). Template Qwen bọc kết quả tool trong lượt user
    # nên kết quả tool cũng bị che. Khác train_sft.py: các lượt assistant trong history cũng được học (vô hại).
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    trainer.train()

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("LoRA ->", a.out)
    if "merged" in a.export:
        model.save_pretrained_merged(a.out + "-merged", tok, save_method="merged_16bit")
        print("merged ->", a.out + "-merged")
    if "gguf" in a.export:
        model.save_pretrained_gguf(a.out + "-gguf", tok, quantization_method="q8_0")
        print("GGUF Q8_0 ->", a.out + "-gguf")


if __name__ == "__main__":
    main()
