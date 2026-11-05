"""ASR tiếng Việt cho mọi file audio của một split (lượt cuối + các lượt user trong history).

    pip install -U transformers torch soundfile librosa jiwer
    python lifemate/asr.py --split public_test --model vinai/PhoWhisper-small   # -> lifemate/data/asr_public_test.json
    python lifemate/asr.py --split train --model vinai/PhoWhisper-small --eval  # đo WER so với audio_text

Output: {"audio/<id>/<id>.wav": "văn bản", ...} — agent.py --transcripts dùng file này.
"""
import argparse
import json
from pathlib import Path

from common import load_items, split_dir

OUT = Path(__file__).parent / "data"


def collect(items):
    """[(audio_path, audio_text_or_None)] gồm lượt cuối và các lượt user trong history."""
    out = []
    for x in items:
        out.append((x["audio"], x.get("audio_text")))
        out += [(t["audio"], t.get("audio_text")) for t in x.get("history", []) if t["role"] == "user"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--model", default="vinai/PhoWhisper-small")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--eval", action="store_true", help="tính WER (chỉ train có audio_text)")
    a = ap.parse_args()

    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=a.model, device=dev,
                   torch_dtype=torch.float16 if dev == 0 else torch.float32)

    pairs = collect(load_items(a.split))[:a.limit]
    paths = [str(split_dir(a.split) / p) for p, _ in pairs]
    texts = [r["text"].strip() for r in asr(paths, batch_size=a.batch, generate_kwargs={"language": "vi"})]

    OUT.mkdir(exist_ok=True)
    dst = OUT / f"asr_{a.split}.json"
    json.dump({p: t for (p, _), t in zip(pairs, texts)}, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{len(texts)} file -> {dst}")

    if a.eval:
        import jiwer
        ref = [r.lower() for _, r in pairs if r]
        hyp = [t.lower() for (_, r), t in zip(pairs, texts) if r]
        print(f"WER = {jiwer.wer(ref, hyp):.3f} trên {len(ref)} câu")


if __name__ == "__main__":
    main()
