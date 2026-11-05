"""Step 1 - speech recognition. Transcribes every user turn (final request + history turns) of a split and caches
the result as JSONL {"audio": <relative path>, "text": ..., "sec": ...}, so the agent step never waits on ASR twice.

Backends
  oracle         train only: the gold `audio_text` (upper bound for the agent; no model needed)
  faster-whisper CTranslate2 Whisper / PhoWhisper-ct2 (pip install faster-whisper)
  hf             transformers ASR pipeline, e.g. vinai/PhoWhisper-medium (pip install transformers torch soundfile)

Decoding is deterministic: greedy/fixed beam, temperature 0, no temperature fallback.

  python -m anhnn.asr --split train --backend oracle
  python -m anhnn.asr --split public_test --backend faster-whisper --model large-v3-turbo --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import DEFAULT_ROOT, load_samples, read_jsonl, utf8_stdout, write_jsonl


def user_turns(sample: dict) -> list[tuple[str, str | None]]:
    """(audio path, gold transcript or None) for every user turn in history and the final request, in order."""
    out = [(t["audio"], t.get("audio_text")) for t in sample.get("history", []) if t.get("role") == "user" and t.get("audio")]
    out.append((sample["audio"], sample.get("audio_text")))
    return out


SAMPLE_RATE = 16000


def load_wav(wav_path: str):
    """Read a WAV as mono float32 at 16 kHz with soundfile (no PyAV / ffmpeg needed; the LifeMate audio is
    already 16 kHz mono 16-bit). Resamples with torchaudio only if a file has another rate."""
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio.functional as F
        audio = F.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


class Transcriber:
    def transcribe(self, wav_path: str) -> str:
        raise NotImplementedError


def _add_cuda12_dlls() -> None:
    """Windows: ctranslate2 (faster-whisper) needs the CUDA 12 cuBLAS/cuDNN DLLs, while torch ships CUDA 13.
    They come from the pip wheels nvidia-cublas-cu12 / nvidia-cudnn-cu12; put their bin folders on the DLL path."""
    import os
    import sys
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for base in getattr(nvidia, "__path__", []):
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            d = os.path.join(base, sub, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


class FasterWhisper(Transcriber):
    def __init__(self, model: str, device: str, compute_type: str | None, beam: int, prompt: str | None):
        _add_cuda12_dlls()
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device, compute_type=compute_type or ("float16" if device == "cuda" else "int8"))
        self.beam, self.prompt = beam, prompt

    def transcribe(self, wav_path: str) -> str:
        segs, _ = self.m.transcribe(load_wav(wav_path), language="vi", beam_size=self.beam, temperature=0.0,
                                    condition_on_previous_text=False, initial_prompt=self.prompt, vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()


class HFWhisper(Transcriber):
    def __init__(self, model: str, device: str, beam: int):
        import torch
        from transformers import pipeline
        torch.manual_seed(0)
        self.p = pipeline("automatic-speech-recognition", model=model,
                          device=0 if device == "cuda" else -1,
                          torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        self.kw = {"generate_kwargs": {"language": "vi", "task": "transcribe", "num_beams": beam, "do_sample": False}}

    def transcribe(self, wav_path: str) -> str:
        return self.p({"raw": load_wav(wav_path), "sampling_rate": SAMPLE_RATE}, **self.kw)["text"].strip()


def main(argv=None):
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", required=True)
    ap.add_argument("--backend", choices=["oracle", "faster-whisper", "hf"], required=True)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--prompt", default=None, help="initial prompt / domain vocabulary (faster-whisper)")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/transcripts/<split>.<backend>.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    samples = load_samples(a.root, a.split)[: a.limit]
    out = a.out or Path("runs/transcripts") / f"{a.split}.{a.backend}.jsonl"
    done = {r["audio"]: r for r in read_jsonl(out)} if out.exists() else {}

    if a.backend == "oracle":
        if any(s.get("audio_text") is None for s in samples):
            raise SystemExit("oracle backend needs audio_text, which only the train split has")
        model = None
    elif a.backend == "faster-whisper":
        model = FasterWhisper(a.model, a.device, a.compute_type, a.beam, a.prompt)
    else:
        model = HFWhisper(a.model, a.device, a.beam)

    rows, t0 = dict(done), time.time()
    todo = [(p, g) for s in samples for p, g in user_turns(s) if p not in rows]
    for i, (rel, gold) in enumerate(todo, 1):
        t = time.time()
        text = gold if model is None else model.transcribe(str(a.root / a.split / rel))
        rows[rel] = {"audio": rel, "text": text, "sec": round(time.time() - t, 3)}
        if i % 200 == 0 or i == len(todo):
            write_jsonl(out, rows.values())  # checkpoint, so a crash resumes where it stopped
            print(f"[asr] {i}/{len(todo)} files, {time.time() - t0:.0f}s")
    write_jsonl(out, rows.values())
    print(f"[asr] {len(rows)} transcripts -> {out}")


if __name__ == "__main__":
    main()
