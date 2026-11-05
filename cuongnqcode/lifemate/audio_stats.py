"""Đo chất lượng audio (chỉ cần numpy + soundfile) để quyết định có cần denoise không.

    .venv/bin/python lifemate/audio_stats.py      # -> lifemate/data/audio_stats.jsonl, audio_specs.json

Mỗi file (16kHz mono): thời lượng, peak, tỉ lệ clipping, mức RMS (dBFS), DC offset,
nhiễu nền (phân vị 10% năng lượng khung), mức tiếng nói (phân vị 95%), SNR ước lượng = hiệu hai mức,
tỉ lệ khung im lặng, im lặng đầu/cuối, băng thông hiệu dụng (tần số chứa 99% năng lượng),
tỉ lệ năng lượng < 100Hz (ù điện / gió), tốc độ nói (ký tự/giây, chỉ train).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from common import load_items, split_dir  # noqa: E402

OUT = Path(__file__).parent / "data"
FR, HOP, NFFT = 400, 160, 512          # khung 25ms, bước 10ms ở 16kHz


def frames(x):
    n = 1 + max(0, (len(x) - FR) // HOP)
    idx = np.arange(FR)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(FR)[None, :]


def analyze(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    dur = len(x) / sr
    peak = float(np.abs(x).max()) if len(x) else 0.0
    f = frames(np.pad(x, (0, max(0, FR - len(x)))))
    e_db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    noise, speech = np.percentile(e_db, 10), np.percentile(e_db, 95)
    voiced = e_db > noise + 0.5 * (speech - noise)          # ngưỡng giữa nhiễu nền và tiếng nói
    vi = np.flatnonzero(voiced)
    spec = np.abs(np.fft.rfft(f, NFFT)) ** 2                # (khung, 257)
    ps = spec.sum(0)
    cum = np.cumsum(ps) / (ps.sum() + 1e-12)
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)
    return {
        "dur": round(dur, 3), "sr": sr,
        "peak": round(peak, 4),
        "clip": round(float((np.abs(x) >= 0.999).mean()), 6),
        "rms_db": round(float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-10)), 2),
        "dc": round(float(x.mean()), 5),
        "noise_db": round(float(noise), 2), "speech_db": round(float(speech), 2),
        "snr": round(float(speech - noise), 2),
        "silence": round(float(1 - voiced.mean()), 3),
        "lead": round(vi[0] * HOP / sr, 3) if len(vi) else round(dur, 3),
        "trail": round(dur - (vi[-1] * HOP + FR) / sr, 3) if len(vi) else 0.0,
        "bw99": int(freqs[np.searchsorted(cum, 0.99)]),
        "low100": round(float(ps[freqs < 100].sum() / (ps.sum() + 1e-12)), 4),
    }, spec, e_db


def spectrogram_png_data(spec, w=320, h=80):
    """Spectrogram log rút gọn w x h, mỗi ô 0..255, để vẽ lên canvas trong trang báo cáo."""
    s = 10 * np.log10(spec + 1e-10).T                       # (tần số, thời gian)
    fi = np.linspace(0, s.shape[0] - 1, h).astype(int)
    ti = np.linspace(0, s.shape[1] - 1, min(w, s.shape[1])).astype(int)
    s = s[fi][:, ti][::-1]                                  # tần số cao ở trên
    lo, hi = np.percentile(s, 2), np.percentile(s, 99.5)
    return np.clip((s - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8).tolist()


def main():
    rows, keep = [], {}
    for split in ["train", "public_test"]:
        for x in load_items(split):
            turns = [("final", x["audio"], x.get("audio_text"))]
            turns += [("history", t["audio"], t.get("audio_text")) for t in x.get("history", []) if t["role"] == "user"]
            for kind, rel, text in turns:
                st, spec, _ = analyze(split_dir(split) / rel)
                st.update(id=x["id"], split=split, kind=kind, file=Path(rel).name)
                if text and st["dur"] > 0:
                    st["cps"] = round(len(text) / st["dur"], 2)
                rows.append(st)
                keep[f"{split}/{rel}"] = spec
            if len(rows) % 500 < len(turns):
                print(len(rows), flush=True)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "audio_stats.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Spectrogram mẫu: SNR thấp nhất / trung vị / cao nhất, clipping nặng nhất, băng thông hẹp nhất
    finals = [r for r in rows if r["kind"] == "final"]
    pick = {}
    by_snr = sorted(finals, key=lambda r: r["snr"])
    for lab, r in [("SNR thấp nhất", by_snr[0]), ("SNR thấp thứ 2", by_snr[1]), ("SNR thấp thứ 3", by_snr[2]),
                   ("SNR trung vị", by_snr[len(by_snr) // 2]), ("SNR cao nhất", by_snr[-1]),
                   ("Clipping nhiều nhất", max(finals, key=lambda r: r["clip"])),
                   ("Băng thông hẹp nhất", min(finals, key=lambda r: r["bw99"])),
                   ("Ù tần thấp nhiều nhất", max(finals, key=lambda r: r["low100"]))]:
        key = f"{r['split']}/audio/{r['id']}/{r['file']}"
        pick[lab] = {"key": f"{r['split']}/{r['file']}", "stats": r, "img": spectrogram_png_data(keep[key])}
    json.dump(pick, open(OUT / "audio_specs.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(rows)} file -> {OUT / 'audio_stats.jsonl'}")


if __name__ == "__main__":
    main()
