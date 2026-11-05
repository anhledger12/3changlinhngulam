"""Full flow on a labelled split: pick samples -> ASR -> agent -> LifeMate metric -> CSV.

  python -m anhnncode.fullflow --num-samples 200                      # 200-sample train pilot (default settings)
  python -m anhnncode.fullflow --num-samples 0                        # every train sample
  python -m anhnncode.fullflow --num-samples 200 --asr oracle         # agent only, gold transcripts

Each step runs in its own process so the GPU memory of the ASR model is freed before the LLM loads.
Everything lands in runs/<run-name>/: ids.txt, submission.jsonl, log.jsonl, score.jsonl, results.csv, summary.csv.
Re-running the same command resumes (ASR cache and inference log are reused).
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from .common import DEFAULT_ROOT, load_samples, utf8_stdout


# ============================================================================================ Sample selection
def pick_ids(root: Path, split: str, num_samples: int, seed: int) -> list[str]:
    """Random but reproducible subset (seeded), kept in file order; 0 = all samples."""
    ids = [sample["id"] for sample in load_samples(root, split)]
    if num_samples <= 0 or num_samples >= len(ids):
        return ids
    chosen = set(random.Random(seed).sample(ids, num_samples))
    return [sample_id for sample_id in ids if sample_id in chosen]


# ============================================================================================ Step runner
def run_step(title: str, module: str, args: list[str]) -> float:
    """Run `python -m anhnncode.<module> <args>` in a child process (frees its GPU memory when it exits).
    Stops the whole flow if the step fails. Returns the step's duration in seconds."""
    print(f"\n=== {title} ===", flush=True)
    started = time.time()
    command = [sys.executable, "-m", f"anhnncode.{module}", *args]
    print(" ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(f"step '{title}' failed with exit code {completed.returncode}")
    elapsed = time.time() - started
    print(f"--- {title} done in {elapsed / 60:.1f} min", flush=True)
    return elapsed


# ============================================================================================ Command line
def main(argv=None):
    """Run ASR -> agent -> metric + CSV for one set of samples."""
    utf8_stdout()

    # --- Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="train", help="must be labelled (train) for the metric")
    parser.add_argument("--num-samples", type=int, default=200, help="0 = all samples")
    parser.add_argument("--seed", type=int, default=0, help="seed for picking the samples")
    parser.add_argument("--asr", choices=["faster-whisper", "hf", "oracle"], default="faster-whisper")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--compute-type", default="int8_float16", help="faster-whisper compute type")
    parser.add_argument("--agent", choices=["hf", "cuongnq"], default="hf",
                        help="hf = this project's prompt; cuongnq = cuongnqcode's prompt/loop (for its fine-tuned models)")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora", default=None, help="--agent cuongnq: LoRA adapter folder")
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="per-sample agent deadline; the competition limit is 30 s")
    parser.add_argument("--run-name", default=None, help="default: <split>-<n>-<asr>-<agent>-<model>")
    args = parser.parse_args(argv)

    # --- Pick the samples and create the run folder (ids.txt records exactly which samples were used)
    ids = pick_ids(args.root, args.split, args.num_samples, args.seed)
    model_tag = args.hf_model.split("/")[-1]
    run_name = args.run_name or f"{args.split}-{len(ids)}-{args.asr}-{args.agent}-{model_tag}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ids_file = run_dir / "ids.txt"
    ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"run: {run_dir}  ({len(ids)} samples of {args.split})")

    # --- Step 1: ASR (transcripts are cached in runs/transcripts/ and reused across runs)
    timings = {}
    asr_args = ["--root", str(args.root), "--split", args.split, "--backend", args.asr, "--ids", str(ids_file)]
    if args.asr != "oracle":
        asr_args += ["--model", args.asr_model]
    if args.asr == "faster-whisper":
        asr_args += ["--compute-type", args.compute_type]
    timings["asr"] = run_step("1/3 ASR", "asr", asr_args)

    # --- Step 2: agent inference -> submission.jsonl + log.jsonl in the run folder
    infer_args = ["--root", str(args.root), "--split", args.split, "--agent", args.agent, "--asr", args.asr,
                  "--ids", str(ids_file), "--hf-model", args.hf_model, "--time-limit", str(args.time_limit),
                  "--out-dir", str(run_dir)]
    if args.lora:
        infer_args += ["--lora", args.lora]
    timings["inference"] = run_step("2/3 Agent inference", "infer", infer_args)

    # --- Step 3: metric -> score.jsonl, results.csv, summary.csv in the run folder
    timings["score"] = run_step("3/3 Metric + CSV", "score", [
        "--root", str(args.root), "--split", args.split, "--submission", str(run_dir / "submission.jsonl")])

    # --- Summary
    print("\n=== Full flow finished ===")
    for step, seconds in timings.items():
        print(f"  {step:<10} {seconds / 60:6.1f} min")
    print(f"  results: {run_dir / 'results.csv'}\n           {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
