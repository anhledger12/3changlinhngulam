"""Step 2 - batch inference: one ToolEnv per sample, agent acts, write submission.jsonl (+ per-sample log).

  python -m anhnncode.infer --split train --agent oracle --asr oracle --limit 50
  python -m anhnncode.infer --split public_test --agent llm --asr faster-whisper \
        --llm-url http://localhost:8000/v1 --llm-model Qwen/Qwen2.5-3B-Instruct --workers 8

Output (default runs/<split>.<agent>.<asr>/):
  submission.jsonl  one line per sample in input order: {"id", "tool_calls", "state_hash"}
  log.jsonl         per sample: elapsed seconds, every call with its result, agent info, errors
Resumable: samples already in log.jsonl are skipped (use --overwrite to start over).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agents import make_agent
from .common import (DEFAULT_ROOT, catalog_by_name, import_zaic_tools, load_catalog, load_samples, read_jsonl,
                     select_samples, utf8_stdout, write_jsonl)

TIME_LIMIT = 30.0  # seconds per sample (competition rule)
LLM_AGENTS = ("llm", "hf", "cuongnq")  # agents that need transcripts


# ============================================================================================ Environment wrapper
class RestrictedEnv:
    """Exposes only the methods allowed at inference (validate, call, get_tool_calls, get_state_hash),
    and records each call's result for the log."""

    def __init__(self, env):
        self._env = env
        self.trace: list[dict] = []  # every call with its result and duration, for log.jsonl

    def validate(self, name, arguments=None):
        return self._env.validate(name, arguments)

    def call(self, name, arguments=None):
        t = time.time()
        r = self._env.call(name, arguments)
        self.trace.append({"name": name, "arguments": arguments, "result": r, "sec": round(time.time() - t, 4)})
        return r

    def get_tool_calls(self):
        return self._env.get_tool_calls()

    def get_state_hash(self):
        return self._env.get_state_hash()


# ============================================================================================ One sample
def load_transcripts(path: Path | None) -> dict[str, str]:
    """{audio path: transcript} from an ASR cache file written by asr.py."""
    if path is None:
        return {}
    return {r["audio"]: r["text"] for r in read_jsonl(path)}


def run_one(zt, root: Path, split: str, sample: dict, agent, transcripts: dict,
            time_limit: float = TIME_LIMIT) -> tuple[dict, dict]:
    """Run the agent on one sample. Returns (submission line, log entry). An agent crash is logged, not raised:
    the calls made before the crash still form the submission."""
    t0 = time.time()
    env = RestrictedEnv(zt.ToolEnv(split, sample["id"], root=root))  # exactly one environment per sample
    info, err = {}, None
    try:
        info = agent.run(sample, transcripts, env, deadline=t0 + time_limit) or {}
    except Exception:
        err = traceback.format_exc(limit=3)
    sub = {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
    sec = round(time.time() - t0, 3)
    # over_time always uses the competition's 30 s, even when --time-limit is raised for a slow GPU
    log = {"id": sample["id"], "sec": sec, "over_time": sec > TIME_LIMIT, "transcript": transcripts.get(sample["audio"]),
           "calls": env.trace, "info": info, "error": err}
    return sub, log


# ============================================================================================ Command line
def main(argv=None):
    """Run an agent over a split and write submission.jsonl + log.jsonl."""
    utf8_stdout()

    # --- Arguments
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", required=True)
    ap.add_argument("--agent", choices=["empty", "oracle", "llm", "hf", "cuongnq"], required=True)
    ap.add_argument("--asr", default="oracle", help="transcript cache name: runs/transcripts/<split>.<asr>.jsonl")
    ap.add_argument("--transcripts", type=Path, default=None, help="explicit transcript file (overrides --asr)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated sample ids, or a file with one id per line")
    ap.add_argument("--workers", type=int, default=1, help="parallel samples (useful with a batching LLM server)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-max-steps", type=int, default=8)
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="model for --agent hf")
    ap.add_argument("--device", default="cuda", help="device for --agent hf / cuongnq")
    ap.add_argument("--cuongnq-backend", choices=["hf", "openai"], default="hf",
                    help="--agent cuongnq: hf = --hf-model (+ --lora) in-process; openai = --llm-url / --llm-model server")
    ap.add_argument("--lora", default=None, help="--agent cuongnq: LoRA adapter folder from cuongnqcode's training")
    ap.add_argument("--think", action="store_true", help="--agent cuongnq: enable Qwen3/3.5 thinking")
    ap.add_argument("--cuongnq-dir", type=Path, default=None, help="cuongnqcode/lifemate folder (default: next to anhnncode)")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                    help="per-sample agent deadline in seconds (competition: 30; raise it for slow dev GPUs, "
                         "samples over 30 s are still flagged)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    # --- Samples
    zt = import_zaic_tools(a.root, a.split)
    samples = select_samples(load_samples(a.root, a.split), a.ids, a.limit)

    # --- Transcripts (required for LLM agents; every selected sample must have one)
    tpath = a.transcripts or Path("runs/transcripts") / f"{a.split}.{a.asr}.jsonl"
    transcripts = {}
    if a.agent in LLM_AGENTS:
        if not tpath.exists():
            raise SystemExit(f"no transcripts at {tpath}; run: python -m anhnncode.asr --split {a.split} --backend ...")
        transcripts = load_transcripts(tpath)
        missing = [s["id"] for s in samples if s["audio"] not in transcripts]
        if missing:
            raise SystemExit(f"{len(missing)} samples have no transcript, e.g. {missing[:3]}")
    elif tpath.exists():
        transcripts = load_transcripts(tpath)

    # --- Agent
    catalog = catalog_by_name(load_catalog(a.root, a.split))
    kw = {}
    if a.agent == "llm":
        if not a.llm_model:
            raise SystemExit("--llm-model is required for --agent llm")
        kw = dict(base_url=a.llm_url, model=a.llm_model, max_steps=a.llm_max_steps, seed=a.seed)
    elif a.agent == "hf":
        kw = dict(model=a.hf_model, device=a.device, max_steps=a.llm_max_steps, seed=a.seed)
    elif a.agent == "cuongnq":
        use_server = a.cuongnq_backend == "openai"
        if use_server and not a.llm_model:
            raise SystemExit("--llm-model is required for --agent cuongnq --cuongnq-backend openai")
        kw = dict(model=a.llm_model if use_server else a.hf_model, backend=a.cuongnq_backend, lora=a.lora,
                  base_url=a.llm_url, think=a.think, device=a.device)
        if a.cuongnq_dir:
            kw["code_dir"] = a.cuongnq_dir
    agent = make_agent(a.agent, catalog, **kw)

    # --- Output folder + resume: skip samples present in both log.jsonl and submission.jsonl
    out = a.out_dir or Path("runs") / f"{a.split}.{a.agent}.{a.asr if a.agent in LLM_AGENTS else 'na'}"
    out.mkdir(parents=True, exist_ok=True)
    log_path, sub_path = out / "log.jsonl", out / "submission.jsonl"
    if a.overwrite:
        for p in (log_path, sub_path):
            p.unlink(missing_ok=True)
    subs = {r["id"]: r for r in read_jsonl(sub_path)} if sub_path.exists() else {}
    logged = {r["id"] for r in read_jsonl(log_path)} if log_path.exists() else set()
    todo = [s for s in samples if s["id"] not in (logged & set(subs))]
    print(f"[infer] {a.split}: {len(samples)} samples, {len(samples) - len(todo)} already done, agent={a.agent}, out={out}")

    # --- Run (in parallel with --workers); log each sample as soon as it finishes, checkpoint the submission
    t0, n, slow, errors = time.time(), 0, 0, 0
    with open(log_path, "a", encoding="utf-8", newline="\n") as flog, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(run_one, zt, a.root, a.split, s, agent, transcripts, a.time_limit) for s in todo]
        report_every = 10 if a.agent in LLM_AGENTS else 100
        for f in as_completed(futs):
            sub, log = f.result()
            subs[sub["id"]] = sub
            flog.write(json.dumps(log, ensure_ascii=False) + "\n")
            flog.flush()
            n += 1
            slow += log["over_time"]
            errors += log["error"] is not None
            if n % report_every == 0 or n == len(todo):
                write_jsonl(sub_path, [subs[s["id"]] for s in samples if s["id"] in subs])  # checkpoint
                print(f"[infer] {n}/{len(todo)}  {time.time() - t0:.0f}s  over-30s={slow}  errors={errors}")

    # --- Final submission, in input order
    write_jsonl(sub_path, [subs[s["id"]] for s in samples if s["id"] in subs])
    print(f"[infer] wrote {sub_path} ({len(subs)} lines); over-30s={slow}, errors={errors}")


if __name__ == "__main__":
    main()
