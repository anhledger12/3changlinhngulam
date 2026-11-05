# anhnn — batch inference + metric

Run from `LifeMate/` (the folder holding `train+public_test/`). Core code is Python stdlib only; the ASR backends need
`faster-whisper` or `transformers`+`torch`, and the LLM agent needs an OpenAI-compatible server (vLLM etc.) running locally.

```
1. ASR (cached)   python -m anhnn.asr   --split <split> --backend oracle|faster-whisper|hf [--model ...]
2. Inference      python -m anhnn.infer --split <split> --agent empty|oracle|llm [--asr <backend>] [--llm-url ... --llm-model ...]
3. Metric         python -m anhnn.score --split train --submission runs/<run>/submission.jsonl [--all]
```

| File | What it does |
|---|---|
| `common.py` | paths (`--root` default `train+public_test/`), JSONL I/O, catalog, imports the split's own `zaic_tools` |
| `asr.py` | transcribes every user turn (history + final) → `runs/transcripts/<split>.<backend>.jsonl`; resumable |
| `agents.py` | `empty`, `oracle` (train: replays gold), `llm` (tool-calling loop, temperature 0, 30 s deadline) |
| `infer.py` | one `ToolEnv` per sample, behind a wrapper exposing only `validate/call/get_tool_calls/get_state_hash`; writes `submission.jsonl` + `log.jsonl` (calls, results, seconds, errors); resumable, `--workers N` |
| `score.py` | the metric from the competition document; per-category breakdown, tie-break F1, `score.jsonl` per sample |

Examples:

```
# end-to-end with gold transcripts on train (upper bound for the agent)
python -m anhnn.asr   --split train --backend oracle
python -m anhnn.infer --split train --agent llm --asr oracle --llm-url http://localhost:8000/v1 --llm-model Qwen/Qwen2.5-3B-Instruct --workers 8
python -m anhnn.score --split train --submission runs/train.llm.oracle/submission.jsonl

# public test submission
python -m anhnn.asr   --split public_test --backend faster-whisper --model large-v3-turbo
python -m anhnn.infer --split public_test --agent llm --asr faster-whisper --llm-url ... --llm-model ...
cd train+public_test && PYTHONPATH=public_test python -m zaic_tools --split public_test --root . --validate ../runs/public_test.llm.faster-whisper/submission.jsonl
```

Sanity checks (measured): oracle agent on train = 100.00; empty agent = 7.18 (111/1545 out-of-scope samples).

Metric notes: the state hash is taken from replaying the submitted calls (as the organisers do); the gold hash from
replaying `label.tool_calls`. The document leaves three details open; the choices are listed at the top of `score.py`
(empty values = absent, strict equality for non-fuzzy values, `\w+` tokens for token-F1).
