"""Chạy agent trên một split -> file nộp (id, tool_calls, state_hash).

Backend:
  oracle : replay đáp án train (kiểm tra pipeline, phải ra 100 điểm)
  hf     : model HF/LoRA chạy bằng transformers (GPU; CPU chạy được nhưng rất chậm)
  openai : server OpenAI-compatible — nhanh nhất khi suy luận nhiều mẫu:
           vLLM (GPU lớn) : vllm serve <merged> --served-model-name lifemate --enable-auto-tool-choice \
                              --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 32768
           llama.cpp (3070 8GB, GGUF Q8_0): llama-server -m q35-4b-Q8_0.gguf --jinja -c 32768 -ngl 99 --port 8000

    python lifemate/agent.py --split train --ids lifemate/data/ids_val_unseen.txt --backend oracle --out lifemate/out/oracle.jsonl
    python lifemate/agent.py --split train --ids lifemate/data/ids_val_unseen.txt --backend hf --model <path_merged_or_base> [--lora <adapter>]
    python lifemate/agent.py --split public_test --backend openai --model lifemate --base-url http://localhost:8000/v1 \
        --transcripts lifemate/data/asr_public_test.json --out submission.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from common import (load_catalog, load_items, parse_tool_calls, prompt_messages, tool_env,
                    tools_for, call_msg, result_msg)

MAX_STEPS = 10


class OracleBackend:
    def __init__(self, items):
        self.labels = {x["id"]: x["label"]["tool_calls"] for x in items}

    def run(self, item, env, msgs, tools):
        for c in self.labels[item["id"]]:
            env.call(c["name"], c["arguments"])


class LMBackend:
    """Vòng lặp agent: model sinh tool_call -> env.call -> đưa kết quả lại -> lặp đến khi model trả lời text."""

    def generate(self, msgs, tools) -> str:
        raise NotImplementedError

    @staticmethod
    def unseen_ids(call, tools, msgs):
        """Tham số x-ref (ID lấy từ app khác) phải là giá trị đã xuất hiện trong kết quả tool trước đó.
        Trên train: 1277/1278 ID trong gold thỏa điều này -> chặn được gần như mọi ID bịa."""
        schema = next((t["function"]["parameters"] for t in tools if t["function"]["name"] == call["name"]), {})
        props = schema.get("properties", {})
        seen_text = "".join(m["content"] for m in msgs if m["role"] == "tool")
        bad = []
        for p, v in call["arguments"].items():
            if props.get(p, {}).get("x-ref"):
                bad += [u for u in (v if isinstance(v, list) else [v]) if f'"{u}"' not in seen_text]
        return bad

    @staticmethod
    def coerce(call, tools):
        """Ép kiểu theo schema: tool call dạng XML đọc '123456' thành số, trong khi schema cần string."""
        schema = next((t["function"]["parameters"] for t in tools if t["function"]["name"] == call["name"]), {})
        for p, v in call["arguments"].items():
            typ = schema.get("properties", {}).get(p, {}).get("type")
            if typ == "string" and isinstance(v, (int, float)) and not isinstance(v, bool):
                call["arguments"][p] = str(v)

    read_tools: set = set()                           # main() gán từ catalog (kind == "read")

    def run(self, item, env, msgs, tools):
        seen = Counter()
        for _ in range(MAX_STEPS):
            calls = parse_tool_calls(self.generate(msgs, tools))
            if not calls:
                return
            for c in calls[:1]:                       # mỗi lượt 1 call (đúng như lúc train)
                key = json.dumps(c, sort_keys=True, ensure_ascii=False)
                # Call ĐỌC lặp y hệt = không có thông tin mới -> model đang kẹt, dừng.
                # Call GHI lặp là hợp lệ (vd. đổi 2 vé = redeem 2 lần, train_0922/1447), chỉ chặn khi lặp quá 4 lần.
                limit = 1 if c["name"] in self.read_tools else 4
                if seen[key] >= limit:
                    return
                seen[key] += 1
                self.coerce(c, tools)
                msgs.append(call_msg(c["name"], c["arguments"]))
                v = env.validate(c["name"], c["arguments"])   # validate không bị ghi vào bài nộp
                if not v.get("ok"):                           # sai schema: báo lỗi cho model tự sửa, KHÔNG call
                    msgs.append(result_msg({"ok": False, "error": v.get("error"), "note": "chưa thực hiện, hãy sửa"}))
                    continue
                if bad := self.unseen_ids(c, tools, msgs[:-1]):  # ID bịa: không call, bắt model tra cứu trước
                    msgs.append(result_msg({"ok": False, "error": f"ID chưa tra cứu: {bad}",
                                            "note": "chưa thực hiện; hãy gọi tool tra cứu để lấy ID đúng"}))
                    continue
                msgs.append(result_msg(env.call(c["name"], c["arguments"])))
                if c["name"] == "sys.ask_missing_info":
                    return


class HFBackend(LMBackend):
    def __init__(self, model, lora=None, think=False, max_new_tokens=384):
        import torch
        from transformers import AutoTokenizer
        from train_sft import load_model
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = load_model(model, dtype, device_map="auto")
        self.think = think
        self.max_new_tokens = max_new_tokens * (8 if think else 1)
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.model.eval()

    def generate(self, msgs, tools):
        import torch
        ids = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_tensors="pt",
                                           enable_thinking=self.think).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)


class OpenAIBackend(LMBackend):
    """vLLM hoặc llama.cpp server (lệnh ở docstring đầu file)."""

    def __init__(self, model, base_url, think=False):
        from openai import OpenAI
        self.client, self.model, self.think = OpenAI(base_url=base_url, api_key="x"), model, think

    def generate(self, msgs, tools):
        wire = []
        for m in msgs:                                 # OpenAI API cần arguments dạng chuỗi + id
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"id": f"c{i}", "type": "function",
                                    "function": {"name": t["function"]["name"],
                                                 "arguments": json.dumps(t["function"]["arguments"], ensure_ascii=False)}}
                                   for i, t in enumerate(m["tool_calls"])]
            wire.append(m)
        r = self.client.chat.completions.create(
            model=self.model, messages=wire, tools=tools, temperature=0, max_tokens=3072 if self.think else 384,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.think}})
        msg = r.choices[0].message
        if msg.tool_calls:
            return "".join(f"<tool_call>{json.dumps({'name': t.function.name, 'arguments': t.function.arguments}, ensure_ascii=False)}</tool_call>"
                           for t in msg.tool_calls)
        return msg.content or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", help="file chứa danh sách id (mỗi dòng 1 id); bỏ trống = cả split")
    ap.add_argument("--backend", choices=["oracle", "hf", "openai"], default="oracle")
    ap.add_argument("--model")
    ap.add_argument("--lora")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--transcripts", help="json {audio_path: text} từ asr.py; bắt buộc với public/private test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--think", action="store_true", help="bật thinking của Qwen3/3.5 (chậm hơn; thử so điểm)")
    ap.add_argument("--out", default="lifemate/out/pred.jsonl")
    a = ap.parse_args()

    items = load_items(a.split)
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        items = [x for x in items if x["id"] in keep]
    items = items[:a.limit] if a.limit else items
    catalog = load_catalog(a.split)
    transcripts = json.load(open(a.transcripts, encoding="utf-8")) if a.transcripts else None

    backend = {"oracle": lambda: OracleBackend(items),
               "hf": lambda: HFBackend(a.model, a.lora, a.think),
               "openai": lambda: OpenAIBackend(a.model, a.base_url, a.think)}[a.backend]()
    backend.read_tools = {n for n, t in catalog.items() if t["kind"] == "read"}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for k, item in enumerate(items, 1):
            env = tool_env(a.split, item["id"])         # đúng 1 phiên / mẫu
            try:
                backend.run(item, env, prompt_messages(item, transcripts), tools_for(item, catalog))
            except Exception as e:                      # lỗi model/parse không được làm hỏng cả file nộp
                print(f"[!] {item['id']}: {e}")
            f.write(json.dumps({"id": item["id"], "tool_calls": env.get_tool_calls(),
                                "state_hash": env.get_state_hash()}, ensure_ascii=False) + "\n")
            if k % 20 == 0:
                print(f"{k}/{len(items)}", flush=True)
    print(f"{len(items)} dòng -> {a.out}")


if __name__ == "__main__":
    main()
