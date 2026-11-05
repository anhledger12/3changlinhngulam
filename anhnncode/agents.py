"""Step 2 - agents. An agent receives one sample, its transcripts and ONE ToolEnv, and acts only through
env.validate / env.call (the only methods allowed at inference besides get_tool_calls / get_state_hash).

  empty   always returns []: scores only the out-of-scope samples (sanity baseline)
  oracle  train only: replays label.tool_calls (should score 100; checks the runner + scorer end to end)
  llm     tool-calling loop against an OpenAI-compatible chat endpoint (vLLM / SGLang / llama.cpp server running
          locally, e.g. `vllm serve Qwen/Qwen2.5-3B-Instruct --enable-auto-tool-choice --tool-call-parser hermes`).
          The endpoint must be local/offline at scoring time. Decoding is temperature 0 with a fixed seed.
"""
from __future__ import annotations

import json
import time
import urllib.request


class Agent:
    name = "base"

    def run(self, sample: dict, transcripts: dict[str, str], env, deadline: float) -> dict:
        """Act on env. Returns free-form info for the log (e.g. final assistant text, steps)."""
        raise NotImplementedError


class EmptyAgent(Agent):
    name = "empty"

    def run(self, sample, transcripts, env, deadline):
        return {}


class OracleAgent(Agent):
    name = "oracle"

    def run(self, sample, transcripts, env, deadline):
        for c in sample["label"]["tool_calls"]:
            env.call(c["name"], c.get("arguments") or {})
        return {}


# ----------------------------------------------------------------------------------------------- LLM agent
SYSTEM_PROMPT = """Bạn là trợ lý giọng nói tiếng Việt điều khiển các ứng dụng của người dùng bằng cách gọi tool.
Nhiệm vụ: xử lý YÊU CẦU CUỐI của người dùng (lời nói đã được nhận dạng, có thể sai chính tả nhẹ).

Quy tắc:
- Chỉ gọi các tool được cung cấp. Lấy ID (device_id, contact_id, event_id, ...) từ kết quả tool đọc, không tự bịa.
- Gọi đúng và đủ các tool cần thiết; KHÔNG gọi thừa (mỗi call thừa bị trừ điểm). Không gọi lại tool đã có kết quả.
- Thời gian dùng ISO-8601 có múi giờ, tính theo thời điểm hiện tại bên dưới ("mai", "tối nay", "5 rưỡi" ...).
- Nếu yêu cầu thiếu thông tin bắt buộc mà không suy ra được: gọi sys.ask_missing_info với tool_name và params
  (tên tham số bắt buộc còn thiếu) rồi dừng.
- Nếu không app nào xử lý được yêu cầu: không gọi tool nào, trả lời ngắn gọn.
- Khi đã xong, trả lời ngắn gọn bằng tiếng Việt và không gọi thêm tool.

Bối cảnh:
{context}"""


def _fn_name(tool: str) -> str:
    return tool.replace(".", "__")  # OpenAI function names allow [a-zA-Z0-9_-] only


def _clean_schema(p):
    """Drop x-* extension keys from a JSON Schema (moved into the description where useful)."""
    if isinstance(p, dict):
        out = {k: _clean_schema(v) for k, v in p.items() if not k.startswith("x-")}
        hints = []
        if p.get("x-ref"):
            hints.append(f"ID lấy từ {p['x-ref']}")
        if p.get("x-match") == "fuzzy":
            hints.append("văn bản tự do")
        if hints and "type" in p:
            out["description"] = (p.get("description", "") + " (" + "; ".join(hints) + ")").strip()
        return out
    if isinstance(p, list):
        return [_clean_schema(x) for x in p]
    return p


def build_tools(sample: dict, catalog: dict[str, dict]) -> list[dict]:
    names = [n for a in sample["context"]["installed_apps"].values() for n in a.get("tools", [])]
    names.append("sys.ask_missing_info")
    tools = []
    for n in dict.fromkeys(names):
        spec = catalog.get(n)
        if not spec:
            continue
        params = _clean_schema(spec.get("parameters") or {"type": "object", "properties": {}})
        if params.get("x-one-of") or (spec.get("parameters") or {}).get("x-one-of"):
            groups = spec["parameters"]["x-one-of"]
            params["description"] = "Cần một trong các nhóm: " + " / ".join(" + ".join(g) for g in groups)
        desc = spec.get("description", "")
        if spec.get("returns"):
            desc += f" Trả về: {spec['returns']}"
        tools.append({"type": "function", "function": {"name": _fn_name(n), "description": desc, "parameters": params}})
    return tools


def render_context(sample: dict) -> str:
    c = sample["context"]
    keep = {k: c.get(k) for k in ("current_time", "timezone", "location", "user_profile")}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def history_messages(sample: dict, transcripts: dict[str, str]) -> list[dict]:
    msgs = []
    for t in sample.get("history", []):
        if t.get("role") == "user":
            msgs.append({"role": "user", "content": transcripts.get(t.get("audio"), "")})
        else:
            text = t.get("text") or ""
            if t.get("tool_calls"):  # already executed in the environment; shown as a summary, not re-run
                done = [{"call": c, "result": r} for c, r in zip(t["tool_calls"], t.get("tool_results") or [None] * len(t["tool_calls"]))]
                text = f"{text}\n[Đã thực hiện: {json.dumps(done, ensure_ascii=False)}]".strip()
            msgs.append({"role": "assistant", "content": text})
    return msgs


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, base_url: str, model: str, catalog: dict[str, dict], api_key: str = "EMPTY",
                 max_steps: int = 8, max_tokens: int = 512, seed: int = 0, request_timeout: float = 25.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.catalog, self.key = model, catalog, api_key
        self.max_steps, self.max_tokens, self.seed, self.request_timeout = max_steps, max_tokens, seed, request_timeout

    def _chat(self, messages, tools, timeout):
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "temperature": 0.0, "top_p": 1.0, "seed": self.seed, "max_tokens": self.max_tokens}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]

    def run(self, sample, transcripts, env, deadline):
        tools = build_tools(sample, self.catalog)
        fn2tool = {_fn_name(n): n for n in self.catalog}
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=render_context(sample))}]
        messages += history_messages(sample, transcripts)
        messages.append({"role": "user", "content": transcripts.get(sample["audio"], "")})
        info = {"steps": 0, "final_text": None}
        for step in range(self.max_steps):
            left = deadline - time.time()
            if left < 2:
                info["stopped"] = "deadline"
                break
            msg = self._chat(messages, tools, timeout=min(self.request_timeout, left))
            info["steps"] = step + 1
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                            else {"role": "assistant", "content": msg.get("content") or ""})
            if not calls:
                info["final_text"] = msg.get("content")
                break
            stop = False
            for c in calls:
                fn = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = None
                tool = fn2tool.get(fn, fn.replace("__", "."))
                if not isinstance(args, dict):
                    result = {"ok": False, "error": "arguments không phải JSON object"}  # not sent to env: no wasted call
                else:
                    result = env.call(tool, args)
                    stop |= tool == "sys.ask_missing_info"
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(result, ensure_ascii=False)})
            if stop:  # asking the user ends the turn
                break
        return info


def make_agent(name: str, catalog: dict[str, dict], **kw) -> Agent:
    if name == "empty":
        return EmptyAgent()
    if name == "oracle":
        return OracleAgent()
    if name == "llm":
        return LLMAgent(catalog=catalog, **kw)
    raise ValueError(name)
