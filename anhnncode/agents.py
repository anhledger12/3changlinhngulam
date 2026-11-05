"""Step 2 - agents. An agent receives one sample, its transcripts and ONE ToolEnv, and acts only through
env.validate / env.call (the only methods allowed at inference besides get_tool_calls / get_state_hash).

  empty   always returns []: scores only the out-of-scope samples (sanity baseline)
  oracle  train only: replays label.tool_calls (should score 100; checks the runner + scorer end to end)
  llm     tool-calling loop against an OpenAI-compatible chat endpoint (vLLM / SGLang / llama.cpp server running
          locally, e.g. `vllm serve Qwen/Qwen2.5-3B-Instruct --enable-auto-tool-choice --tool-call-parser hermes`).
          The endpoint must be local/offline at scoring time. Decoding is temperature 0 with a fixed seed.
  hf      same loop as `llm`, but the model runs in-process with transformers (no server needed, works on Windows).
"""
from __future__ import annotations

import json
import re
import time
import urllib.request


# ============================================================================================ Base + baseline agents
class Agent:
    """Interface of every agent."""
    name = "base"

    def run(self, sample: dict, transcripts: dict[str, str], env, deadline: float) -> dict:
        """Act on env. Returns free-form info for the log (e.g. final assistant text, steps)."""
        raise NotImplementedError


class EmptyAgent(Agent):
    """Never calls a tool. Correct only on out-of-scope samples, so it gives the floor score."""
    name = "empty"

    def run(self, sample, transcripts, env, deadline):
        return {}


class OracleAgent(Agent):
    """Replays the gold calls of a train sample. Should score 100: a check of the runner and scorer."""
    name = "oracle"

    def run(self, sample, transcripts, env, deadline):
        for c in sample["label"]["tool_calls"]:
            env.call(c["name"], c.get("arguments") or {})
        return {}


# ============================================================================================ Prompt building
# System prompt (Vietnamese): the rules of the task, then the user context (time, location, profile).
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
    """Catalog tool name -> function name the LLM sees ("sms.send" -> "sms__send")."""
    return tool.replace(".", "__")  # OpenAI function names allow [a-zA-Z0-9_-] only


def _clean_schema(p):
    """Drop x-* extension keys from a JSON Schema (moved into the description where useful)."""
    if isinstance(p, dict):
        out = {k: _clean_schema(v) for k, v in p.items() if not k.startswith("x-")}
        hints = []
        if p.get("x-ref"):
            hints.append(f"ID lấy từ {p['x-ref']}")  # "ID taken from <app data>"
        if p.get("x-match") == "fuzzy":
            hints.append("văn bản tự do")  # "free text"
        if hints and "type" in p:
            out["description"] = (p.get("description", "") + " (" + "; ".join(hints) + ")").strip()
        return out
    if isinstance(p, list):
        return [_clean_schema(x) for x in p]
    return p


def build_tools(sample: dict, catalog: dict[str, dict]) -> list[dict]:
    """OpenAI-style function definitions for the sample's installed apps, plus sys.ask_missing_info."""
    names = [n for a in sample["context"]["installed_apps"].values() for n in a.get("tools", [])]
    names.append("sys.ask_missing_info")
    tools = []
    for n in dict.fromkeys(names):  # de-duplicate, keep order
        spec = catalog.get(n)
        if not spec:
            continue
        params = _clean_schema(spec.get("parameters") or {"type": "object", "properties": {}})
        # x-one-of (at least one parameter group required) has no JSON Schema equivalent: say it in words
        if params.get("x-one-of") or (spec.get("parameters") or {}).get("x-one-of"):
            groups = spec["parameters"]["x-one-of"]
            params["description"] = "Cần một trong các nhóm: " + " / ".join(" + ".join(g) for g in groups)
        desc = spec.get("description", "")
        if spec.get("returns"):
            desc += f" Trả về: {spec['returns']}"  # "Returns: ..."
        tools.append({"type": "function", "function": {"name": _fn_name(n), "description": desc, "parameters": params}})
    return tools


def render_context(sample: dict) -> str:
    """The user context shown in the system prompt: current time, timezone, location, profile."""
    c = sample["context"]
    keep = {k: c.get(k) for k in ("current_time", "timezone", "location", "user_profile")}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def history_messages(sample: dict, transcripts: dict[str, str]) -> list[dict]:
    """Earlier turns as chat messages: user turns use their ASR transcript, assistant turns their text plus
    a summary of the calls they already made (those are already applied in the environment)."""
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


# ============================================================================================ LLM agent (server)
class LLMAgent(Agent):
    """Tool-calling loop: ask the model, run the calls it wants, feed back the results, repeat until it stops,
    asks for missing info, reaches max_steps or the deadline. The model is an OpenAI-compatible server."""
    name = "llm"

    def __init__(self, base_url: str, model: str, catalog: dict[str, dict], api_key: str = "EMPTY",
                 max_steps: int = 8, max_tokens: int = 512, seed: int = 0, request_timeout: float = 25.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.catalog, self.key = model, catalog, api_key
        self.max_steps, self.max_tokens, self.seed, self.request_timeout = max_steps, max_tokens, seed, request_timeout

    def _chat(self, messages, tools, timeout):
        """One model step: POST /chat/completions (temperature 0, fixed seed) and return the assistant message."""
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "temperature": 0.0, "top_p": 1.0, "seed": self.seed, "max_tokens": self.max_tokens}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]

    def run(self, sample, transcripts, env, deadline):
        # --- Build the conversation: system prompt + earlier turns + the final request
        tools = build_tools(sample, self.catalog)
        fn2tool = {_fn_name(n): n for n in self.catalog}
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=render_context(sample))}]
        messages += history_messages(sample, transcripts)
        messages.append({"role": "user", "content": transcripts.get(sample["audio"], "")})
        info = {"steps": 0, "final_text": None}

        # --- Loop: one model step per iteration
        for step in range(self.max_steps):
            left = deadline - time.time()
            if left < 2:  # not enough time for another step
                info["stopped"] = "deadline"
                break
            msg = self._chat(messages, tools, timeout=min(self.request_timeout, left))
            info["steps"] = step + 1
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                            else {"role": "assistant", "content": msg.get("content") or ""})
            if not calls:  # the model answered in text: done
                info["final_text"] = msg.get("content")
                break

            # --- Execute each requested call and give the result back to the model
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


# ============================================================================================ HF agent (in-process)
# Qwen-style tool call in the generated text: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class HFAgent(LLMAgent):
    """Same tool-calling loop as LLMAgent, but the model runs in-process with transformers (no server; works on
    Windows). Uses the model's own chat template with `tools`; tool calls are parsed from Qwen-style
    <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks. Greedy decoding, fixed seed."""
    name = "hf"

    def __init__(self, model: str, catalog: dict[str, dict], device: str = "cuda",
                 max_steps: int = 8, max_tokens: int = 512, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.catalog, self.max_steps, self.max_tokens, self.seed = catalog, max_steps, max_tokens, seed
        self.model_name, self.device = model, device
        self.request_timeout = float("inf")  # no network request; the loop's deadline still applies per step
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        dtype = torch.float16 if device == "cuda" else torch.float32  # fp16: GTX 1650 has no bf16
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device).eval()

    @staticmethod
    def _template_messages(messages: list[dict]) -> list[dict]:
        """Chat templates expect tool-call arguments as dicts (they JSON-encode them); the loop stores strings."""
        converted = []
        for message in messages:
            if message.get("tool_calls"):
                message = dict(message)
                message["tool_calls"] = [
                    {"type": "function", "function": {"name": call["function"]["name"],
                                                      "arguments": _parse_arguments(call["function"].get("arguments"))}}
                    for call in message["tool_calls"]]
            converted.append(message)
        return converted

    def _chat(self, messages, tools, timeout):
        """One model step: render the chat template, generate greedily, and return an OpenAI-style message."""
        import torch
        # --- Prompt -> generate
        prompt_ids = self.tokenizer.apply_chat_template(self._template_messages(messages), tools=tools,
                                                        add_generation_prompt=True, return_tensors="pt",
                                                        return_dict=True).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**prompt_ids, max_new_tokens=self.max_tokens, do_sample=False,
                                             temperature=None, top_p=None, top_k=None,
                                             pad_token_id=self.tokenizer.eos_token_id)
        new_ids = output_ids[0, prompt_ids["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        self.last_prompt_tokens = int(prompt_ids["input_ids"].shape[1])

        # --- Parse <tool_call> blocks into OpenAI-style tool_calls (invalid JSON blocks are skipped)
        tool_calls = []
        for index, block in enumerate(TOOL_CALL_PATTERN.findall(text)):
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            tool_calls.append({"id": f"call_{index}", "type": "function",
                               "function": {"name": parsed.get("name", ""),
                                            "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False)}})
        content = TOOL_CALL_PATTERN.sub("", text).strip()
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}

    def run(self, sample, transcripts, env, deadline):
        info = super().run(sample, transcripts, env, deadline)
        info["prompt_tokens"] = getattr(self, "last_prompt_tokens", None)  # prompt size of the last step, for the log
        return info


def _parse_arguments(arguments):
    """Tool-call arguments as a dict, whether given as a dict or a JSON string; {} if unparseable."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ============================================================================================ Factory
def make_agent(name: str, catalog: dict[str, dict], **kw) -> Agent:
    """Create an agent by its --agent name; kw are passed to the LLM agents' constructors."""
    if name == "empty":
        return EmptyAgent()
    if name == "oracle":
        return OracleAgent()
    if name == "llm":
        return LLMAgent(catalog=catalog, **kw)
    if name == "hf":
        return HFAgent(catalog=catalog, **kw)
    raise ValueError(name)
