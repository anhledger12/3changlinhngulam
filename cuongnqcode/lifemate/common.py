"""Phần dùng chung: đọc dữ liệu, dựng prompt (system + tools + history), parse tool call của model.

Format hội thoại theo chuẩn OpenAI/HF chat template (Qwen hiểu trực tiếp):
  system -> user -> assistant(tool_calls) -> tool -> assistant(tool_calls) -> tool -> ... -> assistant(text)
Mỗi lượt assistant gọi đúng 1 tool để model thấy kết quả trước khi quyết định bước tiếp.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # /home/.../zalo-ai-challenge


def _find_data_dir() -> Path:
    """Thư mục chứa train/, public_test/: biến môi trường LIFEMATE_DATA, hoặc train+public_test nằm cạnh
    thư mục code (bố cục gốc) hay cạnh thư mục cha (bố cục LifeMate/cuongnqcode/ + LifeMate/train+public_test/)."""
    import os
    if os.environ.get("LIFEMATE_DATA"):
        return Path(os.environ["LIFEMATE_DATA"])
    candidates = [ROOT / "train+public_test", ROOT.parent / "train+public_test"]
    return next((path for path in candidates if path.is_dir()), candidates[0])


DATA = _find_data_dir()                                 # thư mục chứa train/, public_test/

# Câu trả lời kết thúc lượt (model học để dừng gọi tool)
END_DONE = "Đã xong."
END_ASK = "Bạn cho mình xin thêm thông tin còn thiếu nhé."
END_OOS = "Xin lỗi, hiện không có ứng dụng nào hỗ trợ yêu cầu này."

SYSTEM_TEMPLATE = """Bạn là trợ lý giọng nói LifeMate. Nhiệm vụ: thực hiện yêu cầu tiếng Việt của người dùng bằng cách gọi tool của các app đã cài.
Quy tắc:
- Mỗi lượt gọi đúng 1 tool, đọc kết quả rồi mới quyết định bước tiếp. Chỉ dùng tool có trong danh sách.
- Không bịa ID: tra cứu trước (search/list/get_status...) để lấy id, rồi mới gọi tool ghi.
- Thời gian dùng định dạng ISO 8601 có múi giờ, ví dụ 2026-11-29T09:00:00+07:00. Hiểu "mai", "tối nay"... theo thời điểm hiện tại.
- Nếu thiếu tham số bắt buộc mà không suy ra được từ ngữ cảnh/dữ liệu: gọi sys.ask_missing_info(tool_name, params) rồi dừng.
- Nếu không app nào làm được yêu cầu: không gọi tool nào, trả lời ngắn rằng không hỗ trợ.
- Yêu cầu có điều kiện ("nếu mai mưa thì..."): kiểm tra điều kiện trước, không thỏa thì dừng.
- Làm xong thì trả lời ngắn, không gọi thêm tool.

Thời điểm hiện tại: {current_time} ({timezone})
Lịch tham chiếu:
{date_table}
Vị trí hiện tại: {location}
Người dùng: {profile}"""

_WD = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
_REL = {-1: "hôm qua", 0: "hôm nay", 1: "mai", 2: "ngày kia/mốt"}


def date_table(current_time: str) -> str:
    """Bảng ngày -> thứ cho 15 ngày tới: SLM rất yếu tính lịch, đưa sẵn để model chỉ cần tra."""
    from datetime import datetime, timedelta
    now = datetime.fromisoformat(current_time)
    mon = now.date() - timedelta(days=now.weekday())
    rows = []
    for d in range(-1, 14):
        day = now.date() + timedelta(days=d)
        week = (day - mon).days // 7
        tag = [_REL[d]] if d in _REL else []
        tag += [{0: "tuần này", 1: "tuần sau"}.get(week, "")]
        tag = ", ".join(t for t in tag if t)
        rows.append(f"- {day.isoformat()} {_WD[day.weekday()]}" + (f" ({tag})" if tag else ""))
    return "\n".join(rows)


def split_dir(split: str) -> Path:
    return DATA / split


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_items(split: str) -> list[dict]:
    return load_jsonl(split_dir(split) / f"{split}.jsonl")


def load_catalog(split: str) -> dict[str, dict]:
    cat = json.load(open(split_dir(split) / f"{split}_tools_catalog.json", encoding="utf-8"))
    return {t["name"]: t for a in cat["apps"] for t in a["tools"]}


def tool_env(split: str, qid: str):
    """ToolEnv chính thức của BTC (zaic_tools của đúng split)."""
    p = str(split_dir(split))
    if p not in sys.path:
        sys.path.insert(0, p)
    from zaic_tools import ToolEnv
    return ToolEnv(split, qid, root=DATA)


def tools_for(item: dict, catalog: dict[str, dict]) -> list[dict]:
    """Danh sách tool (format OpenAI) của các app đã cài + sys.ask_missing_info (luôn có)."""
    names = [n for app in item["context"]["installed_apps"].values() for n in app["tools"]]
    if "sys.ask_missing_info" not in names:
        names.append("sys.ask_missing_info")
    out = []
    for n in names:
        t = catalog[n]
        desc = t["description"]
        if t.get("returns"):
            desc += f" | Trả về: {t['returns']}"
        out.append({"type": "function",
                    "function": {"name": n, "description": desc, "parameters": t["parameters"]}})
    return out


def system_prompt(item: dict) -> str:
    c = item["context"]
    return SYSTEM_TEMPLATE.format(
        current_time=c["current_time"], timezone=c["timezone"], date_table=date_table(c["current_time"]),
        location=json.dumps(c["location"], ensure_ascii=False),
        profile=json.dumps(c["user_profile"], ensure_ascii=False))


def call_msg(name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


def result_msg(result) -> dict:
    return {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}


def user_text(turn_or_item: dict, transcripts: dict | None) -> str:
    """Lời người dùng: dùng ASR transcript nếu có, không thì audio_text (chỉ train có)."""
    audio = turn_or_item.get("audio")
    if transcripts is not None and audio in transcripts:
        return transcripts[audio]
    if "audio_text" in turn_or_item:
        return turn_or_item["audio_text"]
    raise KeyError(f"Không có transcript cho {audio}; chạy asr.py trước")


def history_messages(item: dict, transcripts: dict | None = None) -> list[dict]:
    msgs = []
    for turn in item.get("history", []):
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": user_text(turn, transcripts)})
            continue
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or [None] * len(calls)
        for c, r in zip(calls, results):
            msgs.append(call_msg(c["name"], c["arguments"]))
            msgs.append(result_msg(r if r is not None else {"ok": True}))
        msgs.append({"role": "assistant", "content": turn.get("text", "")})
    return msgs


def prompt_messages(item: dict, transcripts: dict | None = None) -> list[dict]:
    """system + history + yêu cầu cuối (chưa có câu trả lời)."""
    return ([{"role": "system", "content": system_prompt(item)}]
            + history_messages(item, transcripts)
            + [{"role": "user", "content": user_text(item, transcripts)}])


def end_text(calls: list[dict]) -> str:
    if not calls:
        return END_OOS
    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        return END_ASK
    return END_DONE


# ---------- parse output của model (format Qwen: <tool_call>{json}</tool_call>) ----------
_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


_FN = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def _xml_call(raw: str):
    """Format XML của Qwen3.5 / Qwen3-Coder / LFM3: <function=name><parameter=k>v</parameter></function>."""
    m = _FN.search(raw)
    if not m:
        return None
    args = {}
    for k, v in _PARAM.findall(m.group(2)):
        try:
            args[k] = json.loads(v)          # số, bool, list, object
        except json.JSONDecodeError:
            args[k] = v                      # chuỗi thuần
    return {"name": m.group(1), "arguments": args}


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for raw in _TC.findall(text):
        if raw.lstrip().startswith("<function="):
            if (c := _xml_call(raw)):
                calls.append(c)
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if obj.get("name"):
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls
