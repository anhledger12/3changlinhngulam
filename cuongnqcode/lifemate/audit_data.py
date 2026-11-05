"""Kiểm tra chất lượng dữ liệu BTC: tìm lỗi thật trước khi quyết định clean gì.

    python3 lifemate/audit_data.py [--split train] [--show 3]

Nguyên tắc: chỉ sửa/loại lỗi RÕ RÀNG (mâu thuẫn với simulator, thiếu file, sai schema).
KHÔNG "sửa" quy ước của đáp án (vd. có gọi contacts.search hay không) — test được gán nhãn cùng quy ước,
model cần học đúng quy ước đó dù mình thấy nó chưa hợp lý.
"""
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict

from common import load_catalog, load_items, split_dir, tool_env

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--show", type=int, default=3)
a = ap.parse_args()

items = load_items(a.split)
cat = load_catalog(a.split)
has_label = "label" in items[0]
issues = defaultdict(list)


def add(kind, qid, detail=""):
    issues[kind].append((qid, detail))


for x in items:
    qid = x["id"]
    d = split_dir(a.split)
    # 1. File audio
    if not (d / x["audio"]).exists():
        add("audio lượt cuối bị thiếu", qid, x["audio"])
    for t in x.get("history", []):
        if t["role"] == "user" and not (d / t["audio"]).exists():
            add("audio history bị thiếu", qid, t["audio"])
        if t["role"] == "assistant" and len(t.get("tool_calls") or []) != len(t.get("tool_results") or t.get("tool_calls") or []):
            add("history: số tool_calls != số tool_results", qid)
    # 2. Tool đã cài phải có trong catalog
    for app in x["context"]["installed_apps"].values():
        for n in app["tools"]:
            if n not in cat:
                add("tool đã cài nhưng không có trong catalog", qid, n)
    if not has_label:
        continue

    # 3. Văn bản
    txt = x["audio_text"]
    if not txt.strip():
        add("audio_text rỗng", qid)
    if txt != unicodedata.normalize("NFC", txt):
        add("audio_text chưa chuẩn Unicode NFC (dấu tiếng Việt tách rời)", qid, txt)
    if re.search(r"\s{2,}|^\s|\s$", txt):
        add("audio_text thừa khoảng trắng", qid, repr(txt))

    lab = x["label"]
    calls, results = lab["tool_calls"], lab["tool_results"]
    if len(calls) != len(results):
        add("label: số tool_calls != số tool_results", qid)
    installed = {n for app in x["context"]["installed_apps"].values() for n in app["tools"]} | {"sys.ask_missing_info"}

    # 4. Replay đáp án: mọi call phải hợp lệ, kết quả phải khớp tool_results, state khớp final_state
    env = tool_env(a.split, qid)
    for i, c in enumerate(calls):
        if c["name"] not in installed:
            add("label gọi tool KHÔNG được cài", qid, c["name"])
        r = env.call(c["name"], c["arguments"])
        if not r.get("ok"):
            add("label có call bị từ chối (sai schema)", qid, f"{c['name']}: {r.get('error')}")
        elif i < len(results) and json.dumps(r, sort_keys=True, ensure_ascii=False) != json.dumps(results[i], sort_keys=True, ensure_ascii=False):
            add("tool_results trong label khác kết quả simulator", qid, c["name"])
        if r.get("ok") and r.get("done") is False:
            add("label có call chạy nhưng không làm gì (done=false)", qid, f"{c['name']}: {r.get('message')}")
    st = env.get_state()
    for app, v in lab.get("final_state", {}).items():
        if json.dumps(st.get(app), sort_keys=True, ensure_ascii=False) != json.dumps(v, sort_keys=True, ensure_ascii=False):
            add("final_state khác state sau replay (so nguyên văn)", qid, app)

    # 5. Call lặp y hệt trong cùng đáp án
    keys = Counter(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls)
    for k, n in keys.items():
        if n > 1:
            add("label có call lặp y hệt", qid, k[:120])

    # 6. ask_missing_info: tool_name phải tồn tại, params phải là tham số thật của tool đó
    for c in calls:
        if c["name"] != "sys.ask_missing_info":
            continue
        tn, ps = c["arguments"].get("tool_name"), c["arguments"].get("params", [])
        if tn not in cat:
            add("ask_missing_info: tool_name không có trong catalog", qid, tn)
            continue
        props = cat[tn]["parameters"].get("properties", {})
        bad = [p for p in ps if p not in props]
        if bad:
            add("ask_missing_info: params không có trong schema", qid, f"{tn} {bad}")
        if tn not in installed:
            add("ask_missing_info: hỏi cho tool không được cài", qid, tn)

# 7. Trùng lặp / mâu thuẫn: cùng câu nói + cùng history + cùng bộ app nhưng đáp án khác tên tool
if has_label:
    groups = defaultdict(list)
    for x in items:
        key = (unicodedata.normalize("NFC", x["audio_text"]).lower().strip(),
               json.dumps([t.get("audio_text") for t in x["history"]], ensure_ascii=False),
               tuple(sorted(x["context"]["installed_apps"])))
        groups[key].append(x)
    for xs in groups.values():
        if len(xs) < 2:
            continue
        sigs = {tuple(c["name"] for c in x["label"]["tool_calls"]) for x in xs}
        kind = "MÂU THUẪN: câu giống hệt, chuỗi tool khác" if len(sigs) > 1 else "trùng lặp câu + chuỗi tool"
        add(kind, ",".join(x["id"] for x in xs), " | ".join(" > ".join(s) or "[]" for s in sigs)[:200])

print(f"Split {a.split}: {len(items)} mẫu\n")
if not issues:
    print("Không tìm thấy lỗi nào.")
for kind, lst in sorted(issues.items(), key=lambda kv: -len(kv[1])):
    ids = {q for q, _ in lst}
    print(f"[{len(lst):>4} lần / {len(ids):>4} mẫu] {kind}")
    for q, dt in lst[:a.show]:
        print(f"        {q}  {dt}")
