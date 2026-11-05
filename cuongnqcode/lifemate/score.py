"""Chấm local (ước lượng): mỗi mẫu điểm = F1 tool call nếu state_hash đúng, sai hash = 0.

Khác score.py cũ: tham số free-text (x-match: fuzzy trong catalog, vd. sms content, title...) so GẦN ĐÚNG
(độ giống chuỗi >= --fuzzy), còn lại so chính xác. Công thức F1 chính xác của BTC chưa công bố -> đây là xấp xỉ.

    python lifemate/score.py --pred lifemate/out/pred_val.jsonl --gold lifemate/data/gold_train.jsonl --split train
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

from common import load_catalog, load_jsonl, tool_env


def norm(s) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s:]", " ", s)).strip()


def fuzzy_params(catalog) -> dict[str, set]:
    return {n: {p for p, s in t["parameters"].get("properties", {}).items() if s.get("x-match") == "fuzzy"}
            for n, t in catalog.items()}


def same_call(p, g, fz, thr) -> bool:
    if p.get("name") != g.get("name"):
        return False
    pa, ga = p.get("arguments") or {}, g.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    for k, gv in ga.items():
        pv = pa[k]
        if k in fz.get(g["name"], ()):
            if SequenceMatcher(None, norm(pv), norm(gv)).ratio() < thr:
                return False
        elif json.dumps(pv, sort_keys=True, ensure_ascii=False) != json.dumps(gv, sort_keys=True, ensure_ascii=False):
            if isinstance(pv, list) and isinstance(gv, list) and sorted(map(str, pv)) == sorted(map(str, gv)):
                continue
            return False
    return True


def f1(pc, gc, fz, thr) -> float:
    if not pc and not gc:
        return 1.0
    if not pc or not gc:
        return 0.0
    used, tp = set(), 0
    for p in pc:
        for j, g in enumerate(gc):
            if j not in used and same_call(p, g, fz, thr):
                used.add(j); tp += 1; break
    if tp == 0:
        return 0.0
    pr, rc = tp / len(pc), tp / len(gc)
    return 2 * pr * rc / (pr + rc)


def category(gc):
    if not gc:
        return "out_of_scope"
    return "ask_missing_info" if any(c["name"] == "sys.ask_missing_info" for c in gc) else "action"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", default="train", help="lấy catalog để biết tham số fuzzy")
    ap.add_argument("--fuzzy", type=float, default=0.6)
    ap.add_argument("--only-pred-ids", action="store_true", default=True,
                    help="chỉ chấm các id có trong pred (vd. tập val)")
    ap.add_argument("--show-errors", type=int, default=5)
    a = ap.parse_args()

    fz = fuzzy_params(load_catalog(a.split))
    pred = {r["id"]: r for r in load_jsonl(a.pred)}
    gold = {r["id"]: r for r in load_jsonl(a.gold)}
    ids = [i for i in gold if i in pred] if a.only_pred_ids else list(gold)

    st = defaultdict(lambda: [0, 0.0, 0, 0.0])          # n, score, hash_ok, f1
    errs, total = [], 0.0
    bad_hash = 0
    for i in ids:
        g, p = gold[i], pred.get(i, {"tool_calls": [], "state_hash": None})
        cat = category(g["tool_calls"])
        # BTC replay tool_calls của bài nộp trên state ban đầu -> tự replay, không tin state_hash ghi trong file
        env = tool_env(a.split, i)
        for c in p["tool_calls"]:
            env.call(c["name"], c.get("arguments") or {})
        replay = env.get_state_hash()
        bad_hash += p.get("state_hash") != replay
        h = replay == g["state_hash"]
        f = f1(p["tool_calls"], g["tool_calls"], fz, a.fuzzy)
        s = f if h else 0.0
        st[cat][0] += 1; st[cat][1] += s; st[cat][2] += h; st[cat][3] += f
        total += s
        if s < 1:
            errs.append((i, cat, h, f, p["tool_calls"], g["tool_calls"]))

    if bad_hash:
        print(f"[!] {bad_hash} dòng có state_hash ghi trong file khác hash khi replay tool_calls")
    print(f"\nĐIỂM: {100 * total / max(len(ids), 1):.2f}  ({len(ids)} mẫu)")
    print(f"{'Loại':<18}{'n':>6}{'Điểm':>8}{'Hash':>8}{'F1':>7}")
    for c, (n, s, h, f) in sorted(st.items()):
        print(f"{c:<18}{n:>6}{100 * s / n:>8.2f}{100 * h / n:>7.1f}%{f / n:>7.3f}")
    for i, c, h, f, pc, gc in errs[:a.show_errors]:
        print(f"\n--- {i} [{c}] hash={'OK' if h else 'SAI'} F1={f:.2f}")
        print("  pred:", json.dumps(pc, ensure_ascii=False))
        print("  gold:", json.dumps(gc, ensure_ascii=False))


if __name__ == "__main__":
    main()
