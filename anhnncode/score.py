"""Step 3 - LifeMate metric (as specified in the competition document), computed on a labelled split (train).

Per sample:
  gold list empty      -> 1 if the submitted list is also empty, else 0
  otherwise            -> 0 if state_hash != gold state_hash or M == 0, else F1 = 2PR/(P+R)
                          P = M / (M + U), R = M / |G|
  Final score          =  100 / N * sum(score_i)

Call matching: same tool name, every non-fuzzy argument identical, every fuzzy argument (catalog `x-match: "fuzzy"`)
with token-F1 > 0.6 after lowercasing + Unicode NFC. One-to-one (maximum bipartite matching), order ignored.
U = submitted calls left unmatched, including calls rejected with a schema error.

State hash: like the organisers, the submitted tool_calls are REPLAYED on the initial state with the split's
zaic_tools and that hash is used (a mismatch with the submitted hash is reported). The gold hash is obtained by
replaying label.tool_calls the same way. (Scoring-only code: it uses zaic_tools.Executor, which inference may not.)

Interpretation choices where the document is silent (flagged here so they can be changed in one place):
  - an argument whose value is empty (None, "", [], {}) counts as absent (the validator treats it the same way);
  - non-fuzzy values compare with strict JSON equality (True != 1, list order matters, string case matters);
  - token-F1 uses word tokens (\\w+) of the NFC-lowercased text, counted as a multiset; both empty -> 1.

  python -m anhnn.score --split train --submission runs/train.oracle.na/submission.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .common import DEFAULT_ROOT, catalog_by_name, import_zaic_tools, load_catalog, load_samples, read_jsonl, utf8_stdout, write_jsonl

FUZZY_THRESHOLD = 0.6
EMPTY = (None, "", [], {})


# ----------------------------------------------------------------------------------------------- call matching
def tokens(v) -> list[str]:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())


def token_f1(a, b) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def strict_equal(a, b) -> bool:
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                       and not isinstance(a, bool) and not isinstance(b, bool)):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def fuzzy_params(catalog: dict[str, dict]) -> dict[str, set[str]]:
    return {n: {k for k, p in ((t.get("parameters") or {}).get("properties") or {}).items() if p.get("x-match") == "fuzzy"}
            for n, t in catalog.items()}


def calls_match(sub: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    if sub.get("name") != gold.get("name"):
        return False
    fz = fuzzy.get(gold["name"], set())
    sa = {k: v for k, v in (sub.get("arguments") or {}).items() if v not in EMPTY}
    ga = {k: v for k, v in (gold.get("arguments") or {}).items() if v not in EMPTY}
    for k in sa.keys() | ga.keys():
        if k in fz:
            if k not in sa or k not in ga or token_f1(sa[k], ga[k]) <= FUZZY_THRESHOLD:
                return False
        elif k not in sa or k not in ga or not strict_equal(sa[k], ga[k]):
            return False
    return True


def max_matching(sub: list[dict], gold: list[dict], fuzzy) -> int:
    """Size of a maximum one-to-one matching between submitted and gold calls (Kuhn's augmenting paths)."""
    adj = [[j for j, g in enumerate(gold) if calls_match(s, g, fuzzy)] for s in sub]
    owner = [-1] * len(gold)

    def augment(i, seen):
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] == -1 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(sub)))


def sample_score(sub_calls, gold_calls, hash_ok: bool, fuzzy) -> dict:
    if not gold_calls:
        s = 1.0 if not sub_calls else 0.0
        return {"score": s, "f1_nohash": s, "M": 0, "U": len(sub_calls), "G": 0, "P": None, "R": None}
    m = max_matching(sub_calls, gold_calls, fuzzy)
    u = len(sub_calls) - m
    p = m / (m + u) if m + u else 0.0
    r = m / len(gold_calls)
    f1 = 2 * p * r / (p + r) if m else 0.0
    return {"score": f1 if hash_ok else 0.0, "f1_nohash": f1, "M": m, "U": u, "G": len(gold_calls), "P": p, "R": r}


# ----------------------------------------------------------------------------------------------- categories
def category(sample: dict) -> str:
    gold = sample["label"]["tool_calls"]
    if not gold:
        return "out_of_scope"
    if any(c["name"] == "sys.ask_missing_info" for c in gold):
        return "ask_missing_info"
    if all(c["name"].split(".")[0] in ("weather", "gold_price", "fuel_price", "exchange_rate", "lottery", "news")
           or c["name"].endswith((".get_status", ".list", ".search", ".get")) for c in gold):
        return "read_only"
    return "action"


def main(argv=None):
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--all", action="store_true", help="score over every labelled sample (missing ones = empty list); "
                                                       "default: only samples present in the submission")
    ap.add_argument("--report", type=Path, default=None, help="per-sample JSONL (default: <submission dir>/score.jsonl)")
    a = ap.parse_args(argv)

    zt = import_zaic_tools(a.root, a.split)
    samples = load_samples(a.root, a.split)
    if any("label" not in s for s in samples):
        raise SystemExit(f"split {a.split!r} has no labels; the metric can only be computed on train "
                         f"(public/private test are scored by the organisers)")
    catalog = catalog_by_name(load_catalog(a.root, a.split))
    fuzzy = fuzzy_params(catalog)
    ex = zt.Executor(str(a.root / a.split / f"{a.split}_environment"), str(a.root / a.split / f"{a.split}_tools_catalog.json"))

    subs = {r["id"]: r for r in read_jsonl(a.submission)}
    if not a.all:
        samples = [s for s in samples if s["id"] in subs]
    if not samples:
        raise SystemExit("no submitted ids match the split")

    rows, warn = [], Counter()
    for s in samples:
        sub = subs.get(s["id"], {"tool_calls": [], "state_hash": None})
        sub_calls = sub.get("tool_calls") or []
        gold_calls = s["label"]["tool_calls"]

        gs = ex.open(s["id"])
        for c in gold_calls:
            gs.call(c["name"], c.get("arguments") or {})
        gold_hash = gs.get_state_hash()
        gold_state = gs.get_state()
        if any(gold_state.get(app) != st for app, st in s["label"].get("final_state", {}).items()):
            warn["gold replay != label.final_state"] += 1

        ss = ex.open(s["id"])
        rejected = sum(ss.call(c.get("name", ""), c.get("arguments") or {}).get("ok") is False for c in sub_calls)
        replay_hash = ss.get_state_hash()
        if sub.get("state_hash") is not None and sub["state_hash"] != replay_hash:
            warn["submitted state_hash != replayed (replayed used)"] += 1

        r = sample_score(sub_calls, gold_calls, replay_hash == gold_hash, fuzzy)
        r.update(id=s["id"], category=category(s), history=bool(s.get("history")), hash_ok=replay_hash == gold_hash,
                 rejected=rejected, n_sub=len(sub_calls))
        rows.append(r)

    n = len(rows)
    final = 100.0 * sum(r["score"] for r in rows) / n
    tie = 100.0 * sum(r["f1_nohash"] for r in rows) / n
    print(f"\nLifeMate score on {a.split}: {final:.2f}   (N={n}{', all samples' if a.all else ''})")
    print(f"  tie-break F1 (no state check): {tie:.2f}")
    print(f"  state hash match: {sum(r['hash_ok'] for r in rows)}/{n}   "
          f"calls: submitted={sum(r['n_sub'] for r in rows)} matched={sum(r['M'] for r in rows)} "
          f"unmatched(U)={sum(r['U'] for r in rows)} rejected={sum(r['rejected'] for r in rows)} gold={sum(r['G'] for r in rows)}")

    def table(key, title):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        print(f"\n  by {title}:")
        for k, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
            print(f"    {str(k):<18} n={len(g):<5} score={100 * sum(x['score'] for x in g) / len(g):6.2f}  "
                  f"hash_ok={sum(x['hash_ok'] for x in g) / len(g):6.1%}")
    table("category", "category")
    table("history", "has history")
    for w, c in warn.items():
        print(f"  WARNING: {w}: {c}")

    rep = a.report or a.submission.parent / "score.jsonl"
    write_jsonl(rep, rows)
    print(f"\n  per-sample report -> {rep}")
    return final


if __name__ == "__main__":
    main()
