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

  python -m anhnncode.score --split train --submission runs/train.oracle.na/submission.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .common import DEFAULT_ROOT, catalog_by_name, import_zaic_tools, load_catalog, load_samples, read_jsonl, utf8_stdout, write_jsonl

FUZZY_THRESHOLD = 0.6  # a fuzzy argument matches when token-F1 > 0.6
EMPTY = (None, "", [], {})  # values treated as "argument not given"


# ============================================================================================ Argument comparison
def tokens(v) -> list[str]:
    """Word tokens of a value after Unicode NFC + lowercasing (non-strings are JSON-encoded first)."""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())


def token_f1(a, b) -> float:
    """Token-level F1 between two values (multiset overlap); 1.0 when both are empty."""
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def strict_equal(a, b) -> bool:
    """Exact JSON equality: same type (int/float may mix, bool may not), same keys, same list order."""
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                       and not isinstance(a, bool) and not isinstance(b, bool)):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def fuzzy_params(catalog: dict[str, dict]) -> dict[str, set[str]]:
    """{tool name: names of its parameters marked x-match: "fuzzy" in the catalog}."""
    return {n: {k for k, p in ((t.get("parameters") or {}).get("properties") or {}).items() if p.get("x-match") == "fuzzy"}
            for n, t in catalog.items()}


# ============================================================================================ Call matching
def calls_match(sub: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    """True when a submitted call matches a gold call: same tool, every argument present on both sides,
    fuzzy arguments with token-F1 > threshold, all other arguments strictly equal."""
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
    adj = [[j for j, g in enumerate(gold) if calls_match(s, g, fuzzy)] for s in sub]  # gold calls each sub call can take
    owner = [-1] * len(gold)  # owner[j] = submitted call currently matched to gold call j

    def augment(i, seen):
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] == -1 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(sub)))


# ============================================================================================ Per-sample score
def sample_score(sub_calls, gold_calls, hash_ok: bool, fuzzy) -> dict:
    """The document's per-sample formula. Also returns the F1 without the hash check (tie-break) and M/U/G/P/R."""
    if not gold_calls:  # out-of-scope sample: right only if nothing was called
        s = 1.0 if not sub_calls else 0.0
        return {"score": s, "f1_nohash": s, "M": 0, "U": len(sub_calls), "G": 0, "P": None, "R": None}
    m = max_matching(sub_calls, gold_calls, fuzzy)
    u = len(sub_calls) - m
    p = m / (m + u) if m + u else 0.0
    r = m / len(gold_calls)
    f1 = 2 * p * r / (p + r) if m else 0.0
    return {"score": f1 if hash_ok else 0.0, "f1_nohash": f1, "M": m, "U": u, "G": len(gold_calls), "P": p, "R": r}


# ============================================================================================ Sample categories
def category(sample: dict) -> str:
    """Rough type of a train sample from its gold calls, for the breakdown tables."""
    gold = sample["label"]["tool_calls"]
    if not gold:
        return "out_of_scope"
    if any(c["name"] == "sys.ask_missing_info" for c in gold):
        return "ask_missing_info"
    if all(c["name"].split(".")[0] in ("weather", "gold_price", "fuel_price", "exchange_rate", "lottery", "news")
           or c["name"].endswith((".get_status", ".list", ".search", ".get")) for c in gold):
        return "read_only"
    return "action"


# ============================================================================================ Command line
def main(argv=None):
    """Score a submission against the train labels; print the metric and write score.jsonl + CSVs."""
    utf8_stdout()

    # --- Arguments
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--all", action="store_true", help="score over every labelled sample (missing ones = empty list); "
                                                       "default: only samples present in the submission")
    ap.add_argument("--report", type=Path, default=None, help="per-sample JSONL (default: <submission dir>/score.jsonl)")
    a = ap.parse_args(argv)

    # --- Labels, catalog and the replay environment
    zt = import_zaic_tools(a.root, a.split)
    samples = load_samples(a.root, a.split)
    if any("label" not in s for s in samples):
        raise SystemExit(f"split {a.split!r} has no labels; the metric can only be computed on train "
                         f"(public/private test are scored by the organisers)")
    catalog = catalog_by_name(load_catalog(a.root, a.split))
    fuzzy = fuzzy_params(catalog)
    ex = zt.Executor(str(a.root / a.split / f"{a.split}_environment"), str(a.root / a.split / f"{a.split}_tools_catalog.json"))

    # --- Submission
    subs = {r["id"]: r for r in read_jsonl(a.submission)}
    if not a.all:
        samples = [s for s in samples if s["id"] in subs]
    if not samples:
        raise SystemExit("no submitted ids match the split")

    # --- Score every sample
    rows, warn = [], Counter()
    for s in samples:
        sub = subs.get(s["id"], {"tool_calls": [], "state_hash": None})
        sub_calls = sub.get("tool_calls") or []
        gold_calls = s["label"]["tool_calls"]

        # Gold hash: replay the gold calls on a fresh environment (and check it reproduces label.final_state)
        gs = ex.open(s["id"])
        for c in gold_calls:
            gs.call(c["name"], c.get("arguments") or {})
        gold_hash = gs.get_state_hash()
        gold_state = gs.get_state()
        if any(gold_state.get(app) != st for app, st in s["label"].get("final_state", {}).items()):
            warn["gold replay != label.final_state"] += 1

        # Submitted hash: replay the submitted calls, as the organisers do (count schema rejections)
        ss = ex.open(s["id"])
        rejected = sum(ss.call(c.get("name", ""), c.get("arguments") or {}).get("ok") is False for c in sub_calls)
        replay_hash = ss.get_state_hash()
        if sub.get("state_hash") is not None and sub["state_hash"] != replay_hash:
            warn["submitted state_hash != replayed (replayed used)"] += 1

        r = sample_score(sub_calls, gold_calls, replay_hash == gold_hash, fuzzy)
        r.update(id=s["id"], category=category(s), history=bool(s.get("history")), hash_ok=replay_hash == gold_hash,
                 rejected=rejected, n_sub=len(sub_calls), gold_text=s.get("audio_text"),
                 submitted_calls=sub_calls, gold_calls=gold_calls)
        rows.append(r)

    # --- Print the overall metric
    n = len(rows)
    final = 100.0 * sum(r["score"] for r in rows) / n
    tie = 100.0 * sum(r["f1_nohash"] for r in rows) / n
    print(f"\nLifeMate score on {a.split}: {final:.2f}   (N={n}{', all samples' if a.all else ''})")
    print(f"  tie-break F1 (no state check): {tie:.2f}")
    print(f"  state hash match: {sum(r['hash_ok'] for r in rows)}/{n}   "
          f"calls: submitted={sum(r['n_sub'] for r in rows)} matched={sum(r['M'] for r in rows)} "
          f"unmatched(U)={sum(r['U'] for r in rows)} rejected={sum(r['rejected'] for r in rows)} gold={sum(r['G'] for r in rows)}")

    # --- Print breakdowns
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

    # --- Write the per-sample report and the CSVs
    rep = a.report or a.submission.parent / "score.jsonl"
    write_jsonl(rep, rows)
    print(f"\n  per-sample report -> {rep}")

    sample_csv, summary_csv = export_csv(rows, rep.parent, rep.parent / "log.jsonl", warn)
    print(f"  CSV -> {sample_csv}\n         {summary_csv}")
    return final


# ============================================================================================ CSV export
# Column order of results.csv
SAMPLE_COLUMNS = ["id", "category", "history", "score", "f1_nohash", "hash_ok", "P", "R", "M", "U", "G",
                  "n_sub", "rejected", "sec", "over_time", "error", "gold_text", "transcript",
                  "submitted_calls", "gold_calls"]


def export_csv(rows: list[dict], out_dir: Path, log_path: Path, warnings: Counter) -> tuple[Path, Path]:
    """Write results.csv (one row per sample, joined with the inference log when present) and summary.csv
    (overall metric + breakdowns). UTF-8 with BOM so Excel shows Vietnamese correctly."""
    logs = {entry["id"]: entry for entry in read_jsonl(log_path)} if log_path.exists() else {}
    sample_csv, summary_csv = out_dir / "results.csv", out_dir / "summary.csv"

    # --- results.csv: one row per sample
    with open(sample_csv, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            log = logs.get(row["id"], {})
            record = {key: row.get(key) for key in SAMPLE_COLUMNS}
            record.update(sec=log.get("sec"), over_time=log.get("over_time"), transcript=log.get("transcript"),
                          error=(log.get("error") or "").strip().splitlines()[-1] if log.get("error") else "")
            for key in ("score", "f1_nohash", "P", "R"):
                if isinstance(record[key], float):
                    record[key] = round(record[key], 4)
            for key in ("submitted_calls", "gold_calls"):
                record[key] = json.dumps(record[key], ensure_ascii=False)
            writer.writerow(record)

    # --- summary.csv: overall row, then one row per category and per has-history value
    def summarize(group_name: str, group_value, group: list[dict]) -> dict:
        timed = [logs[row["id"]]["sec"] for row in group if row["id"] in logs]
        return {"group": group_name, "value": group_value, "n": len(group),
                "score": round(100 * sum(row["score"] for row in group) / len(group), 2),
                "tie_break_f1": round(100 * sum(row["f1_nohash"] for row in group) / len(group), 2),
                "hash_match_rate": round(sum(row["hash_ok"] for row in group) / len(group), 4),
                "submitted_calls": sum(row["n_sub"] for row in group), "matched_calls": sum(row["M"] for row in group),
                "unmatched_calls": sum(row["U"] for row in group), "rejected_calls": sum(row["rejected"] for row in group),
                "gold_calls": sum(row["G"] for row in group),
                "avg_sec": round(sum(timed) / len(timed), 2) if timed else "",
                "max_sec": round(max(timed), 2) if timed else "",
                "over_30s": sum(bool(logs[row["id"]].get("over_time")) for row in group if row["id"] in logs)}

    summary_rows = [summarize("overall", "all", rows)]
    for key in ("category", "history"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        summary_rows += [summarize(key, value, group) for value, group in sorted(groups.items(), key=lambda kv: str(kv[0]))]
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
        for warning, count in warnings.items():  # scorer warnings go at the bottom
            writer.writerow({"group": "warning", "value": warning, "n": count})
    return sample_csv, summary_csv


if __name__ == "__main__":
    main()
