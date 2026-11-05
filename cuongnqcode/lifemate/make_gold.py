"""Tạo gold.jsonl thật từ nhãn train: replay label.tool_calls bằng zaic_tools -> state_hash.

    python lifemate/make_gold.py                  # -> lifemate/data/gold_train.jsonl
"""
import json
from pathlib import Path

from common import load_items, tool_env

OUT = Path(__file__).parent / "data" / "gold_train.jsonl"


def main():
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for item in load_items("train"):
            env = tool_env("train", item["id"])
            for c in item["label"]["tool_calls"]:
                env.call(c["name"], c["arguments"])
            f.write(json.dumps({"id": item["id"], "tool_calls": env.get_tool_calls(),
                                "state_hash": env.get_state_hash()}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} dòng -> {OUT}")


if __name__ == "__main__":
    main()
