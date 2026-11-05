"""Chuyển train.jsonl -> dữ liệu SFT dạng chat (messages + tools) để fine-tune SLM.

Chia tập:
  - val_unseen: mẫu có dùng tool của các app HOLDOUT (bị loại khỏi train) -> mô phỏng public test có app mới.
  - val_seen  : ~5% ngẫu nhiên phần còn lại.
  - train     : còn lại.
Mỗi mẫu train: system + history + user cuối + (assistant tool_call -> tool result)* + assistant text kết thúc.
Khi train chỉ tính loss trên các lượt assistant của YÊU CẦU CUỐI (trường "train_from" = index message bắt đầu).

    python lifemate/build_sft.py            # -> lifemate/data/sft_{train,val_seen,val_unseen}.jsonl + ids
    python lifemate/build_sft.py --full     # dùng toàn bộ train (khi đã chốt cấu hình, train bản nộp)
"""
import argparse
import json
import random
from pathlib import Path

from common import (call_msg, end_text, load_catalog, load_items, prompt_messages,
                    result_msg, tools_for)

OUT = Path(__file__).parent / "data"
# Nhãn lỗi thật (tìm bằng audit_data.py) -> loại khỏi dữ liệu train/val
EXCLUDE = {
    "train_1696",   # chat_ott.send_message tới chat_6142 không tồn tại (simulator: "không tìm thấy chat_id")
}
# App giữ lại làm "app chưa thấy" khi đánh giá (đều KHÔNG có trong public test, nên không mất gì cho public)
HOLDOUT = {"package_tracking", "library", "loyalty", "fitness", "lottery", "learning"}


def to_example(item, catalog, shuffle_tools=False, rng=None):
    tools = tools_for(item, catalog)
    if shuffle_tools:
        rng.shuffle(tools)
    msgs = prompt_messages(item)
    start = len(msgs)
    lab = item["label"]
    for c, r in zip(lab["tool_calls"], lab["tool_results"]):
        msgs.append(call_msg(c["name"], c["arguments"]))
        msgs.append(result_msg(r))
    msgs.append({"role": "assistant", "content": end_text(lab["tool_calls"])})
    return {"id": item["id"], "tools": tools, "messages": msgs, "train_from": start}


def apps_used(item):
    return {c["name"].split(".")[0] for c in item["label"]["tool_calls"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="không chia val, train trên tất cả")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copies", type=int, default=1, help="nhân bản mỗi mẫu với thứ tự tool xáo trộn")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    OUT.mkdir(exist_ok=True)
    catalog = load_catalog("train")
    items = [x for x in load_items("train") if x["id"] not in EXCLUDE]

    if a.full:
        splits = {"train": items}
    else:
        unseen = [x for x in items if apps_used(x) & HOLDOUT]
        rest = [x for x in items if not apps_used(x) & HOLDOUT]
        rng.shuffle(rest)
        k = max(1, len(rest) // 20)
        splits = {"train": rest[k:], "val_seen": rest[:k], "val_unseen": unseen}

    for name, xs in splits.items():
        with open(OUT / f"sft_{name}.jsonl", "w", encoding="utf-8") as f:
            for x in xs:
                n = a.copies if name == "train" else 1
                for i in range(n):
                    ex = to_example(x, catalog, shuffle_tools=i > 0, rng=rng)
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        (OUT / f"ids_{name}.txt").write_text("\n".join(x["id"] for x in xs) + "\n")
        print(f"{name:<11} {len(xs):>5} mẫu -> {OUT / f'sft_{name}.jsonl'}")


if __name__ == "__main__":
    main()
