"""Shared helpers: data paths, JSONL I/O, catalog access and importing the split's own zaic_tools."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

# Folder that holds the split folders (train/, public_test/, later private_test/). Override with --root.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "train+public_test"


def utf8_stdout() -> None:
    """Windows consoles default to cp1252; Vietnamese text would crash print()."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_samples(root: Path, split: str) -> list[dict]:
    return list(read_jsonl(root / split / f"{split}.jsonl"))


def load_catalog(root: Path, split: str) -> dict:
    with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return json.load(f)


def catalog_by_name(catalog: dict) -> dict[str, dict]:
    return {t["name"]: t for a in catalog["apps"] for t in a.get("tools", [])}


def import_zaic_tools(root: Path, split: str):
    """Import <root>/<split>/zaic_tools. Each split ships its own version, so use ONE split per process."""
    split_dir = str((root / split).resolve())
    loaded = sys.modules.get("zaic_tools")
    if loaded is not None:
        if not str(Path(loaded.__file__).resolve()).startswith(split_dir):
            raise RuntimeError(f"zaic_tools already imported from {loaded.__file__}; run one split per process")
        return loaded
    sys.path.insert(0, split_dir)
    import zaic_tools  # noqa: E402
    return zaic_tools
