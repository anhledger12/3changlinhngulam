"""Shared helpers: data paths, JSONL I/O, catalog access and importing the split's own zaic_tools."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

# ============================================================================================ Paths
# Folder that holds the split folders (train/, public_test/, later private_test/). Override with --root.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "train+public_test"


# ============================================================================================ Console
def utf8_stdout() -> None:
    """Windows consoles default to cp1252; Vietnamese text would crash print()."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ============================================================================================ JSONL files
def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per non-empty line of a UTF-8 JSON Lines file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dicts as UTF-8 JSON Lines (Vietnamese kept as-is, Unix line endings); creates the parent folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================================================ Samples
def load_samples(root: Path, split: str) -> list[dict]:
    """All samples of a split, from <root>/<split>/<split>.jsonl, in file order."""
    return list(read_jsonl(root / split / f"{split}.jsonl"))


def parse_ids(value: str | None) -> set[str] | None:
    """--ids value: a file with one id per line, or a comma-separated list. None = all samples."""
    if not value:
        return None
    if Path(value).is_file():
        return {line.strip() for line in Path(value).read_text(encoding="utf-8").splitlines() if line.strip()}
    return {sample_id.strip() for sample_id in value.split(",") if sample_id.strip()}


def select_samples(samples: list[dict], ids: str | None, limit: int | None) -> list[dict]:
    """Keep only the samples named by --ids (if given), then the first --limit of them (if given)."""
    wanted = parse_ids(ids)
    if wanted is not None:
        samples = [sample for sample in samples if sample["id"] in wanted]
    return samples[:limit]


# ============================================================================================ Tools catalog
def load_catalog(root: Path, split: str) -> dict:
    """The split's tools catalog (<split>_tools_catalog.json): every app and the JSON Schema of each tool."""
    with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return json.load(f)


def catalog_by_name(catalog: dict) -> dict[str, dict]:
    """Flatten the catalog into {tool name: tool spec}, e.g. {"smart_home.light_set": {...}}."""
    return {t["name"]: t for a in catalog["apps"] for t in a.get("tools", [])}


# ============================================================================================ zaic_tools
def import_zaic_tools(root: Path, split: str):
    """Import <root>/<split>/zaic_tools. Each split ships its own version, so use ONE split per process."""
    split_dir = str((root / split).resolve())
    loaded = sys.modules.get("zaic_tools")
    if loaded is not None:
        # Already imported: only OK if it is this split's copy
        if not str(Path(loaded.__file__).resolve()).startswith(split_dir):
            raise RuntimeError(f"zaic_tools already imported from {loaded.__file__}; run one split per process")
        return loaded
    sys.path.insert(0, split_dir)
    import zaic_tools  # noqa: E402
    return zaic_tools
