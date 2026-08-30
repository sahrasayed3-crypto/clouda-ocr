from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import PageRecord
from .validators import validate_page_record


def write_page_manifest(records: list[PageRecord], path: str | Path) -> None:
    for record in records:
        validate_page_record(record)
    Path(path).write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_page_manifest(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input manifest must be a list of page records.")
    return data
