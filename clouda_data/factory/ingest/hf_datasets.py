"""Hugging Face dataset-source adapter (from System A, network-optional).

Vendored pieces of ocrbench.ingest: the Dataset Viewer request helpers with
their polite rate-limit spacing, the selector reader, and row/download
fetching. Rendering is NOT coupled here (the unified factory renders through
its own backends). Nothing runs at import time and nothing is downloaded
unless `fetch_row`/`download` are called explicitly.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

VIEWER = "https://datasets-server.huggingface.co"
_LAST_VIEWER_CALL = 0.0


class ViewerError(RuntimeError):
    """Raised when the HF Dataset Viewer request ultimately fails."""


@dataclass(frozen=True)
class Selector:
    dataset: str
    split: str
    row: int


def read_selectors(path: Path) -> list[Selector]:
    """Read configs/source_selectors.csv (dataset,split,row)."""
    selectors: list[Selector] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            selectors.append(
                Selector(
                    dataset=str(record["dataset"]).strip(),
                    split=str(record["split"]).strip(),
                    row=int(record["row"]),
                )
            )
    return selectors


def _get_json(endpoint: str, **params: Any) -> dict[str, Any]:
    """Vendored rate-limited Viewer GET (spacing preserved from System A)."""
    global _LAST_VIEWER_CALL
    url = f"{VIEWER}/{endpoint}?{urlencode(params)}"
    for attempt in range(3):
        spacing = 2.2 - (time.monotonic() - _LAST_VIEWER_CALL)
        if spacing > 0:
            time.sleep(spacing)
        _LAST_VIEWER_CALL = time.monotonic()
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise ViewerError(f"Dataset Viewer request failed: {url}")
            time.sleep(2.0)
    raise ViewerError(f"unreachable: {url}")  # pragma: no cover


def _row(dataset: str, split: str, index: int) -> dict[str, Any] | None:
    raw = _get_json(
        "/rows", dataset=dataset, config="default", split=split, offset=index, length=1
    )
    import json

    payload = json.loads(raw)  # type: ignore
    rows = payload.get("rows") or []
    return rows[0].get("row") if rows else None


def download(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()
