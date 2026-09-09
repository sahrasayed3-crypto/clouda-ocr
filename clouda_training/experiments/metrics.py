from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class MetricRecord:
    timestamp: str
    step: int
    epoch: float
    split: str
    metric_name: str
    value: float
    run_id: str


class MetricLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self, *, step: int, epoch: float, split: str, metric_name: str, value: float
    ) -> MetricRecord:
        record = MetricRecord(
            utc_now(), step, epoch, split, metric_name, float(value), self.run_id
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def read_metrics(path: Path) -> Iterator[dict]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupt metrics JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Metric at line {line_number} must be an object")
            yield row
