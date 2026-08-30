from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .status import PageStatus


@dataclass
class Checkpoint:
    job_id: str
    statuses: dict[str, PageStatus] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def mark(self, item_id: str, status: PageStatus) -> None:
        self.statuses[item_id] = status
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def increment_attempt(self, item_id: str) -> int:
        self.attempts[item_id] = self.attempts.get(item_id, 0) + 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self.attempts[item_id]


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> None:
    payload = asdict(checkpoint)
    payload["statuses"] = {
        key: status.value for key, status in checkpoint.statuses.items()
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_checkpoint(path: str | Path) -> Checkpoint:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Checkpoint(
        job_id=payload["job_id"],
        statuses={
            key: PageStatus(value) for key, value in payload.get("statuses", {}).items()
        },
        attempts=dict(payload.get("attempts", {})),
        updated_at=payload.get("updated_at", datetime.now(timezone.utc).isoformat()),
    )
