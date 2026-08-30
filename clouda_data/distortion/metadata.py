from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DistortionMetadata:
    name: str
    version: str
    probability: float
    severity: str
    parameters: dict[str, Any]
    random_seed: int
    input_requirements: list[str] = field(default_factory=list)
    output_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
