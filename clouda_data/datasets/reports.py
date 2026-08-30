from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import list_sources, verify_license


def generate_download_report(
    project_root: Path,
    output: str | Path = "outputs/reports/dataset_download_report.json",
) -> Path:
    manifests = []
    for path in (project_root / "data/manifests/download_manifests").glob("*.json"):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {"downloads": manifests}
    out = project_root / output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def license_matrix_rows(
    registry_path: str | Path = "data/manifests/dataset_registry.json",
) -> list[dict[str, Any]]:
    rows = []
    for source in list_sources(registry_path):
        result = verify_license(source)
        rows.append(
            {
                "source_id": source["source_id"],
                "name": source["name"],
                "classification": source["classification"],
                "license": source["license"],
                "commercial_use_allowed": result["commercial_use_allowed"],
                "redistribution_status": source["redistribution_status"],
                "risk_level": source["risk_level"],
            }
        )
    return rows
