import json
import os
from dataclasses import asdict
from pathlib import Path

from .models import PageResult


def checkpoint_path(job_root: str | Path) -> Path:
    return Path(job_root) / "checkpoint.json"


def save_checkpoint(job_root: str | Path, results: dict[int, PageResult]) -> None:
    path = checkpoint_path(job_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    payload = {
        "completed_pages": sorted(results),
        "results": [asdict(results[page]) for page in sorted(results)],
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_checkpoint(job_root: str | Path) -> dict[int, PageResult]:
    path = checkpoint_path(job_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            int(row["page_no"]): PageResult(**row)
            for row in payload.get("results", [])
            if isinstance(row, dict) and row.get("page_no")
        }
    except (OSError, ValueError, TypeError):
        return {}


def prepare_failed_page_retry(
    job_root: str | Path, threshold: float = 90.0
) -> list[int]:
    results = load_checkpoint(job_root)
    failed = [
        page_no
        for page_no, result in results.items()
        if result.text_quality_score is None
        or float(result.text_quality_score) < threshold
        or result.model_used.startswith("failed:")
    ]
    successful = {
        page_no: result for page_no, result in results.items() if page_no not in failed
    }
    save_checkpoint(job_root, successful)
    return sorted(failed)
