import json
import logging
import os
from dataclasses import asdict, fields
from pathlib import Path

from .models import PageResult

logger = logging.getLogger(__name__)

_PAGE_RESULT_FIELDS = frozenset(field.name for field in fields(PageResult))


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
    """Load completed-page results, tolerating schema drift.

    Rows carrying unknown fields (e.g. written by another build) are
    recovered with the unknown fields dropped; only rows that cannot fit
    the dataclass at all are skipped, so a resume never silently loses
    every completed page.
    """

    path = checkpoint_path(job_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
    except (OSError, ValueError):
        return {}
    recovered: dict[int, PageResult] = {}
    dropped = 0
    for row in rows:
        if not isinstance(row, dict) or not row.get("page_no"):
            continue
        compatible = {
            key: value for key, value in row.items() if key in _PAGE_RESULT_FIELDS
        }
        try:
            page_no = int(row["page_no"])
            recovered[page_no] = PageResult(**compatible)
        except (TypeError, ValueError):
            dropped += 1
    if dropped:
        logger.warning(
            "Checkpoint %s: %d row(s) ignored due to incompatible fields",
            path,
            dropped,
        )
    return recovered


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
