import io
import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .checkpoints import load_checkpoint, save_checkpoint
from .database import Database, utc_now
from .docx_export import markdown_to_docx
from .job_queue import JobCancelled
from .limits import limits_from_settings, validate_pdf_limits
from .ocr_pipeline import process_pdf
from .openrouter_client import capture_openrouter_telemetry
from .accuracy import final_acceptance_decision
from .settings import load_settings
from .correction_learning import apply_correction_rules
from .models import PageResult
from pypdf import PdfReader


def _finite_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


class SilentProgress:
    def progress(self, *_args, **_kwargs) -> None:
        return None


class SilentStatus:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None

    def success(self, *_args, **_kwargs) -> None:
        return None


def _page_final_score(page: PageResult) -> float | None:
    return (
        page.text_quality_score
        if page.text_quality_score is not None
        else page.quality_score
    )


def _normalize_manual_review_flags(
    page_results: list[PageResult], threshold: float = 90.0
) -> None:
    for page in page_results:
        score = _page_final_score(page)
        decision = final_acceptance_decision(
            page.markdown,
            estimated_text_quality=score,
            threshold=threshold,
            expected_non_empty=True,
        )
        page.corruption_diagnostics = decision["diagnostics"]
        if score is None:
            page.text_quality_score = decision["estimated_text_quality"]
        needs_review = score is None or bool(decision["requires_manual_review"])
        page.requires_manual_review = bool(page.requires_manual_review or needs_review)
        if page.requires_manual_review and not page.review_reason:
            page.review_reason = decision["review_reason"]
        page.accepted = not page.requires_manual_review
        if not page.route_used:
            page.route_used = page.model_used


class LiveConversionProgress:
    def __init__(self, job_root: str, total_pages: int) -> None:
        self.path = Path(job_root) / "progress.json"
        self.total_pages = total_pages
        self.current_page: int | None = None

    def _write(self, *, stage: str, completed_pages: int | None = None) -> None:
        payload = {
            "stage": stage,
            "current_page": self.current_page,
            "completed_pages": completed_pages,
            "total_pages": self.total_pages,
            "updated_at": utc_now(),
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def progress(self, value: float, text: str = "") -> None:
        completed = round(max(0.0, min(1.0, float(value))) * self.total_pages)
        self._write(stage=text or "معالجة الصفحات", completed_pages=completed)

    def info(self, message: str) -> None:
        match = re.search(r"(?:الصفحة|page)\s+(\d+)", str(message), re.IGNORECASE)
        if match:
            self.current_page = int(match.group(1))
        self._write(stage=str(message))

    def warning(self, message: str) -> None:
        self._write(stage=str(message))

    def success(self, message: str) -> None:
        self._write(stage=str(message), completed_pages=self.total_pages)


@dataclass(frozen=True)
class ConversionRequest:
    conversion_id: int
    job_id: str
    username: str
    pdf_path: Path
    job_root: str
    docx_path: str
    page_numbers: list[int]
    api_key: str
    fast_model: str
    accurate_model: str
    max_parallel_pages: int = 2


@dataclass(frozen=True)
class WorkerConversionRequest:
    job_id: str
    pdf_path: Path
    docx_path: Path
    page_numbers: list[int]
    api_key: str
    fast_model: str
    accurate_model: str
    settings: dict
    correction_rules: list[dict] | None = None


def execute_worker_conversion(request: WorkerConversionRequest) -> dict:
    """Run conversion without opening the server's SQLite database."""
    started_at = time.perf_counter()
    pdf_bytes = request.pdf_path.read_bytes()
    limits = limits_from_settings(request.settings)
    validate_pdf_limits(
        len(pdf_bytes),
        len(PdfReader(io.BytesIO(pdf_bytes)).pages),
        limits=limits,
    )
    cloud_attempts: list[dict] = []
    telemetry_events: list[dict] = []
    budget_lock = threading.Lock()
    accumulated_cost = 0.0
    reserved_cost = 0.0
    reservations: dict[int, list[float]] = {}
    document_cost_start = _finite_float(request.settings.get("current_document_cost"))
    daily_cost_start = _finite_float(request.settings.get("daily_cost_spent"))
    file_cost_limit = _finite_float(request.settings.get("file_cost_limit"), -1.0)
    daily_cost_limit = _finite_float(request.settings.get("daily_cost_limit"), -1.0)

    def record_telemetry(event: dict) -> None:
        nonlocal accumulated_cost, reserved_cost
        with budget_lock:
            thread_reservations = reservations.get(threading.get_ident(), [])
            if thread_reservations:
                reserved_cost = max(0.0, reserved_cost - thread_reservations.pop(0))
            accumulated_cost += _finite_float(event.get("cost"))
            row = dict(event)
            row["_thread_id"] = threading.get_ident()
            telemetry_events.append(row)

    def cloud_attempt_allowed(estimated_cost: float | None = 0.0) -> bool:
        nonlocal reserved_cost
        if estimated_cost is None:
            return False
        with budget_lock:
            if file_cost_limit < 0 or daily_cost_limit < 0:
                return False
            estimate = max(0.0, _finite_float(estimated_cost, -1.0))
            if estimate < 0:
                return False
            projected_cost = accumulated_cost + reserved_cost + estimate
            over_file = file_cost_limit > 0 and (
                document_cost_start + accumulated_cost + reserved_cost
                >= file_cost_limit
                or document_cost_start + projected_cost > file_cost_limit
            )
            over_daily = daily_cost_limit > 0 and (
                daily_cost_start + accumulated_cost + reserved_cost >= daily_cost_limit
                or daily_cost_start + projected_cost > daily_cost_limit
            )
            allowed = not (over_file or over_daily)
            if allowed and estimate > 0:
                reserved_cost += estimate
                reservations.setdefault(threading.get_ident(), []).append(estimate)
            return allowed

    def record_cloud_attempt(event: dict) -> None:
        thread_id = threading.get_ident()
        with budget_lock:
            matching = [
                row
                for row in telemetry_events
                if row.get("_thread_id") == thread_id
                and row.get("model") == event.get("model")
            ]
            if not matching:
                matching = [
                    row
                    for row in telemetry_events
                    if row.get("_thread_id") == thread_id
                ]
            for row in matching:
                telemetry_events.remove(row)
        event = dict(event)
        event.update(
            {
                "prompt_tokens": sum(
                    int(row.get("prompt_tokens") or 0) for row in matching
                ),
                "completion_tokens": sum(
                    int(row.get("completion_tokens") or 0) for row in matching
                ),
                "cost": sum(float(row.get("cost") or 0) for row in matching),
                "cost_is_estimated": int(
                    not matching
                    or any(row.get("cost_is_estimated") for row in matching)
                ),
                "cost_unknown": int(
                    not matching or any(row.get("cost_unknown") for row in matching)
                ),
            }
        )
        cloud_attempts.append(event)

    with capture_openrouter_telemetry(record_telemetry):
        page_results, _ = process_pdf(
            pdf_bytes=pdf_bytes,
            from_page=None,
            to_page=None,
            api_key=request.api_key,
            fast_model=request.fast_model,
            accurate_model=request.accurate_model,
            progress_bar=SilentProgress(),
            status_placeholder=SilentStatus(),
            speed_mode="turbo",
            page_numbers=request.page_numbers,
            max_parallel_pages=limits.max_parallel_pages,
            acceptance_threshold=float(
                request.settings.get("acceptance_threshold", 90.0)
            ),
            max_cloud_attempts=min(
                limits.max_ocr_attempts,
                int(request.settings.get("free_model_attempts", 10))
                + int(request.settings.get("paid_model_attempts", 5)),
            ),
            scan_dpi=min(limits.max_dpi, int(request.settings.get("scan_dpi", 300))),
            max_dpi=limits.max_dpi,
            enabled_engines=list(request.settings.get("enabled_engines") or []),
            enabled_models=list(request.settings.get("enabled_models") or []),
            batch_size=int(request.settings.get("batch_size", 10)),
            cloud_attempt_allowed=cloud_attempt_allowed,
            cloud_attempt_callback=record_cloud_attempt,
        )
    _normalize_manual_review_flags(
        page_results,
        threshold=float(request.settings.get("acceptance_threshold", 90.0)),
    )
    correction_applications = []
    for page in page_results:
        applied = apply_correction_rules(
            page.markdown,
            request.correction_rules or [],
            job_id=request.job_id,
            document_type="",
        )
        page.markdown = applied.text
        correction_applications.extend(applied.applications)
    request.docx_path.write_bytes(markdown_to_docx(page_results))
    engines = [item.model_used for item in page_results]
    winner = Counter(engines).most_common(1)[0][0] if engines else ""
    file_types = {
        (
            "scan"
            if "ocr" in engine.lower()
            else "digital" if "pypdf" in engine.lower() else "mixed"
        )
        for engine in engines
    }
    text_scores = [
        item.text_quality_score
        for item in page_results
        if item.text_quality_score is not None
    ]
    layout_scores = [
        item.layout_quality_score
        for item in page_results
        if item.layout_quality_score is not None
    ]
    final_scores = [
        item.quality_score for item in page_results if item.quality_score is not None
    ]
    manual_review = any(item.requires_manual_review for item in page_results)
    cost_limit_reached = any(
        "cost_limit_reached" in (item.review_reason or "")
        or "حد التكلفة" in (item.review_reason or "")
        for item in page_results
    )
    return {
        "status": "manual_review" if manual_review else "completed",
        "output_type": "text_only_docx",
        "output_policy": "tables_and_images_are_not_rebuilt",
        "file_type": next(iter(file_types)) if len(file_types) == 1 else "mixed",
        "text_quality_score": (
            sum(text_scores) / len(text_scores) if text_scores else None
        ),
        "layout_quality_score": (
            sum(layout_scores) / len(layout_scores) if layout_scores else None
        ),
        "final_quality_score": (
            sum(final_scores) / len(final_scores) if final_scores else None
        ),
        "winning_engine": winner,
        "processing_time": time.perf_counter() - started_at,
        "manual_review_pages": [
            item.page_no for item in page_results if item.requires_manual_review
        ],
        "correction_applications": correction_applications,
        "cloud_attempts": cloud_attempts,
        "cost_limit_reached": cost_limit_reached,
    }


def execute_conversion(
    request: ConversionRequest, cancellation_check: Callable[[], bool]
) -> None:
    database = Database()
    started_at = time.perf_counter()
    database.update_conversion(
        request.conversion_id,
        {"status": "processing", "updated_at": utc_now(), "error_message": None},
    )
    existing = load_checkpoint(request.job_root)
    settings = load_settings(database)
    limits = limits_from_settings(settings)
    pdf_bytes = request.pdf_path.read_bytes()
    validate_pdf_limits(
        len(pdf_bytes),
        len(PdfReader(io.BytesIO(pdf_bytes)).pages),
        limits=limits,
    )
    attempt_number = len(database.list_attempts(request.conversion_id))
    conversion = database.get_conversion(request.job_id)
    accumulated_cost = float((conversion or {}).get("total_cost") or 0)
    budget_lock = threading.Lock()
    reserved_cost = 0.0
    reservations: dict[int, list[float]] = {}

    def record_telemetry(event: dict) -> None:
        nonlocal attempt_number, accumulated_cost, reserved_cost
        with budget_lock:
            thread_reservations = reservations.get(threading.get_ident(), [])
            if thread_reservations:
                reserved_cost = max(0.0, reserved_cost - thread_reservations.pop(0))
            attempt_number += 1
            current_attempt = attempt_number
            accumulated_cost += float(event.get("cost") or 0)
        database.record_attempt(
            {
                "conversion_id": request.conversion_id,
                "engine_name": "openrouter",
                "model_name": event.get("model") or "",
                "engine_type": "cloud",
                "attempt_number": current_attempt,
                "cost": float(event.get("cost") or 0),
                "cost_is_estimated": int(bool(event.get("cost_is_estimated"))),
                "prompt_tokens": int(event.get("prompt_tokens") or 0),
                "completion_tokens": int(event.get("completion_tokens") or 0),
                "processing_time": float(event.get("processing_time") or 0),
                "success": 1,
                "created_at": utc_now(),
            }
        )

    def should_cancel() -> bool:
        return cancellation_check()

    def cloud_attempt_allowed(estimated_cost: float | None = 0.0) -> bool:
        nonlocal reserved_cost
        if estimated_cost is None:
            return False
        with budget_lock:
            file_limit = float(settings.get("file_cost_limit") or 0)
            daily_limit = float(settings.get("daily_cost_limit") or 0)
            estimate = max(0.0, float(estimated_cost))
            over_file = file_limit > 0 and (
                accumulated_cost + reserved_cost >= file_limit
                or accumulated_cost + reserved_cost + estimate > file_limit
            )
            daily_cost = database.daily_cost()
            over_daily = daily_limit > 0 and (
                daily_cost + reserved_cost >= daily_limit
                or daily_cost + reserved_cost + estimate > daily_limit
            )
            allowed = not (over_file or over_daily)
            if allowed and estimate > 0:
                reserved_cost += estimate
                reservations.setdefault(threading.get_ident(), []).append(estimate)
            return allowed

    live_progress = LiveConversionProgress(request.job_root, len(request.page_numbers))
    try:
        with capture_openrouter_telemetry(record_telemetry):
            page_results, _ = process_pdf(
                pdf_bytes=pdf_bytes,
                from_page=None,
                to_page=None,
                api_key=request.api_key,
                fast_model=request.fast_model,
                accurate_model=request.accurate_model,
                progress_bar=live_progress,
                status_placeholder=live_progress,
                speed_mode="turbo",
                page_numbers=request.page_numbers,
                max_parallel_pages=min(
                    limits.max_parallel_pages, request.max_parallel_pages
                ),
                cancellation_check=should_cancel,
                checkpoint_callback=lambda results: save_checkpoint(
                    request.job_root, results
                ),
                existing_results=existing,
                acceptance_threshold=float(settings.get("acceptance_threshold", 90.0)),
                max_cloud_attempts=min(
                    limits.max_ocr_attempts,
                    int(settings.get("free_model_attempts", 10))
                    + int(settings.get("paid_model_attempts", 5)),
                ),
                scan_dpi=min(limits.max_dpi, int(settings.get("scan_dpi", 300))),
                max_dpi=limits.max_dpi,
                enabled_engines=list(settings.get("enabled_engines") or []),
                enabled_models=list(settings.get("enabled_models") or []),
                batch_size=int(settings.get("batch_size", 10)),
                cloud_attempt_allowed=cloud_attempt_allowed,
            )
        _normalize_manual_review_flags(
            page_results,
            threshold=float(settings.get("acceptance_threshold", 90.0)),
        )
        if should_cancel():
            raise JobCancelled("تم إلغاء المهمة بواسطة المستخدم")
        approved_rules = database.enabled_correction_rules()
        for page in page_results:
            for rule in approved_rules:
                page.markdown = page.markdown.replace(
                    rule["pattern"], rule["replacement"]
                )
        docx_bytes = markdown_to_docx(page_results)
        Path(request.docx_path).write_bytes(docx_bytes)

        engines = [item.model_used for item in page_results]
        winner = Counter(engines).most_common(1)[0][0] if engines else ""
        file_types = {
            (
                "scan"
                if "ocr" in engine.lower()
                else "digital" if "pypdf" in engine.lower() else "mixed"
            )
            for engine in engines
        }
        file_type = next(iter(file_types)) if len(file_types) == 1 else "mixed"
        text_scores = [
            item.text_quality_score
            for item in page_results
            if item.text_quality_score is not None
        ]
        layout_scores = [
            item.layout_quality_score
            for item in page_results
            if item.layout_quality_score is not None
        ]
        final_scores = [
            item.quality_score
            for item in page_results
            if item.quality_score is not None
        ]
        existing_attempts = len(database.list_attempts(request.conversion_id))
        for offset, item in enumerate(page_results, start=1):
            if item.model_used.startswith("local:"):
                database.record_attempt(
                    {
                        "conversion_id": request.conversion_id,
                        "engine_name": item.model_used.split(":", 1)[1],
                        "model_name": "",
                        "engine_type": "local",
                        "attempt_number": existing_attempts + offset,
                        "quality_score": item.quality_score,
                        "cost": 0.0,
                        "cost_is_estimated": 0,
                        "processing_time": 0.0,
                        "success": int(not item.requires_manual_review),
                        "failure_reason": (
                            "جودة أقل من حد القبول"
                            if item.requires_manual_review
                            else None
                        ),
                        "created_at": utc_now(),
                    }
                )
        database.update_conversion(
            request.conversion_id,
            {
                "file_type": file_type,
                "text_quality_score": (
                    sum(text_scores) / len(text_scores) if text_scores else None
                ),
                "layout_quality_score": (
                    sum(layout_scores) / len(layout_scores) if layout_scores else None
                ),
                "final_quality_score": (
                    sum(final_scores) / len(final_scores) if final_scores else None
                ),
                "winning_engine": winner,
                "processing_time": time.perf_counter() - started_at,
                "status": (
                    "manual_review"
                    if any(item.requires_manual_review for item in page_results)
                    else "completed"
                ),
                "updated_at": utc_now(),
            },
        )
    except JobCancelled as exc:
        database.update_conversion(
            request.conversion_id,
            {
                "status": "cancelled",
                "processing_time": time.perf_counter() - started_at,
                "updated_at": utc_now(),
                "error_message": str(exc),
            },
        )
    except Exception as exc:
        database.update_conversion(
            request.conversion_id,
            {
                "status": "failed",
                "processing_time": time.perf_counter() - started_at,
                "updated_at": utc_now(),
                "error_message": str(exc),
            },
        )
        raise
