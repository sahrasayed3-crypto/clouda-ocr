import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clouda_contracts.storage import StorageRoots

from .database import Database
from .engines import get_engine_registry


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    app_role: str
    redis_url: str
    redis_namespace: str
    rq_queue_name: str
    server_base_url: str
    worker_api_key: str
    worker_name: str
    local_processing_enabled: bool
    local_ocr_enabled: bool
    storage_root: Path
    temp_root: Path
    database_path: Path
    worker_concurrency: int
    job_timeout_seconds: int
    job_retry_count: int
    correction_memory_enabled: bool
    correction_min_source_files: int
    correction_auto_apply_threshold: float
    correction_context_window: int
    correction_max_rules_per_page: int


def runtime_settings() -> RuntimeSettings:
    roots = StorageRoots.from_env()
    storage_root = Path(
        os.getenv("STORAGE_ROOT", str(roots.runtime_root / "conversions"))
    )
    return RuntimeSettings(
        app_role=os.getenv("APP_ROLE", "server").strip().lower(),
        redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip(),
        redis_namespace=os.getenv("CLOUDA_REDIS_NAMESPACE", "clouda").strip()
        or "clouda",
        rq_queue_name=os.getenv("RQ_QUEUE_NAME", "pdf_conversion").strip(),
        server_base_url=os.getenv("SERVER_BASE_URL", "http://127.0.0.1:8000").rstrip(
            "/"
        ),
        worker_api_key=os.getenv("WORKER_API_KEY", ""),
        worker_name=os.getenv("WORKER_NAME", "").strip(),
        local_processing_enabled=_env_bool("LOCAL_PROCESSING_ENABLED", False),
        local_ocr_enabled=_env_bool("CLOUDA_LOCAL_OCR_ENABLED", False),
        storage_root=storage_root,
        temp_root=Path(os.getenv("TEMP_ROOT", str(storage_root / "temporary"))),
        database_path=Path(
            os.getenv(
                "CLOUDA_DATABASE_PATH",
                os.getenv("DATABASE_PATH", str(roots.database_path)),
            )
        ),
        worker_concurrency=max(1, int(os.getenv("WORKER_CONCURRENCY", "1"))),
        job_timeout_seconds=max(60, int(os.getenv("JOB_TIMEOUT_SECONDS", "7200"))),
        job_retry_count=max(0, int(os.getenv("JOB_RETRY_COUNT", "2"))),
        correction_memory_enabled=_env_bool("CORRECTION_MEMORY_ENABLED", True),
        correction_min_source_files=max(
            1, int(os.getenv("CORRECTION_MIN_SOURCE_FILES", "3"))
        ),
        correction_auto_apply_threshold=max(
            0.0, min(1.0, float(os.getenv("CORRECTION_AUTO_APPLY_THRESHOLD", "0.90")))
        ),
        correction_context_window=max(
            5, int(os.getenv("CORRECTION_CONTEXT_WINDOW", "30"))
        ),
        correction_max_rules_per_page=max(
            1, int(os.getenv("CORRECTION_MAX_RULES_PER_PAGE", "50"))
        ),
    )


DEFAULT_SETTINGS: dict[str, Any] = {
    "acceptance_threshold": 90.0,
    "escalation_threshold": 90.0,
    "minimum_improvement": 1.0,
    "local_attempts": 1,
    "free_model_attempts": 0,
    "paid_model_attempts": 0,
    "local_timeout_seconds": 120,
    "cloud_timeout_seconds": 0,
    "file_cost_limit": 0.0,
    "daily_cost_limit": 0.0,
    "smart_routing_min_samples": 20,
    "max_concurrent_jobs": 2,
    "batch_size": 10,
    "default_language": "embedded-pdf-text",
    "default_font": "Arial",
    "scan_dpi": 0,
    "max_pdf_pages": 500,
    "max_dpi": 0,
    "max_parallel_pages": 1,
    "max_ocr_attempts": 1,
    "page_timeout_seconds": 300,
    "file_timeout_seconds": 7200,
    "enabled_engines": [
        "direct_pdf_text",
        "local_model_ocr",
        "future_ocr_engine",
    ],
    "enabled_models": [],
    "storage_root": "conversions",
    "temporary_retention_hours": 24,
    "backup_retention_days": 14,
}


def _decode(value: str, value_type: str):
    if value_type == "bool":
        return value.lower() in {"1", "true", "yes"}
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "json":
        return json.loads(value)
    return value


def _encode(value) -> tuple[str, str]:
    if isinstance(value, bool):
        return ("true" if value else "false"), "bool"
    if isinstance(value, int):
        return str(value), "int"
    if isinstance(value, float):
        return str(value), "float"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False), "json"
    return str(value), "str"


def load_settings(database: Database) -> dict:
    values = dict(DEFAULT_SETTINGS)
    for key, row in database.get_setting_rows().items():
        try:
            values[key] = _decode(row["value"], row["type"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return values


def validate_setting(key: str, value):
    if (
        key in {"acceptance_threshold", "escalation_threshold"}
        and not 0 <= float(value) <= 100
    ):
        raise ValueError("Quality thresholds must be between 0 and 100.")
    if (
        key in {"local_attempts", "free_model_attempts", "paid_model_attempts"}
        and not 0 <= int(value) <= 20
    ):
        raise ValueError("Attempt counts must be between 0 and 20.")
    if key == "max_concurrent_jobs" and int(value) != 2:
        raise ValueError("This version supports two concurrent jobs.")
    if key in {"file_cost_limit", "daily_cost_limit"} and float(value) < 0:
        raise ValueError("Cost limits cannot be negative.")
    if key == "smart_routing_min_samples" and not 20 <= int(value) <= 30:
        raise ValueError("Smart routing samples must be between 20 and 30.")
    if key == "batch_size" and not 1 <= int(value) <= 100:
        raise ValueError("Batch size must be between 1 and 100 pages.")
    if key == "enabled_engines":
        allowed = set(get_engine_registry().names())
        unknown = set(value or []) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported extraction engines: {', '.join(sorted(unknown))}"
            )
    return value


def save_settings(database: Database, values: dict) -> None:
    for key, value in values.items():
        validate_setting(key, value)
        encoded, value_type = _encode(value)
        database.set_setting(key, encoded, value_type)
