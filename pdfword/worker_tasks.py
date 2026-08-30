import logging
import os
import socket
import tempfile
import threading
from pathlib import Path

from .key_router.audit import sanitize_metadata
from .settings import runtime_settings
from .worker_client import WorkerApiClient

logger = logging.getLogger(__name__)


def _worker_name() -> str:
    return runtime_settings().worker_name or socket.gethostname()


def run_remote_job(job_id: str) -> dict:
    """RQ entrypoint. Its only serialized argument is the opaque job ID."""
    config = runtime_settings()
    if config.app_role != "worker":
        raise RuntimeError("RQ processing requires APP_ROLE=worker")
    client = WorkerApiClient()
    worker_name = _worker_name()
    stop_heartbeat = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(20):
            try:
                client.heartbeat(job_id, worker_name)
            except Exception:
                logger.warning(
                    "heartbeat failed job_id=%s worker=%s", job_id, worker_name
                )

    try:
        job = client.start(job_id, worker_name)
        snapshot_loader = getattr(client, "correction_snapshot", None)
        correction_snapshot = snapshot_loader() if snapshot_loader else {"rules": []}
        config.temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=config.temp_root) as tmp:
            root = Path(tmp)
            input_path = root / "input.pdf"
            output_path = root / "output.docx"
            client.download_input(job_id, input_path)
            heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
            heartbeat.start()

            # Worker-only import: this is where OCR/PDF dependencies are loaded.
            from .conversion_service import (
                WorkerConversionRequest,
                execute_worker_conversion,
            )

            result = execute_worker_conversion(
                WorkerConversionRequest(
                    job_id=job_id,
                    pdf_path=input_path,
                    docx_path=output_path,
                    page_numbers=list(job["page_numbers"]),
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                    fast_model=job["fast_model"],
                    accurate_model=job["accurate_model"],
                    settings=job["settings"],
                    correction_rules=list(correction_snapshot.get("rules") or []),
                )
            )
            with output_path.open("rb") as docx:
                response = client.upload_result(job_id, worker_name, docx, result)
            logger.info(
                "job completed job_id=%s worker=%s status=%s",
                job_id,
                worker_name,
                response["status"],
            )
            return response
    except Exception as exc:
        logger.error(
            "job failed job_id=%s worker=%s error_type=%s",
            job_id,
            worker_name,
            type(exc).__name__,
        )
        try:
            client.fail(job_id, worker_name, str(sanitize_metadata(str(exc))))
        except Exception as failure_exc:
            logger.error(
                "failure report failed job_id=%s worker=%s error_type=%s",
                job_id,
                worker_name,
                type(failure_exc).__name__,
            )
        raise
    finally:
        stop_heartbeat.set()
