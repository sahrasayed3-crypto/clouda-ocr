import logging
import shutil
import sys
import os
import threading

from .settings import runtime_settings
from .operations import RedisSecurityConfig


def worker_health() -> dict:
    config = runtime_settings()
    config.temp_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(config.temp_root).free / (1024**3)
    from .local_engines import available_engine_status

    engines = available_engine_status()
    return {
        "worker_name": config.worker_name or "auto",
        "temporary_free_gb": round(free_gb, 2),
        "active_engines": [row["engine"] for row in engines if row["active"]],
    }


def main() -> int:
    config = runtime_settings()
    if config.app_role != "worker":
        print("Set APP_ROLE=worker before starting the worker.", file=sys.stderr)
        return 2
    if config.worker_concurrency != 1:
        print(
            "This release intentionally supports WORKER_CONCURRENCY=1 only.",
            file=sys.stderr,
        )
        return 2
    if not config.worker_api_key:
        print("WORKER_API_KEY is required.", file=sys.stderr)
        return 2

    from redis import Redis
    from rq import Queue, SimpleWorker, Worker
    from .worker_client import WorkerApiClient

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    security = RedisSecurityConfig.from_env()
    connection = Redis.from_url(security.url, **security.client_kwargs())
    connection.ping()
    client = WorkerApiClient()
    client.health()
    cloud_available = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    try:
        client.report_status(
            config.worker_name or "auto",
            cloud_available,
            "openrouter" if cloud_available else "",
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "server does not accept worker capability status yet"
        )

    stop_status = threading.Event()

    def status_loop() -> None:
        while not stop_status.wait(30):
            try:
                client.report_status(
                    config.worker_name or "auto",
                    cloud_available,
                    "openrouter" if cloud_available else "",
                )
            except Exception:
                logging.getLogger(__name__).warning(
                    "worker capability heartbeat failed"
                )

    threading.Thread(target=status_loop, daemon=True).start()
    logging.getLogger(__name__).info("worker health %s", worker_health())
    queue = Queue(
        f"{config.redis_namespace}:{config.rq_queue_name}", connection=connection
    )
    worker_class = SimpleWorker if os.name == "nt" else Worker
    worker = worker_class(
        [queue], connection=connection, name=config.worker_name or None
    )
    try:
        worker.work(with_scheduler=True)
    finally:
        stop_status.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
