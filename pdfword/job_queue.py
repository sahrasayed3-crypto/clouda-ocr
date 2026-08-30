import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .settings import runtime_settings
from .operations import RedisSecurityConfig


class JobCancelled(RuntimeError):
    pass


@dataclass
class ManagedJob:
    username: str
    cancel_event: threading.Event
    future: Future


class JobQueue:
    def __init__(self, max_workers: int = 2) -> None:
        self.max_workers = max(1, int(max_workers))
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="clouda-job"
        )
        self._jobs: dict[str, ManagedJob] = {}
        self._lock = threading.RLock()

    def submit(
        self, job_id: str, username: str, worker: Callable[[Callable[[], bool]], None]
    ) -> Future:
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing and not existing.future.done():
                return existing.future
            cancel_event = threading.Event()
            future = self._executor.submit(worker, cancel_event.is_set)
            self._jobs[job_id] = ManagedJob(
                username=username, cancel_event=cancel_event, future=future
            )
            return future

    def cancel(self, job_id: str, username: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.username != username:
                return False
            job.cancel_event.set()
            job.future.cancel()
            return True

    def status(self, job_id: str, username: str) -> str | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.username != username:
                return None
            if job.cancel_event.is_set():
                return "cancelled"
            if job.future.cancelled():
                return "cancelled"
            if job.future.done():
                return "failed" if job.future.exception() else "completed"
            return "running" if job.future.running() else "queued"

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if not job.future.done())

    def active_job_ids(self) -> set[str]:
        with self._lock:
            return {
                job_id for job_id, job in self._jobs.items() if not job.future.done()
            }


_QUEUE = JobQueue(max_workers=2)


def get_job_queue() -> JobQueue:
    return _QUEUE


class DistributedJobQueue:
    """Small RQ adapter. Imports Redis/RQ lazily so server imports stay light."""

    def __init__(self) -> None:
        config = runtime_settings()
        self.redis_url = config.redis_url
        self.queue_name = f"{config.redis_namespace}:{config.rq_queue_name}"
        self.job_timeout = config.job_timeout_seconds
        self.retry_count = config.job_retry_count

    def _queue(self):
        from redis import Redis
        from rq import Queue

        security = RedisSecurityConfig.from_env()
        connection = Redis.from_url(security.url, **security.client_kwargs())
        return Queue(self.queue_name, connection=connection)

    def enqueue(self, job_id: str):
        from rq import Retry

        queue = self._queue()
        existing = queue.fetch_job(job_id)
        if existing and existing.get_status(refresh=True) in {
            "queued",
            "started",
            "deferred",
            "scheduled",
        }:
            return existing
        if existing:
            existing.delete()
        retry = (
            Retry(max=self.retry_count, interval=[30, 120][: self.retry_count])
            if self.retry_count
            else None
        )
        return queue.enqueue(
            "pdfword.worker_tasks.run_remote_job",
            job_id,
            job_id=job_id,
            job_timeout=self.job_timeout,
            retry=retry,
            result_ttl=86400,
            failure_ttl=604800,
        )

    def cancel(self, job_id: str) -> bool:
        queue = self._queue()
        job = queue.fetch_job(job_id)
        if job is None:
            return False
        from rq.command import send_stop_job_command

        status = job.get_status(refresh=True)
        if status == "started":
            send_stop_job_command(queue.connection, job.id)
        else:
            job.cancel()
        return True

    def pending_count(self) -> int:
        return len(self._queue())

    def ping(self) -> bool:
        return bool(self._queue().connection.ping())

    def metrics(self) -> dict[str, int | float | bool]:
        queue = self._queue()
        connection = queue.connection
        oldest_age_seconds = 0.0
        job_ids = queue.get_job_ids(offset=0, length=1)
        if job_ids:
            job = queue.fetch_job(job_ids[0])
            if job and job.enqueued_at:
                from datetime import datetime, timezone

                enqueued = job.enqueued_at
                if enqueued.tzinfo is None:
                    enqueued = enqueued.replace(tzinfo=timezone.utc)
                oldest_age_seconds = max(
                    0.0, (datetime.now(timezone.utc) - enqueued).total_seconds()
                )
        return {
            "redis_available": bool(connection.ping()),
            "queue_depth": len(queue),
            "oldest_job_age_seconds": oldest_age_seconds,
            "failed_jobs": int(connection.zcard(f"rq:{self.queue_name}:failed")),
        }


_DISTRIBUTED_QUEUE: DistributedJobQueue | None = None


def get_distributed_queue() -> DistributedJobQueue:
    global _DISTRIBUTED_QUEUE
    if _DISTRIBUTED_QUEUE is None:
        _DISTRIBUTED_QUEUE = DistributedJobQueue()
    return _DISTRIBUTED_QUEUE


def cancel_conversion_job(
    job_id: str,
    username: str,
    *,
    local_processing_enabled: bool,
) -> bool:
    if local_processing_enabled:
        return get_job_queue().cancel(job_id, username)
    return get_distributed_queue().cancel(job_id)
