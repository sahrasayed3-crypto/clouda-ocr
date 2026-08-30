from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

FINAL_PAGE_STATES = {
    "complete",
    "failed",
    "skipped",
    "quarantined",
    "cancelled",
    "manual_review",
}
PAGE_STATES = FINAL_PAGE_STATES | {"queued", "processing"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunCheckpointStore:
    """SQLite recovery index for a distortion run.

    JSONL remains the portable provenance record. This database provides
    transactional claims, heartbeats, retry accounting, and stale recovery.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    input_checksum TEXT NOT NULL,
                    profile_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    generated_page_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    source_page_id TEXT NOT NULL,
                    variant INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL,
                    heartbeat_at TEXT,
                    output_uri TEXT,
                    output_checksum TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(run_id, source_page_id, variant)
                );
                CREATE INDEX IF NOT EXISTS idx_pages_run_status
                ON pages(run_id, status);
                """)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def start_run(
        self,
        run_id: str,
        *,
        input_checksum: str,
        profile_hash: str,
        metadata: dict[str, Any],
        resume: bool,
    ) -> None:
        now = utc_now()
        with self.connect() as connection, connection:
            row = connection.execute(
                "SELECT input_checksum, profile_hash FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is not None:
                if (
                    row["input_checksum"] != input_checksum
                    or row["profile_hash"] != profile_hash
                ):
                    raise ValueError("Duplicate run ID has conflicting provenance")
                if not resume:
                    raise FileExistsError(f"Run {run_id} already exists")
                connection.execute(
                    "UPDATE runs SET state = 'processing', updated_at = ? WHERE run_id = ?",
                    (now, run_id),
                )
                return
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, input_checksum, profile_hash, state,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, 'processing', ?, ?, ?)
                """,
                (
                    run_id,
                    input_checksum,
                    profile_hash,
                    now,
                    now,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ),
            )

    def queue_page(
        self,
        *,
        run_id: str,
        generated_page_id: str,
        source_page_id: str,
        variant: int,
        max_retries: int,
    ) -> str:
        now = utc_now()
        with self.connect() as connection, connection:
            connection.execute(
                """
                INSERT INTO pages (
                    generated_page_id, run_id, source_page_id, variant,
                    status, max_retries, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                ON CONFLICT(generated_page_id) DO NOTHING
                """,
                (
                    generated_page_id,
                    run_id,
                    source_page_id,
                    variant,
                    max_retries,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT status FROM pages WHERE generated_page_id = ?",
                (generated_page_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Checkpoint page insert failed")
        return str(row["status"])

    def claim_page(self, generated_page_id: str) -> bool:
        now = utc_now()
        with self.connect() as connection, connection:
            cursor = connection.execute(
                """
                UPDATE pages
                SET status = 'processing', heartbeat_at = ?, updated_at = ?,
                    retry_count = retry_count + CASE WHEN error IS NULL THEN 0 ELSE 1 END,
                    error = NULL
                WHERE generated_page_id = ?
                  AND (
                    status = 'queued'
                    OR (status = 'failed' AND retry_count < max_retries)
                  )
                """,
                (now, now, generated_page_id),
            )
            return cursor.rowcount == 1

    def heartbeat(self, generated_page_id: str) -> None:
        now = utc_now()
        with self.connect() as connection, connection:
            cursor = connection.execute(
                """
                UPDATE pages SET heartbeat_at = ?, updated_at = ?
                WHERE generated_page_id = ? AND status = 'processing'
                """,
                (now, now, generated_page_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Checkpoint page is not processing")

    def requeue_for_manifest_recovery(self, generated_page_id: str) -> None:
        """Reopen a final checkpoint whose portable manifest record is missing."""
        with self.connect() as connection, connection:
            cursor = connection.execute(
                """
                UPDATE pages
                SET status = 'queued', error = NULL,
                    heartbeat_at = NULL, updated_at = ?
                WHERE generated_page_id = ? AND status IN (
                    'complete', 'manual_review', 'skipped', 'quarantined'
                )
                """,
                (utc_now(), generated_page_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Checkpoint page cannot be reopened for recovery")

    def reconcile_page(
        self,
        generated_page_id: str,
        *,
        status: str,
        output_uri: str | None = None,
        output_checksum: str | None = None,
        error: str | None = None,
    ) -> None:
        """Make SQLite agree with a validated, atomically committed JSONL record."""
        if status not in FINAL_PAGE_STATES:
            raise ValueError(f"Invalid final page state: {status}")
        with self.connect() as connection, connection:
            cursor = connection.execute(
                """
                UPDATE pages
                SET status = ?, output_uri = ?, output_checksum = ?,
                    error = ?, updated_at = ?
                WHERE generated_page_id = ?
                """,
                (
                    status,
                    output_uri,
                    output_checksum,
                    error,
                    utc_now(),
                    generated_page_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Checkpoint page does not exist")

    def finish_page(
        self,
        generated_page_id: str,
        *,
        status: str,
        output_uri: str | None = None,
        output_checksum: str | None = None,
        error: str | None = None,
    ) -> None:
        if status not in FINAL_PAGE_STATES:
            raise ValueError(f"Invalid final page state: {status}")
        with self.connect() as connection, connection:
            cursor = connection.execute(
                """
                UPDATE pages
                SET status = ?, output_uri = ?, output_checksum = ?,
                    error = ?, updated_at = ?
                WHERE generated_page_id = ? AND status = 'processing'
                """,
                (
                    status,
                    output_uri,
                    output_checksum,
                    error,
                    utc_now(),
                    generated_page_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Invalid checkpoint page transition")

    def recover_stale(self, *, stale_after_seconds: int = 300) -> dict[str, int]:
        if stale_after_seconds < 1:
            raise ValueError("stale_after_seconds must be positive")
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        with self.connect() as connection, connection:
            retry = connection.execute(
                """
                UPDATE pages
                SET status = 'queued', error = 'stale_worker_recovered', updated_at = ?
                WHERE status = 'processing'
                  AND heartbeat_at < ?
                  AND retry_count < max_retries
                """,
                (utc_now(), cutoff),
            ).rowcount
            failed = connection.execute(
                """
                UPDATE pages
                SET status = 'failed', error = 'stale_worker_retry_limit', updated_at = ?
                WHERE status = 'processing'
                  AND heartbeat_at < ?
                  AND retry_count >= max_retries
                """,
                (utc_now(), cutoff),
            ).rowcount
        return {"requeued": retry, "failed": failed}

    def finish_run(self, run_id: str, state: str) -> None:
        if state not in {"complete", "failed", "interrupted", "cancelled"}:
            raise ValueError("Invalid run state")
        with self.connect() as connection, connection:
            connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (state, utc_now(), run_id),
            )

    def page(self, generated_page_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM pages WHERE generated_page_id = ?",
                (generated_page_id,),
            ).fetchone()
        return dict(row) if row else None

    def summary(self, run_id: str) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM pages WHERE run_id = ? GROUP BY status
                """,
                (run_id,),
            ).fetchall()
            run = connection.execute(
                "SELECT state, updated_at FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return {
            "run_id": run_id,
            "state": run["state"] if run else "missing",
            "updated_at": run["updated_at"] if run else None,
            "statuses": {str(row["status"]): int(row["count"]) for row in rows},
        }
