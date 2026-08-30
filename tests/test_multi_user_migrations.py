from __future__ import annotations

import sqlite3
from pathlib import Path

from pdfword.database import Database, SCHEMA_VERSION


def table_names(database: Database) -> set[str]:
    with database.connect() as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }


def columns(database: Database, table: str) -> set[str]:
    with database.connect() as connection:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def test_clean_database_has_multi_user_schema(tmp_path: Path):
    database = Database(tmp_path / "clean.sqlite3")

    assert SCHEMA_VERSION >= 7
    assert {
        "auth_users",
        "auth_identities",
        "auth_sessions",
        "auth_audit_events",
        "guest_sessions",
        "guest_jobs",
        "auth_usage_events",
    }.issubset(table_names(database))
    assert {"owner_user_id", "guest_scope_id", "visibility"}.issubset(
        columns(database, "conversions")
    )


def test_schema_upgrade_preserves_existing_conversion(tmp_path: Path):
    path = tmp_path / "upgrade.sqlite3"
    database = Database(path)
    database.create_conversion(
        {
            "job_id": "legacy-job",
            "username": "legacy",
            "original_pdf_name": "legacy.pdf",
            "stored_pdf_path": str(tmp_path / "legacy.pdf"),
            "output_docx_name": "legacy.docx",
            "stored_docx_path": str(tmp_path / "legacy.docx"),
            "page_from": 1,
            "page_to": 1,
            "status": "pending",
            "created_at": "2026-07-29T00:00:00+00:00",
            "updated_at": "2026-07-29T00:00:00+00:00",
        }
    )

    upgraded = Database(path)

    conversion = upgraded.get_conversion("legacy-job")
    assert conversion is not None
    assert conversion["username"] == "legacy"
    assert "owner_user_id" in columns(upgraded, "conversions")


def test_representative_schema_v6_upgrades_to_v7_without_ownership_corruption(
    tmp_path: Path,
):
    path = tmp_path / "legacy_v6.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE schema_meta(version INTEGER NOT NULL);
            INSERT INTO schema_meta(version) VALUES (6);
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL,
                last_login TEXT NOT NULL
            );
            CREATE TABLE conversions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL,
                original_pdf_name TEXT NOT NULL,
                stored_pdf_path TEXT NOT NULL,
                output_docx_name TEXT,
                stored_docx_path TEXT,
                page_from INTEGER,
                page_to INTEGER,
                file_type TEXT,
                text_quality_score REAL,
                layout_quality_score REAL,
                final_quality_score REAL,
                winning_engine TEXT,
                winning_model TEXT,
                total_cost REAL NOT NULL DEFAULT 0,
                processing_time REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                hidden INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT
            );
            INSERT INTO users(username, created_at, last_login)
            VALUES ('legacy-user', '2026-07-28T00:00:00+00:00', '2026-07-28T00:00:00+00:00');
            INSERT INTO conversions(
                job_id, username, original_pdf_name, stored_pdf_path,
                output_docx_name, stored_docx_path, page_from, page_to,
                status, created_at, updated_at
            ) VALUES (
                'legacy-v6-job', 'legacy-user', 'legacy.pdf', 'legacy/input.pdf',
                'legacy.docx', 'legacy/output.docx', 1, 2, 'queued',
                '2026-07-28T00:00:00+00:00', '2026-07-28T00:00:00+00:00'
            );
            INSERT INTO conversions(
                job_id, username, original_pdf_name, stored_pdf_path,
                output_docx_name, stored_docx_path, page_from, page_to,
                status, created_at, updated_at
            ) VALUES (
                'legacy-email-job', 'owner@example.com', 'owner.pdf',
                'owner/input.pdf', 'owner.docx', 'owner/output.docx',
                1, 1, 'completed',
                '2026-07-28T00:00:00+00:00', '2026-07-28T00:00:00+00:00'
            );
            """)

    upgraded = Database(path)
    row = upgraded.get_conversion("legacy-v6-job")
    email_row = upgraded.get_conversion("legacy-email-job")

    assert row is not None
    assert row["username"] == "legacy-user"
    assert row["status"] == "pending"
    assert row["owner_user_id"] == ""
    assert row["guest_scope_id"] == ""
    assert row["visibility"] == "legacy"
    assert row["lifecycle_state"] == "legacy_unclaimed"
    assert email_row is not None
    assert email_row["owner_user_id"]
    assert email_row["visibility"] == "private"
    owner = upgraded.get_auth_user(email_row["owner_user_id"])
    assert owner is not None
    assert owner["normalized_email"] == "owner@example.com"
    assert table_names(upgraded) >= {
        "auth_users",
        "auth_sessions",
        "guest_jobs",
    }
    with upgraded.connect() as connection:
        version = connection.execute("SELECT version FROM schema_meta").fetchone()[0]
    assert version == SCHEMA_VERSION
