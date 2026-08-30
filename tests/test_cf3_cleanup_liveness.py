"""CF-3 regression tests: cross-process cleanup liveness guard."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from pdfword.database import Database
from pdfword.cleanup import cleanup_temporary_directories


@pytest.fixture
def temp_environment(tmp_path):
    """Set up temp storage root and DB for testing."""
    storage_root = tmp_path / "conversions"
    storage_root.mkdir()
    db_path = tmp_path / "test.db"
    db = Database(path=str(db_path))
    return storage_root, db


def _create_temp_dir(storage_root, family, scope_id, age_hours=48):
    """Create a temporary dir with specified age."""
    temp_dir = storage_root / family / scope_id / "temporary"
    temp_dir.mkdir(parents=True)
    (temp_dir / "file.bin").write_bytes(b"x" * 100)
    if age_hours > 0:
        old_time = time.time() - (age_hours * 3600)
        os.utime(temp_dir, (old_time, old_time))
    return temp_dir


def _insert_conversion(db, owner_user_id="", guest_scope_id="", status="processing"):
    """Insert a conversion row into the database."""
    with db.transaction() as conn:
        conn.execute(
            """INSERT INTO conversions
               (job_id, username, original_pdf_name, stored_pdf_path, status,
                created_at, updated_at, owner_user_id, guest_scope_id, total_cost, processing_time)
               VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?, ?, 0, 0)""",
            (
                f"job-{os.urandom(8).hex()}",
                "testuser",
                "test.pdf",
                "/fake/path.pdf",
                status,
                owner_user_id,
                guest_scope_id,
            ),
        )


class TestCF3ActiveUserScopeSurvives:
    """Test 1: Active user scope survives cleanup."""

    def test_active_user_scope_not_deleted(self, temp_environment):
        storage_root, db = temp_environment
        user_id = "abcdef1234567890abcdef1234567890"
        _create_temp_dir(storage_root, "users", user_id, age_hours=48)
        _insert_conversion(db, owner_user_id=user_id, status="processing")

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=db
        )

        assert result["deleted"] == 0
        assert (storage_root / "users" / user_id / "temporary").exists()

    def test_active_user_pending_also_survives(self, temp_environment):
        storage_root, db = temp_environment
        user_id = "abcdef1234567890abcdef1234567890"
        _create_temp_dir(storage_root, "users", user_id, age_hours=48)
        _insert_conversion(db, owner_user_id=user_id, status="pending")

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=db
        )

        assert result["deleted"] == 0
        assert (storage_root / "users" / user_id / "temporary").exists()


class TestCF3ActiveGuestScopeSurvives:
    """Test 2: Active guest scope survives cleanup."""

    def test_active_guest_scope_not_deleted(self, temp_environment):
        storage_root, db = temp_environment
        guest_id = "aabbccdd11223344aabbccdd11223344"
        _create_temp_dir(storage_root, "guests", guest_id, age_hours=48)
        _insert_conversion(db, guest_scope_id=guest_id, status="pending")

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=db
        )

        assert result["deleted"] == 0
        assert (storage_root / "guests" / guest_id / "temporary").exists()


class TestCF3StaleTerminalScopeDeleted:
    """Test 3: Stale terminal scope is deleted."""

    def test_completed_scope_old_dir_deleted(self, temp_environment):
        storage_root, db = temp_environment
        user_id = "deadbeef12345678deadbeef12345678"
        _create_temp_dir(storage_root, "users", user_id, age_hours=48)
        _insert_conversion(db, owner_user_id=user_id, status="completed")

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=db
        )

        assert result["deleted"] == 1
        assert not (storage_root / "users" / user_id / "temporary").exists()

    def test_failed_scope_old_dir_deleted(self, temp_environment):
        storage_root, db = temp_environment
        user_id = "deadbeef12345678deadbeef12345678"
        _create_temp_dir(storage_root, "users", user_id, age_hours=48)
        _insert_conversion(db, owner_user_id=user_id, status="failed")

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=db
        )

        assert result["deleted"] == 1
        assert not (storage_root / "users" / user_id / "temporary").exists()

    def test_nonexistent_scope_old_dir_deleted(self, temp_environment):
        storage_root, db = temp_environment
        user_id = "orphan1234567890orphan1234567890"
        _create_temp_dir(storage_root, "users", user_id, age_hours=48)

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=db
        )

        assert result["deleted"] == 1
        assert not (storage_root / "users" / user_id / "temporary").exists()


class TestCF3DBIsLivenessSource:
    """Test 4: DB is the cross-process liveness source."""

    def test_db_only_no_job_queue_needed(self, temp_environment):
        """Scope active in DB survives even with empty/broken job queue."""
        storage_root, db = temp_environment
        user_id = "abcdef1234567890abcdef1234567890"
        _create_temp_dir(storage_root, "users", user_id, age_hours=48)
        _insert_conversion(db, owner_user_id=user_id, status="processing")

        with patch("pdfword.cleanup.Database", return_value=db):
            result = cleanup_temporary_directories(
                storage_root=storage_root, retention_hours=24, database=db
            )

        assert result["deleted"] == 0
        assert (storage_root / "users" / user_id / "temporary").exists()


class TestCF3NoJobQueueDependency:
    """Test 5: Cleanup no longer depends on get_job_queue."""

    def test_no_job_queue_import(self):
        """cleanup module does not import or call get_job_queue."""
        import pdfword.cleanup as cleanup_module

        source = Path(cleanup_module.__file__).read_text()
        assert "get_job_queue" not in source
        assert "job_queue" not in source


class TestCF3FailSafe:
    """Test 6: DB error causes fail-safe delete-nothing."""

    def test_db_error_deletes_nothing(self, temp_environment):
        storage_root, db = temp_environment
        user_id = "deadbeef12345678deadbeef12345678"
        _create_temp_dir(storage_root, "users", user_id, age_hours=48)

        class BrokenDB:
            def active_conversion_scope_ids(self):
                raise RuntimeError("DB unavailable")

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=BrokenDB()
        )

        assert result["deleted"] == 0
        assert result["bytes_freed"] == 0
        assert any("DB liveness lookup failed" in e for e in result["errors"])
        assert (storage_root / "users" / user_id / "temporary").exists()


class TestCF3RetentionGuard:
    """Test 7: Inactive but recent directory survives retention cutoff."""

    def test_recent_inactive_dir_not_deleted(self, temp_environment):
        storage_root, db = temp_environment
        user_id = "recentuser123456789012345678901234"[:32]
        _create_temp_dir(storage_root, "users", user_id, age_hours=0)

        result = cleanup_temporary_directories(
            storage_root=storage_root, retention_hours=24, database=db
        )

        assert result["deleted"] == 0
        assert (storage_root / "users" / user_id / "temporary").exists()
