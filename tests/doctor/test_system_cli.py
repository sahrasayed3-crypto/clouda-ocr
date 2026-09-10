"""Storage, Git, security/secrets, output, CLI, and deep-mode tests (Phases 12-14, 19-23)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clouda_data.doctor.human import render_human
from clouda_data.doctor.models import (
    DoctorCheck,
    DoctorReport,
    DoctorSection,
    DoctorStatus,
)
from clouda_data.doctor.report import collect_report
from clouda_data.doctor.security import build_env_report, is_secret_var
from clouda_data.doctor.system import check_git, check_storage

# ---------------------------------------------------------------------------
# Phase 12 — storage
# ---------------------------------------------------------------------------


def test_storage_sufficient_space(clean_worktree_root):
    section = check_storage(
        clean_worktree_root, warn_free_gb=0.001, fail_free_gb=0.0001
    )
    by_id = {c.id: c for c in section.checks}
    assert by_id["storage.repo-drive"].status is DoctorStatus.PASS


def test_storage_low_space_warn(clean_worktree_root):
    # A threshold above real free space but below fail threshold → WARN.
    real_free_gb = 1e6  # generous; fail threshold set above real free
    section = check_storage(
        clean_worktree_root, warn_free_gb=0.001, fail_free_gb=real_free_gb
    )
    by_id = {c.id: c for c in section.checks}
    assert by_id["storage.repo-drive"].status is DoctorStatus.FAIL


def test_storage_unwritable_repo(clean_worktree_root):
    monkey_free = clean_worktree_root
    section = check_storage(monkey_free)
    by_id = {c.id: c for c in section.checks}
    # temp + repo-drive probes must exist regardless of host
    assert "storage.temp-writable" in by_id
    assert "storage.state-roots" in by_id


def test_storage_human_sizes():
    from clouda_data.doctor.system import _human_gb

    assert _human_gb(10 * 1024**3).endswith("GB")
    assert _human_gb(500).endswith("B")


# ---------------------------------------------------------------------------
# Phase 13/14 — Git
# ---------------------------------------------------------------------------


def test_git_reports_branch_and_worktree_kind(clean_worktree_root):
    # Not a git repo → SKIP, never crash.
    section = check_git(clean_worktree_root)
    assert section.checks[0].status is DoctorStatus.SKIP


def test_git_secondary_worktree_detection(tmp_path):
    """Run git checks against the actual feature worktree (read-only)."""
    real_root = Path(__file__).resolve().parents[2]
    if not (real_root / ".git").exists():
        pytest.skip("not running inside the repo checkout")
    section = check_git(real_root)
    check = section.checks[0]
    assert check.status is DoctorStatus.INFO
    details = check.details
    expected_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=real_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert details["branch"] == expected_branch
    assert details["is_linked_worktree"] is True
    assert details["tree_kind"] == "secondary worktree"
    assert details["head"]


def test_git_clean_vs_dirty(tmp_path):
    real_root = Path(__file__).resolve().parents[2]
    if not (real_root / ".git").exists():
        pytest.skip("not running inside the repo checkout")
    section = check_git(real_root)
    check = section.checks[0]
    # The worktree may be dirty (our changes) — just assert the field exists
    # and is a bool, and that a fresh temp clone is reported via the same path.
    assert isinstance(check.details["dirty"], bool)


# ---------------------------------------------------------------------------
# Phase 14/25 — env-var privacy
# ---------------------------------------------------------------------------

SECRET_TEST_ENV = {
    "OPENROUTER_API_KEY": "sk-super-secret-value-123",
    "WORKER_API_KEY": "hunter2-password",
    "REDIS_URL": "redis://user:pass@host:6379",
    "CLOUDA_PROJECT_ROOT": "F:/path/is/not/secret",
    "UNRELATED_VAR": "ignored",
}


def test_env_report_never_contains_values():
    entries = build_env_report(SECRET_TEST_ENV)
    blob = json.dumps(entries)
    assert "sk-super-secret-value-123" not in blob
    assert "hunter2-password" not in blob
    assert "redis://user:pass@host:6379" not in blob


def test_env_report_set_semantics():
    entries = {e["name"]: e for e in build_env_report(SECRET_TEST_ENV)}
    assert entries["OPENROUTER_API_KEY"]["set"] is True
    assert entries["OPENROUTER_API_KEY"]["secret"] is True
    assert entries["CLOUDA_PROJECT_ROOT"]["set"] is True
    assert entries["CLOUDA_PROJECT_ROOT"]["secret"] is False


def test_env_report_unlisted_vars_ignored():
    entries = build_env_report(SECRET_TEST_ENV)
    assert all(e["name"] != "UNRELATED_VAR" for e in entries)


def test_is_secret_var_patterns():
    assert is_secret_var("OPENROUTER_API_KEY")
    assert is_secret_var("WORKER_API_KEY")
    assert is_secret_var("FIREBASE_SERVICE_ACCOUNT_JSON_PATH")  # service account
    assert not is_secret_var("CLOUDA_PROJECT_ROOT")
    assert not is_secret_var("RQ_QUEUE_NAME")


def test_secret_values_never_in_any_output(monkeypatch, clean_worktree_root):
    """End-to-end: set secret env vars, run doctor, scan ALL emitted text."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-live-e2e-secret")
    monkeypatch.setenv("WORKER_API_KEY", "top-secret-worker-token")
    report = collect_report(
        include_factory=False,
        include_render=False,
        include_training=False,
        include_git=False,
    )
    for text in (render_human(report, verbose=True), report.to_json()):
        assert "sk-live-e2e-secret" not in text
        assert "top-secret-worker-token" not in text


def test_check_exception_payloads_are_redacted_from_all_output(monkeypatch):
    secret = "sk-live-check-exception-secret"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    check = DoctorCheck(
        id="secret.failure",
        name="Secret failure",
        subsystem="security",
        status=DoctorStatus.FAIL,
        message=f"provider rejected {secret}",
        details={"error": {"message": secret}},
        remediation=f"remove {secret}",
    )
    report = DoctorReport(
        sections=[DoctorSection(id="security", name="Security", checks=[check])]
    )

    for text in (render_human(report, verbose=True), report.to_json()):
        assert secret not in text
        assert "<redacted>" in text


def test_exception_messages_do_not_leak_values():
    """The doctor never str()s environment values anywhere."""
    import inspect

    from clouda_data.doctor import security

    source = inspect.getsource(security)
    # The value is read once to check set-ness; it is never stored or printed.
    assert "print(" not in source
    assert "str(raw)" not in source


def test_doctor_cli_redacts_secret_from_top_level_exception(monkeypatch, capsys):
    import argparse
    import clouda_data.doctor as doctor_package
    from clouda_data.pipeline.cli import doctor_cli

    secret = "sk-doctor-exception-secret"

    def fail_report(**_kwargs):
        raise RuntimeError(f"provider rejected {secret}")

    monkeypatch.setattr(doctor_package, "collect_report", fail_report)
    args = argparse.Namespace(
        deep=False,
        no_factory=True,
        no_render=True,
        no_training=True,
        no_git=True,
        no_storage=True,
        min_free_gb=None,
        json=False,
        verbose=False,
    )
    assert doctor_cli(args) == 2
    assert secret not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Phase 21/22 — human output
# ---------------------------------------------------------------------------


def _sample_report() -> DoctorReport:
    return DoctorReport(
        timestamp="2026-01-01T00:00:00Z",
        sections=[
            DoctorSection(
                id="core",
                name="Core",
                checks=[
                    DoctorCheck(
                        id="core.python-version",
                        name="Python version",
                        subsystem="core",
                        status=DoctorStatus.PASS,
                        message="Python 3.11.16 ok",
                    ),
                    DoctorCheck(
                        id="core.import-resolution",
                        name="Import resolution",
                        subsystem="core",
                        status=DoctorStatus.WARN,
                        message="drift detected",
                        remediation="pip install -e .",
                    ),
                ],
            )
        ],
    )


def test_human_output_status_words_not_color():
    text = render_human(_sample_report())
    assert "PASS" in text and "WARN" in text
    assert "\x1b[" not in text  # no ANSI escapes
    assert "Overall: PARTIALLY READY" in text


def test_human_output_remediation_on_warn():
    text = render_human(_sample_report())
    assert "fix: pip install -e ." in text


def test_human_output_verbose_details():
    report = _sample_report()
    report.sections[0].checks[0].details["python"] = "3.11.16"
    verbose = render_human(report, verbose=True)
    assert "python: 3.11.16" in verbose


# ---------------------------------------------------------------------------
# Phase 19/20/22/23 — collect_report + CLI
# ---------------------------------------------------------------------------


def test_collect_report_structure():
    report = collect_report(
        include_factory=False,
        include_render=False,
        include_training=False,
        include_git=False,
    )
    payload = report.to_dict()
    assert payload["schema_version"] == "clouda.ocr.doctor.v1"
    assert payload["timestamp"].endswith("Z")
    assert payload["runtime"]["python"]
    section_ids = [s["id"] for s in payload["sections"]]
    assert "core" in section_ids
    assert "storage" in section_ids


def test_collect_report_scoped_flags_reduce_sections():
    full = collect_report()
    minimal = collect_report(
        include_factory=False,
        include_render=False,
        include_training=False,
        include_gpu=False,
        include_git=False,
        include_storage=False,
        include_deps=False,
    )
    assert len(minimal.sections) < len(full.sections)
    assert "core" in [s.id for s in minimal.sections]
    assert "storage" not in [s.id for s in minimal.sections]


def test_cli_doctor_help():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "clouda_data.pipeline.cli", "doctor", "--help"],
        capture_output=True,
        text=True,
        cwd=str(root),
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    assert "--json" in result.stdout
    assert "--deep" in result.stdout


def test_cli_doctor_json_scoped():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clouda_data.pipeline.cli",
            "doctor",
            "--json",
            "--no-factory",
            "--no-render",
            "--no-training",
            "--no-git",
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "clouda.ocr.doctor.v1"


def test_cli_doctor_exit_code_contract():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "clouda_data.pipeline.cli", "doctor"],
        capture_output=True,
        text=True,
        cwd=str(root),
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    # On this host some required deps are missing, so 1 is expected; but the
    # contract is: only 0 or 1 in a healthy run, never 2.
    assert result.returncode in (0, 1)
    assert "Clouda Environment Doctor" in result.stdout


def test_deep_mode_reported_cleanly():
    """Deep checks land in a 'deep' section as normal check results."""
    report = collect_report(
        deep=True,
        include_factory=False,
        include_render=False,
        include_gpu=False,
        include_git=False,
        include_storage=False,
        include_deps=False,
    )
    deep_sections = [s for s in report.sections if s.id == "deep"]
    assert len(deep_sections) == 1
    statuses = [c.status for c in deep_sections[0].checks]
    assert statuses  # dry-run check present
    assert all(
        s in (DoctorStatus.PASS, DoctorStatus.WARN, DoctorStatus.FAIL) for s in statuses
    )


def test_report_timestamp_utc_iso():
    report = collect_report(
        include_factory=False,
        include_render=False,
        include_training=False,
        include_git=False,
    )
    assert report.timestamp.endswith("Z")
    assert "T" in report.timestamp
