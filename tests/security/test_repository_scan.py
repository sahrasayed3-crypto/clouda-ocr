from __future__ import annotations

import subprocess
from pathlib import Path

from tools.validation.repository_scan import scan_repository


def _repository(tmp_path: Path, content: str) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "file.txt").write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)
    return tmp_path


def test_scan_accepts_explicit_test_placeholder(tmp_path: Path) -> None:
    report = scan_repository(_repository(tmp_path, 'api_key = "test-placeholder-key"'))
    assert report["passed"] is True


def test_scan_detects_private_key_without_exposing_value(tmp_path: Path) -> None:
    marker = "-----BEGIN " + "PRIVATE KEY-----"
    report = scan_repository(_repository(tmp_path, f"{marker}\nnot-printed"))
    assert report["passed"] is False
    assert report["secret_findings"] == [
        {"path": "file.txt", "rule": "private_key", "line": 1}
    ]
    assert "not-printed" not in str(report)


def test_scan_flags_archived_size_and_historical_path_evidence(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    archive = tmp_path / "docs" / "archive" / "audits"
    archive.mkdir(parents=True)
    (archive / "inventory.csv").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    (archive / "history.md").write_text(
        "Historical source: E:" + "\\project_" + "collected",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "docs/archive"], cwd=tmp_path, check=True)

    report = scan_repository(tmp_path)

    assert report["passed"] is False
    assert report["large_files"] == [
        {"path": "docs/archive/audits/inventory.csv", "size_bytes": 5 * 1024 * 1024 + 1}
    ]
    assert report["forbidden_source_paths"] == [
        {
            "path": "docs/archive/audits/history.md",
            "pattern": "E:\\project_collected",
        }
    ]


def test_scan_still_detects_secrets_in_archive(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path,
        "placeholder",
    )
    archive = repository / "docs" / "archive"
    archive.mkdir(parents=True)
    (archive / "legacy.txt").write_text(
        'api_key = "' + "sk-" + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "docs/archive"], cwd=repository, check=True)

    report = scan_repository(repository)

    assert report["passed"] is False
    assert report["secret_findings"] == [
        {
            "path": "docs/archive/legacy.txt",
            "rule": "openai_style_key",
            "line": 1,
        },
        {
            "path": "docs/archive/legacy.txt",
            "rule": "generic_secret_assignment",
            "line": 1,
        },
    ]
