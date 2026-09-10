"""CLI surface tests for the integrated Data Factory.

The standalone repo shipped a separate ``clouda-data-factory`` executable;
the integrated surface is ``python -m clouda_data.factory`` plus the
canonical ``clouda-data`` CLI factory-* commands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "clouda_data.factory", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_help_describes_core_options():
    result = run_cli("--help")
    assert result.returncode == 0
    for flag in ("generate", "profiles", "verify", "seeds", "run"):
        assert flag in result.stdout


def test_seeds_prints_vectors():
    result = run_cli("seeds")
    assert result.returncode == 0
    vectors = json.loads(result.stdout)
    assert set(vectors) == {"v1", "ocr_benchmark", "arabic_scan_factory"}
    assert all(isinstance(v, int) for v in vectors.values())


def test_profiles_lists_all_20():
    result = run_cli("profiles")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert len(payload["profiles"]) == 20
    assert "05_old_book_medium" in payload["profiles"]
    assert "old_book_medium" in payload["profiles"]
    assert set(payload["backends"]) == {"raqm", "weasyprint"}


def test_generate_rejects_unknown_profile(tmp_path):
    result = run_cli(
        "generate",
        "no-input.txt",
        "--output",
        tmp_path / "r",
        "--profiles",
        "not_a_profile",
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_verify_reports_missing_run(tmp_path):
    result = run_cli("verify", tmp_path)
    assert result.returncode != 0


def test_canonical_cli_registers_factory_commands():
    result = subprocess.run(
        [sys.executable, "-m", "clouda_data.pipeline.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    for command in (
        "factory-generate",
        "factory-run",
        "factory-profiles",
        "factory-verify",
        "factory-seeds",
        "factory-manifest",
    ):
        assert command in result.stdout


def test_canonical_cli_factory_seeds_and_profiles():
    for command in ("factory-seeds", "factory-profiles"):
        result = subprocess.run(
            [sys.executable, "-m", "clouda_data.pipeline.cli", command],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload


def test_module_main_entry_prints_version():
    result = run_cli("version")
    assert result.returncode == 0
    assert json.loads(result.stdout)["factory_version"]


def test_factory_cli_help_imports_without_heavy_factory_dependencies():
    script = r"""
import builtins
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"cv2", "numpy", "img2pdf", "pikepdf"}:
        raise ImportError(f"blocked optional dependency: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
from clouda_data.factory.cli import build_parser
build_parser().parse_args(["--help"])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
