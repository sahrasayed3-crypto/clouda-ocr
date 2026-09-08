"""End-to-end preparation workflow + CLI smoke tests on the tiny fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clouda_data.pipeline import cli as pipeline_cli
from clouda_data.pretraining import workflow
from clouda_data.pretraining.config import PreparationConfig
from clouda_data.pretraining.manifest import read_manifest
from clouda_data.pretraining.sources import (
    SourceDefinition,
)
from clouda_data.pretraining.workflow import (
    WorkspacePaths,
    prepare_dataset,
    resolve_source,
)

from pretraining_fixture import build_tiny_dataset


@pytest.fixture()
def fixture_roots(tmp_path: Path) -> dict[str, Path]:
    return build_tiny_dataset(tmp_path / "raw")


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


def _register(fixture_roots: dict[str, Path], workspace: Path) -> None:
    workflow.ensure_source_registered(
        workspace,
        SourceDefinition(
            source_id="fixture_a",
            name="Fixture A",
            local_root=str(fixture_roots["source_a"]),
            languages=("ar", "en"),
            classification="training_only",
        ),
    )
    workflow.ensure_source_registered(
        workspace,
        SourceDefinition(
            source_id="fixture_b",
            name="Fixture B",
            local_root=str(fixture_roots["source_b"]),
            adapter="jsonl_records",
            classification="training_only",
        ),
    )


# --------------------------------------------------------------- workflow


def test_prepare_e2e_produces_complete_workspace(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    report = prepare_dataset(workspace, "fixture_a")
    paths = WorkspacePaths(workspace)
    assert paths.manifest.exists()
    assert (workspace / "manifest" / "validation.json").exists()
    assert (workspace / "manifest" / "split_report.json").exists()
    assert (workspace / "manifest" / "dedupe_report.json").exists()
    assert (workspace / "manifest" / "stats.json").exists()
    assert report["split"]["passed"]
    # exports exist for train/validation/test
    export_dir = workspace / "export" / "jsonl"
    assert (export_dir / "train.jsonl").exists()
    assert not (export_dir / "holdout.jsonl").exists()


def test_prepare_is_deterministic_across_reruns(fixture_roots, tmp_path):
    manifests = []
    for run in ("one", "two"):
        ws = tmp_path / f"ws_{run}"
        workflow.ensure_source_registered(
            ws,
            SourceDefinition(
                source_id="fixture_a",
                name="Fixture A",
                local_root=str(fixture_roots["source_a"]),
            ),
        )
        prepare_dataset(ws, "fixture_a", seed=123)
        manifests.append(
            (tmp_path / f"ws_{run}" / "manifest" / "samples.v1.jsonl").read_bytes()
        )
    assert manifests[0] == manifests[1]


def test_prepare_resume_does_not_duplicate_samples(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    first = prepare_dataset(workspace, "fixture_a", seed=9)
    second = prepare_dataset(workspace, "fixture_a", seed=9)
    _, rows_first = read_manifest(workspace / "manifest" / "samples.v1.jsonl")
    _, rows_second = read_manifest(workspace / "manifest" / "samples.v1.jsonl")
    assert len(rows_first) == len(rows_second) == first["samples"]
    assert second["index"]["reused_from_index"] == first["index"]["files"]


def test_prepare_dry_run_writes_nothing(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    prepare_dataset(workspace, "fixture_a", dry_run=True)
    assert not (workspace / "manifest" / "samples.v1.jsonl").exists()


def test_prepare_covers_both_sources_and_all_fixture_cases(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    prepare_dataset(workspace, "fixture_a", seed=5)
    prepare_dataset(workspace, "fixture_b", seed=5)
    _, rows = read_manifest(workspace / "manifest" / "samples.v1.jsonl")
    sources = {row["source_id"] for row in rows}
    assert sources == {"fixture_a", "fixture_b"}
    statuses = {row["validation_status"] for row in rows}
    assert "error" in statuses  # malformed metadata / missing image rows
    dup_states = {row["duplicate_state"] for row in rows}
    assert "duplicate" in dup_states  # exact image duplicate
    scripts = {row["script"] for row in rows}
    assert {"arabic", "latin", "mixed"} <= scripts
    # malformed row and missing-image rows excluded with reasons
    excluded = [r for r in rows if r["validation_status"] == "error"]
    reasons = {r["exclusion_reason"] for r in excluded}
    assert "malformed_metadata" in reasons or "missing_image" in reasons


def test_normalize_stage_preserves_raw_text(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    prepare_dataset(workspace, "fixture_a", seed=5)
    _, before = read_manifest(workspace / "manifest" / "samples.v1.jsonl")
    report = workflow.normalize_workspace(workspace)
    assert report["normalized_now"] == 0  # idempotent second normalization
    _, after = read_manifest(workspace / "manifest" / "samples.v1.jsonl")
    raw_before = {r["sample_id"]: r["raw_text"] for r in before if r.get("raw_text")}
    raw_after = {r["sample_id"]: r["raw_text"] for r in after if r.get("raw_text")}
    assert raw_before == raw_after


def test_stats_and_export_after_full_pipeline(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    prepare_dataset(workspace, "fixture_a", seed=5)
    stats = workflow.stats_workspace(workspace)
    assert stats["total_samples"] > 0
    assert stats["holdout_samples"] >= 0
    assert stats["by_split"].get("unassigned", 0) == stats["excluded_samples"]
    result = workflow.export_workspace(workspace, PreparationConfig())
    assert sum(result.counts.values()) > 0


def test_data_factory_handoff_boundary(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    prepare_dataset(
        workspace,
        "fixture_a",
        seed=5,
        write_handoff=True,
        handoff_profiles=["z_profile", "a_profile", "z_profile"],
        intended_output=str(workspace / "factory_out"),
    )
    request_path = workspace / "handoff" / "data_factory_handoff.json"
    candidates_path = workspace / "handoff" / "data_factory_candidates.jsonl"
    assert request_path.exists() and candidates_path.exists()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["boundary_version"] == "clouda.data_factory.handoff.v1"
    assert request["requested_profiles"] == ["a_profile", "z_profile"]
    assert request["seed"] == 5
    assert request["sample_count"] > 0
    # No clouda-data-factory import/call anywhere in the pretraining package.
    import clouda_data.pretraining as pkg

    pkg_root = Path(pkg.__file__).parent
    for py_file in pkg_root.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "import data_factory" not in text
        assert "from clouda_data_factory" not in text


# -------------------------------------------------------------------- CLI


def _run_cli(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "clouda_data.pipeline.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).parent.parent,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_cli_dataset_prepare_end_to_end(fixture_roots, tmp_path):
    source = str(fixture_roots["source_a"])
    ws = str(tmp_path / "cli_ws")
    code, output = _run_cli("dataset-prepare", source, ws, "--seed", "77")
    assert code == 0, output
    report = json.loads(output)
    assert report["split"]["passed"]
    assert Path(report["manifest"]).exists()


def test_cli_stage_commands_sequence(fixture_roots, tmp_path):
    source = str(fixture_roots["source_a"])
    ws = str(tmp_path / "cli_ws2")
    code, _ = _run_cli(
        "dataset-register-source",
        ws,
        "cli_src",
        "--name",
        "CLI Source",
        "--local-root",
        source,
    )
    assert code == 0
    code, _ = _run_cli("dataset-scan", "cli_src", ws)
    assert code == 0
    # full prepare builds manifest for stage commands
    code, _ = _run_cli("dataset-prepare", "cli_src", ws, "--seed", "7")
    assert code == 0
    for command in (
        ("dataset-validate", ws),
        ("dataset-normalize", ws),
        ("dataset-dedupe", ws),
        ("dataset-split", ws, "--seed", "7"),
        ("dataset-export", ws),
        ("dataset-stats", ws),
    ):
        code, output = _run_cli(*command)
        assert code == 0, f"{command[0]} failed: {output}"


def test_cli_dry_run_writes_nothing(fixture_roots, tmp_path):
    source = str(fixture_roots["source_a"])
    ws = str(tmp_path / "cli_ws3")
    code, output = _run_cli("dataset-prepare", source, ws, "--dry-run")
    assert code == 0, output
    assert not (tmp_path / "cli_ws3" / "manifest" / "samples.v1.jsonl").exists()


def test_cli_invalid_ratios_rejected(tmp_path):
    ws = str(tmp_path / "ws_empty")
    code, _ = _run_cli("dataset-split", ws, "--ratios", "1.0")
    assert code != 0


def test_resolve_source_supports_paths_and_ids(fixture_roots, workspace):
    _register(fixture_roots, workspace)
    resolved = resolve_source(workspace, str(fixture_roots["source_a"]))
    assert resolved.source_id == "fixture_a"
    assert resolve_source(workspace, "fixture_b").source_id == "fixture_b"
    with pytest.raises(KeyError):
        resolve_source(workspace, "unknown_source")


def test_cli_parser_exposes_dataset_commands():
    parser = pipeline_cli.build_parser()
    for command in (
        "dataset-register-source",
        "dataset-scan",
        "dataset-validate",
        "dataset-normalize",
        "dataset-dedupe",
        "dataset-split",
        "dataset-export",
        "dataset-stats",
        "dataset-prepare",
    ):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([command, "--help"])
        assert exc.value.code == 0
