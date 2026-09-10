"""Tests: Training Orchestrator + selection→training pipeline (Phases 14-16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from clouda_lab.dataset_selection import SelectionCriteria
from clouda_lab.models import RunAnalysis
from clouda_lab.run_pipeline import analyze_run
from clouda_lab.selection_history import SelectionHistory
from clouda_lab.training_orchestrator import TrainingOrchestrator
from clouda_training.experiments import ConfigError, RunStatus

SCHEMA = "clouda.pretraining.manifest.v1"


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    lines = [
        json.dumps(
            {
                "_schema_version": SCHEMA,
                "_row_count": len(rows),
                "dataset_role": "training",
                "dataset_id": "lab-fixture",
                "dataset_version": "v1",
            },
            sort_keys=True,
        )
    ]
    lines.extend(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(sample_id: str, split: str = "train") -> dict:
    return {
        "sample_id": sample_id,
        "target_split": split,
        "source_id": "synthetic-source",
        "source_license": "Apache-2.0",
        "document_type": "book",
        "text": f"نص تجريبي {sample_id}",
    }


@pytest.fixture()
def base_manifest(tmp_path: Path) -> Path:
    return _write_manifest(
        tmp_path / "base.jsonl",
        [_row(f"s-{i:03d}") for i in range(1, 9)],
    )


class TestSelectionToExperiment:
    def test_pipeline_creates_manifest_and_config(
        self, base_manifest: Path, tmp_path: Path
    ):
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        result = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(base_manifest),
            criteria=SelectionCriteria(limit=4),
            seed=5,
            output_dir=tmp_path / "lab",
            experiment_name="lab_selection_test",
        )
        derived = Path(result["derived_manifest"])
        assert derived.is_file()
        assert result["derived_manifest_sha256"]
        assert result["validation"]["rows"] == 4
        assert result["selection"]["excluded_protected"] == 0
        header = result["manifest_header"]
        assert header["selection_id"]
        assert header["selection_seed"] == 5
        assert header["source_manifest_sha256"]
        config_path = Path(result["experiment_config_path"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["dataset"]["manifest_path"] == str(derived)
        assert config["runtime"]["dry_run"] is True
        assert config["model"]["adapter_type"] == "mock"

    def test_pipeline_rejects_protected_manifest(self, tmp_path: Path):
        # All rows protected -> selection is empty -> PermissionError.
        protected = _write_manifest(
            tmp_path / "prot.jsonl",
            [_row("h-1", split="holdout"), _row("h-2", split="benchmark_holdout")],
        )
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        with pytest.raises(Exception):
            orchestrator.create_training_experiment_from_selection(
                manifest_path=str(protected),
                criteria=SelectionCriteria(limit=5),
                output_dir=tmp_path / "lab",
                experiment_name="should_fail",
            )

    def test_pipeline_excludes_protected_rows(self, tmp_path: Path):
        # Mixed manifest: protected rows excluded, clean rows trainable.
        mixed = _write_manifest(
            tmp_path / "mixed.jsonl",
            [_row("h-1", split="holdout"), _row("ok-1"), _row("ok-2")],
        )
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        result = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(mixed),
            criteria=SelectionCriteria(limit=5),
            output_dir=tmp_path / "lab",
            experiment_name="mixed_ok",
        )
        assert result["selection"]["excluded_protected"] == 1
        assert "h-1" not in result["selection"]["sample_ids"]

    def test_pipeline_records_history(self, base_manifest: Path, tmp_path: Path):
        history = SelectionHistory(tmp_path / "lab" / "history.jsonl")
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        orchestrator.create_training_experiment_from_selection(
            manifest_path=str(base_manifest),
            criteria=SelectionCriteria(limit=2),
            output_dir=tmp_path / "lab",
            experiment_name="hist_test",
            history=history,
        )
        used = history.used_sample_ids()
        assert len(used) == 2
        records = history.load()
        assert records[0]["purpose"] == "training_subset"

    def test_empty_selection_refused(self, base_manifest: Path, tmp_path: Path):
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        with pytest.raises(ValueError, match="produced no rows"):
            orchestrator.create_training_experiment_from_selection(
                manifest_path=str(base_manifest),
                criteria=SelectionCriteria(dataset_id="no-such-dataset"),
                output_dir=tmp_path / "lab",
                experiment_name="empty",
            )


class TestOrchestratorDryRun:
    def test_full_lifecycle(self, base_manifest: Path, tmp_path: Path):
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        result = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(base_manifest),
            criteria=SelectionCriteria(limit=4),
            output_dir=tmp_path / "lab",
            experiment_name="dryrun_test",
            dry_run=True,
        )
        handle = orchestrator.start_dry_run(result["experiment_config_path"])
        run_id = handle["run_id"]
        assert orchestrator.run_status(run_id) == RunStatus.COMPLETED.value
        inspection = orchestrator.inspect_run(run_id)
        assert inspection["summary"]["status"] == RunStatus.COMPLETED.value
        assert inspection["metrics"], "dry run must log metrics"
        checkpoints = orchestrator.get_checkpoints(run_id)
        assert checkpoints
        runs = orchestrator.list_runs("dryrun_test")
        assert run_id in {r["run_id"] for r in runs}

    def test_real_training_refused(self, base_manifest: Path, tmp_path: Path):
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        result = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(base_manifest),
            criteria=SelectionCriteria(limit=2),
            output_dir=tmp_path / "lab",
            experiment_name="no_real_training",
            config_overrides={
                "model.adapter_type": "real_gpu",
                "runtime.dry_run": False,
            },
        )
        with pytest.raises(Exception):
            orchestrator.start_dry_run(result["experiment_config_path"])

    def test_compare_and_resume_passthrough(self, base_manifest: Path, tmp_path: Path):
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        first = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(base_manifest),
            criteria=SelectionCriteria(limit=4),
            output_dir=tmp_path / "lab",
            experiment_name="cmp",
        )
        handle_a = orchestrator.start_dry_run(first["experiment_config_path"])
        handle_b = orchestrator.start_dry_run(first["experiment_config_path"])
        comparison = orchestrator.compare(handle_a["run_id"], handle_b["run_id"])
        assert "metric_differences" in comparison
        # resume passthrough: the framework rejects completed runs, proving
        # the orchestrator forwards without swallowing framework rules.

        with pytest.raises(ValueError, match="not resumable"):
            orchestrator.resume(handle_a["run_id"])

    def test_dry_run_refused_for_nondry_config(self, tmp_path: Path):
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        config = {
            "schema_version": 1,
            "experiment": {"name": "not_dry"},
            "model": {"model_id": "mock/x", "adapter_type": "mock"},
            "dataset": {
                "dataset_id": "d",
                "dataset_version": "v",
                "manifest_path": "missing.jsonl",
                "split": "train",
            },
            "runtime": {
                "device": "cpu",
                "output_root": str(tmp_path / "runs"),
                "dry_run": False,
                "offline": True,
                "deterministic": True,
            },
        }
        path = tmp_path / "not_dry.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ConfigError, match="dry_run"):
            orchestrator.start_dry_run(path)


class TestPostRunPipeline:
    def test_analyze_run_end_to_end(self, base_manifest: Path, tmp_path: Path):
        from clouda_lab.failure_analysis import SampleMetrics

        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        prepared = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(base_manifest),
            criteria=SelectionCriteria(limit=4),
            output_dir=tmp_path / "lab",
            experiment_name="pipeline",
        )
        handle = orchestrator.start_dry_run(prepared["experiment_config_path"])
        run_id = handle["run_id"]

        baseline = [
            SampleMetrics(
                sample_id="s-001",
                cer=0.50,
                wer=0.60,
                ncer=0.45,
                error_types={"whitespace": 2},
            ),
            SampleMetrics(sample_id="s-002", cer=0.20, wer=0.30, ncer=0.15),
            SampleMetrics(sample_id="s-003", cer=0.70, wer=0.80, ncer=0.65),
        ]
        candidate = [
            SampleMetrics(
                sample_id="s-001",
                cer=0.25,
                wer=0.35,
                ncer=0.20,
                error_types={"whitespace": 0},
            ),
            SampleMetrics(sample_id="s-002", cer=0.45, wer=0.55, ncer=0.40),
            SampleMetrics(sample_id="s-003", cer=0.72, wer=0.80, ncer=0.66),
        ]
        rows = [
            {
                "sample_id": "s-001",
                "cer": 0.25,
                "wer": 0.35,
                "ncer": 0.20,
                "error_type_counts": {"whitespace": 1},
                "model_id": "m1",
                "profile": "bad_scan_heavy",
            },
            {
                "sample_id": "s-002",
                "cer": 0.45,
                "wer": 0.55,
                "ncer": 0.40,
                "error_type_counts": {"diacritic": 2},
                "model_id": "m1",
                "profile": "clean",
            },
            {
                "sample_id": "s-003",
                "cer": 0.72,
                "wer": 0.80,
                "ncer": 0.66,
                "error_type_counts": {"missing_word": 3},
                "model_id": "m1",
                "profile": "clean",
            },
        ]
        analysis = analyze_run(
            run_id=run_id,
            orchestrator=orchestrator,
            baseline_metrics=baseline,
            candidate_metrics=candidate,
            baseline_id="baseline",
            rows=rows,
            recommendation_batch_size=2,
            seed=3,
        )
        assert isinstance(analysis, RunAnalysis)
        assert analysis.status == RunStatus.COMPLETED.value
        assert analysis.metrics_summary["available"] is True
        assert analysis.failure_report is not None
        assert (
            analysis.failure_report.counts["recovered"] == 1
        )  # s-001 crosses failure line
        assert analysis.failure_report.counts["regressed"] == 1  # s-002
        assert analysis.failure_report.counts["unchanged"] == 1  # s-003
        assert analysis.hard_examples
        assert analysis.hard_examples[0].sample_id == "s-003"  # highest CER
        assert analysis.recommendation is not None
        assert len(analysis.recommendation.selections) == 2
        payload = json.dumps(analysis.to_dict(), ensure_ascii=False)
        assert "failure_report" in payload

    def test_mock_trainer_metrics_never_used_as_ocr_truth(
        self, base_manifest: Path, tmp_path: Path
    ):
        # The pipeline consumes caller-provided per-sample metrics; dry-run
        # mock metrics stay confined to run artifacts. Guard: analyze_run
        # requires explicit metric inputs.
        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        prepared = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(base_manifest),
            criteria=SelectionCriteria(limit=2),
            output_dir=tmp_path / "lab",
            experiment_name="no_mock_truth",
        )
        handle = orchestrator.start_dry_run(prepared["experiment_config_path"])
        metrics_summary = orchestrator.inspect_run(handle["run_id"])["summary"]
        # Mock final metrics exist but carry no per-sample CER claims
        assert "final_metrics" in metrics_summary
        with pytest.raises(TypeError):
            analyze_run(run_id=handle["run_id"], orchestrator=orchestrator)  # type: ignore[call-arg]
