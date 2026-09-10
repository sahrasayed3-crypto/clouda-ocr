"""CLI tests: help, ingest, verify, listing, show-page, export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_data.results import cli as results_cli
from clouda_data.results.service import ResultsService
from tests.results.fixtures_e2e import (
    build_fixture_ground_truth,
    build_fixture_pages,
    build_fixture_predictions,
)


@pytest.fixture()
def store(tmp_path):
    svc = ResultsService(tmp_path / "store")
    pages = build_fixture_pages(dataset_id="cli-demo")
    run = svc.create_run(
        model_id="fake-model-a",
        model_revision="1.0",
        dataset_id="cli-demo",
        dataset_version="v1",
        created_at="t1",
    )
    svc.add_pages(run.run_id, pages)
    svc.store.append_ground_truth(run.run_id, build_fixture_ground_truth(pages))
    predictions = build_fixture_predictions(run.run_id, "fake-model-a", "1.0", pages)
    svc.store.append_predictions(run.run_id, predictions)
    for prediction in predictions:
        svc.evaluate_prediction(run.run_id, prediction)
    svc.finalize_run(run.run_id)
    return tmp_path / "store", run.run_id


def _run_cli(capsys, *argv: str) -> int:
    return results_cli.main(list(argv))


class TestHelp:
    def test_help_lists_commands(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            _run_cli(capsys, "--help")
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for command in (
            "ingest",
            "verify",
            "list-runs",
            "list-models",
            "show-page",
            "worst-pages",
            "export",
        ):
            assert command in out


class TestListings:
    def test_list_runs(self, capsys, store) -> None:
        root, run_id = store
        assert _run_cli(capsys, "list-runs", "--store", str(root)) == 0
        runs = json.loads(capsys.readouterr().out)
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id

    def test_list_models_empty(self, capsys, store) -> None:
        root, _ = store
        assert _run_cli(capsys, "list-models", "--store", str(root)) == 0
        assert json.loads(capsys.readouterr().out) == []


class TestVerify:
    def test_verify_ok(self, capsys, store) -> None:
        root, run_id = store
        assert _run_cli(capsys, "verify", run_id, "--store", str(root)) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["ok"] is True

    def test_verify_unknown_run(self, capsys, store) -> None:
        root, _ = store
        code = _run_cli(capsys, "verify", "ghost", "--store", str(root))
        assert code == 2


class TestShowPage:
    def test_show_page(self, capsys, store) -> None:
        root, run_id = store
        assert (
            _run_cli(capsys, "show-page", run_id, "fx_page_001", "--store", str(root))
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["page"]["page_id"] == "fx_page_001"
        assert "القَاهِرَة" in payload["ground_truth"]["raw_text"]
        assert payload["predictions"][0]["model_id"] == "fake-model-a"

    def test_show_missing_page(self, capsys, store) -> None:
        root, run_id = store
        code = _run_cli(capsys, "show-page", run_id, "nope", "--store", str(root))
        assert code == 2


class TestWorstPages:
    def test_worst_pages(self, capsys, store) -> None:
        root, run_id = store
        assert (
            _run_cli(
                capsys,
                "worst-pages",
                run_id,
                "--store",
                str(root),
                "--limit",
                "2",
            )
            == 0
        )
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) <= 2
        values = [row["value"] for row in rows]
        assert values == sorted(values, reverse=True)


class TestExport:
    def test_export_round_trip(self, capsys, store, tmp_path) -> None:
        root, run_id = store
        target = tmp_path / "export"
        assert (
            _run_cli(capsys, "export", run_id, str(target), "--store", str(root)) == 0
        )
        exported = json.loads(capsys.readouterr().out)
        assert exported["exported"]
        # The export is a portable store root (runs/<run_id>/...).
        assert (target / "runs" / run_id / "metadata.json").exists()
        assert (target / "runs" / run_id / "predictions.jsonl").exists()


class TestIngestCommand:
    def test_ingest_real_manifest(self, capsys, tmp_path) -> None:
        manifest = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/ocr_arabic/benchmark_manifest.jsonl"
        )
        root = tmp_path / "ingest-store"
        code = _run_cli(capsys, "ingest", str(manifest), "--store", str(root))
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["pages_total"] == 177
        assert payload["pages_registered"] == 177
        assert payload["ok"] is True
        # Idempotent re-ingestion into a fresh run: dataset row count stable.
        datasets = list(ResultsService(root, read_only=True).list_datasets())
        assert len(datasets) == 1
        assert datasets[0]["page_count"] == 177
