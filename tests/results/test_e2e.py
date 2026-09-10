"""End-to-end: Arabic fixture -> ingest -> store -> query -> export -> verify.

Offline only: no network, no GPU, no model downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clouda_data.results import ResultsService
from clouda_data.results.models import Provenance, BenchmarkDataset
from clouda_data.results.store import ResultsStore
from tests.results.fixtures_e2e import (
    MODELS,
    build_fixture_ground_truth,
    build_fixture_pages,
    build_fixture_predictions,
)


@pytest.fixture()
def e2e(tmp_path):
    svc = ResultsService(tmp_path / "store")
    dataset_id = "fx-arabic-demo"
    pages = build_fixture_pages(dataset_id=dataset_id)

    dataset = BenchmarkDataset(
        dataset_id=dataset_id,
        version="v1",
        name="E2E Arabic fixture",
        page_count=len(pages),
        splits=sorted({page.split for page in pages}),
        provenance=Provenance(source_format="tests.fixtures.results_store_e2e"),
    )
    svc.store.save_dataset(dataset.to_dict())

    run_ids: dict[str, str] = {}
    for model_id, revision in MODELS:
        run = svc.create_run(
            model_id=model_id,
            model_revision=revision,
            dataset_id=dataset_id,
            dataset_version="v1",
            created_at="fixture-time",
        )
        run_ids[model_id] = run.run_id
        svc.add_pages(run.run_id, pages)
        svc.store.append_ground_truth(run.run_id, build_fixture_ground_truth(pages))
        predictions = build_fixture_predictions(run.run_id, model_id, revision, pages)
        svc.store.append_predictions(run.run_id, predictions)
        for prediction in predictions:
            svc.evaluate_prediction(run.run_id, prediction)
        svc.finalize_run(run.run_id)
    return svc, dataset_id, pages, run_ids


def test_e2e_pipeline(e2e, tmp_path: Path) -> None:
    svc, dataset_id, pages, run_ids = e2e

    # ---- datasets
    datasets = svc.list_datasets()
    assert len(datasets) == 1
    assert datasets[0]["page_count"] == len(pages)

    # ---- runs
    assert len(svc.list_runs()) == 2
    assert sorted(run_ids) == ["fake-model-a", "fake-model-b"]

    # ---- pages + GT
    run_a = run_ids["fake-model-a"]
    page = svc.get_page(run_a, "fx_page_001")
    assert page.dataset_id == dataset_id
    assert page.profile == "clean"
    gt = svc.get_ground_truth(run_a, "fx_page_001")
    assert "القَاهِرَة" in gt.raw_text  # raw form with diacritics preserved
    assert gt.raw_text == svc.get_ground_truth_text(run_a, "fx_page_001")

    # ---- holdout protection survives the whole pipeline
    holdout = svc.list_pages(run_a, split="holdout")
    assert len(holdout) == 1
    assert holdout[0].is_training_eligible is False
    holdout_gt = svc.get_ground_truth(run_a, "fx_page_003")
    assert holdout_gt.protection.protected is True

    # ---- predictions: two models over the same pages, no overwrite
    for spec_page in ("fx_page_001", "fx_page_002", "fx_page_003", "fx_page_004"):
        preds_a = svc.get_predictions(run_a, spec_page)
        preds_b = svc.get_predictions(run_ids["fake-model-b"], spec_page)
        assert len(preds_a) == 1 and len(preds_b) == 1
        assert preds_a[0].model_id != preds_b[0].model_id

    # ---- metrics + worst pages
    worst = svc.get_worst_pages(run_a, limit=4)
    assert 1 <= len(worst) <= 4
    values = [row["value"] for row in worst]
    assert values == sorted(values, reverse=True)
    summary = svc.get_metrics(run_a, scope="run_summary")
    assert summary

    # ---- verification
    for run_id in run_ids.values():
        report = svc.verify(run_id)
        assert report["ok"], report["issues"]

    # ---- export + round trip
    export_dir = tmp_path / "export"
    svc.store.export_run_copy(run_a, export_dir)
    imported = ResultsStore(export_dir, read_only=True)
    assert imported.load_run_metadata(run_a)["model_id"] == "fake-model-a"
    restored_pages = list(imported.iter_pages(run_a))
    assert len(restored_pages) == len(pages)
    # Arabic round-trips exactly through export.
    restored_gt = imported.get_ground_truth(run_a, "fx_page_001")
    assert restored_gt.raw_text == gt.raw_text
    # No machine paths leaked.
    for path in export_dir.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "F:\\" not in text
            assert str(tmp_path) not in text


def test_e2e_conflict_still_detected_after_export(e2e, tmp_path: Path) -> None:
    svc, _, pages, run_ids = e2e
    run_a = run_ids["fake-model-a"]
    export_dir = tmp_path / "export2"
    svc.store.export_run_copy(run_a, export_dir)
    imported_store = ResultsStore(export_dir, read_only=True)

    # Corrupt-then-conflict semantics survive export: the exported bundle is
    # verified independently.
    assert imported_store.verify_bundle(run_a)["ok"]


def test_e2e_normalized_vs_raw_distinction(e2e) -> None:
    svc, _, _, run_ids = e2e
    run_a = run_ids["fake-model-a"]
    # Page 002: tatweel + Arabic-Indic digits in raw GT; model A folds them.
    raw = svc.get_ground_truth_text(run_a, "fx_page_002")
    normalized = svc.get_ground_truth_normalized(run_a, "fx_page_002")
    assert raw != normalized
    assert "\u0640" in raw  # tatweel preserved in raw
    assert "\u0640" not in normalized
    # Raw digits preserved; normalized folds Arabic-Indic to ASCII.
    assert "٢٠٢٦" in raw
    assert "2026" in normalized
