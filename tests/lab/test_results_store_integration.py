from __future__ import annotations

import pytest

from clouda_data.results.models import (
    BenchmarkDataset,
    EvaluationRecord,
    ModelRecord,
    PageRecord,
    ProtectionInfo,
)
from clouda_data.results.metrics import compute_page_metrics
from clouda_data.results.service import ResultsService
from clouda_data.results.store import ConflictingRecordError, ResultsStore
from clouda_lab import StoredResultsAnalysisService
from clouda_lab.dataset_selection import SelectionCriteria
from tests.results.fixtures_e2e import (
    build_fixture_ground_truth,
    build_fixture_pages,
    build_fixture_predictions,
)


def _create_stored_run(
    root,
    *,
    model_id: str = "fake-model-a",
    revision: str = "1.0",
    created_at: str = "2026-09-10T00:00:00Z",
) -> tuple[ResultsService, str]:
    results = ResultsService(root)
    results.register_dataset(dataset_id="fx-arabic-demo", version="v1")
    results.register_model(ModelRecord(model_id=model_id, revision=revision))
    run = results.create_run(
        model_id=model_id,
        model_revision=revision,
        dataset_id="fx-arabic-demo",
        dataset_version="v1",
        split="mixed",
        created_at=created_at,
    )
    pages = build_fixture_pages()
    results.add_pages(run.run_id, pages)
    results.store.append_ground_truth(run.run_id, build_fixture_ground_truth(pages))
    results.store.append_predictions(
        run.run_id,
        build_fixture_predictions(run.run_id, model_id, revision, pages),
    )
    return results, run.run_id


def test_analyze_page_reads_canonical_ground_truth_and_prediction(tmp_path) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")

    analysis = StoredResultsAnalysisService(results).analyze_page(run_id, "fx_page_001")

    assert analysis.page_id == "fx_page_001"
    assert analysis.run_id == run_id
    assert analysis.model_id == "fake-model-a"
    assert analysis.dataset_id == "fx-arabic-demo"
    assert analysis.cer > 0.0
    assert analysis.metadata["dataset_version"] == "v1"
    assert analysis.metadata["training_eligible"] is True


def test_stored_sample_preserves_canonical_page_metadata_and_eligibility(
    tmp_path,
) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")

    sample = StoredResultsAnalysisService(results).sample(run_id, "fx_page_002")

    assert sample.metadata["dataset_version"] == "v1"
    assert sample.metadata["document_id"] == "fx_doc_1"
    assert sample.metadata["split"] == "train"
    assert sample.metadata["profile"] == "old_book_medium"
    assert sample.metadata["distortion_seed"] == 1002
    assert sample.metadata["training_eligible"] is True
    assert sample.metadata["protection"] == {
        "protected": False,
        "reasons": [],
        "split": "train",
        "is_training_eligible": True,
    }


def test_malformed_protection_and_nested_source_markers_fail_closed(tmp_path) -> None:
    malformed = ProtectionInfo.from_dict({"protected": "maybe", "split": "train"})
    assert malformed.protected is True
    assert malformed.is_training_eligible is False

    results = ResultsService(tmp_path / "results")
    run = results.create_run(
        model_id="model",
        dataset_id="dataset",
        dataset_version="v1",
        split="train",
        created_at="t",
    )
    source = build_fixture_pages(dataset_id="dataset")[0]
    page = source.__class__(
        **{
            **source.to_dict(),
            "split": "train",
            "metadata": {
                "canonical_manifest_row": {
                    "sample_id": source.page_id,
                    "target_split": "train",
                    "protected": True,
                }
            },
        }
    )
    results.add_page(run.run_id, page)

    selection = StoredResultsAnalysisService(results).select_pages(
        run.run_id, SelectionCriteria(split="train")
    )
    assert selection.sample_ids == ()
    assert selection.excluded_protected == 1


def test_analyze_and_compare_stored_runs_without_parallel_result_index(
    tmp_path,
) -> None:
    root = tmp_path / "results"
    results, baseline_run = _create_stored_run(root)
    _same_results, candidate_run = _create_stored_run(
        root,
        model_id="fake-model-b",
        revision="0.9",
        created_at="2026-09-10T00:01:00Z",
    )
    bridge = StoredResultsAnalysisService(results)

    analyses = bridge.analyze_run(baseline_run)
    rows = bridge.analysis_rows(baseline_run)
    comparison = bridge.compare_runs(baseline_run, candidate_run)

    assert len(analyses) == 4
    assert [item.page_id for item in analyses] == sorted(
        item.page_id for item in analyses
    )
    assert rows[0]["sample_id"] == analyses[0].page_id
    assert rows[0]["ncer"] == analyses[0].normalized_cer
    assert rows[0]["dataset_version"] == "v1"
    assert rows[0]["training_eligible"] is True
    assert comparison.baseline_id == baseline_run
    assert comparison.candidate_id == candidate_run
    assert len(comparison.comparisons) == 4


def test_select_pages_uses_canonical_metadata_and_excludes_holdout(tmp_path) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")
    bridge = StoredResultsAnalysisService(results)

    selection = bridge.select_pages(run_id, SelectionCriteria(), seed=17)

    assert selection.sample_ids == (
        "fx_page_001",
        "fx_page_002",
        "fx_page_004",
    )
    assert selection.excluded_protected == 1
    assert selection.source_manifest == f"results://runs/{run_id}/pages"
    assert len(selection.source_manifest_sha256) == 64
    assert all(row["dataset_version"] == "v1" for row in selection.rows)
    assert all(row["protected"] is False for row in selection.rows)
    assert {row["sample_id"]: row["training_eligible"] for row in selection.rows} == {
        "fx_page_001": True,
        "fx_page_002": True,
        "fx_page_004": False,
    }


def test_results_contract_uses_shared_fail_closed_protection_policy() -> None:
    assert ProtectionInfo(split="train").is_training_eligible is True
    assert ProtectionInfo(split="validation").is_training_eligible is False
    assert ProtectionInfo(split="test").is_training_eligible is False
    assert ProtectionInfo(split="unassigned").is_training_eligible is False
    assert ProtectionInfo(split=" train+HOLDOUT ").protected is True

    train_page = PageRecord(
        page_id="train-page",
        document_id="doc",
        dataset_id="dataset",
        split="train",
    )
    assert train_page.is_training_eligible is True

    page = PageRecord(
        page_id="protected-role",
        document_id="doc",
        dataset_id="dataset",
        split="train",
        metadata={"role": "benchmark"},
    )
    assert page.protection.protected is True
    assert page.is_training_eligible is False


def test_recommendation_from_store_is_eligible_explainable_and_source_linked(
    tmp_path,
) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")
    bridge = StoredResultsAnalysisService(results)

    rows = bridge.recommend_next(
        run_id,
        strategy="hardest_only",
        batch_size=10,
        seed=23,
        history=("fx_page_001",),
    )

    assert [row["sample_id"] for row in rows] == ["fx_page_002"]
    assert rows[0]["page_id"] == "fx_page_002"
    assert isinstance(rows[0]["score"], float)
    assert rows[0]["bucket"]
    assert rows[0]["rationale"]
    assert rows[0]["prior_selection_state"] == "new"
    assert rows[0]["dataset_id"] == "fx-arabic-demo"
    assert rows[0]["dataset_version"] == "v1"
    assert rows[0]["source_metadata"]["document_id"] == "fx_doc_1"
    assert rows[0]["training_eligible"] is True


def test_export_refuses_to_delete_an_existing_destination(tmp_path) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination"):
        results.store.export_run_copy(run_id, destination)

    assert marker.read_text(encoding="utf-8") == "user data"


def test_dataset_registry_is_idempotent_but_rejects_conflicting_identity(
    tmp_path,
) -> None:
    store = ResultsStore(tmp_path / "results")
    original = BenchmarkDataset(
        dataset_id="dataset",
        version="v1",
        name="original",
        created_at="2026-09-10T00:00:00Z",
    ).to_dict()
    conflicting = {**original, "name": "replacement"}

    store.save_dataset(original)
    store.save_dataset(original)
    with pytest.raises(ConflictingRecordError, match="dataset"):
        store.save_dataset(conflicting)

    assert store.load_dataset("dataset", "v1")["name"] == "original"


def test_model_service_is_idempotent_but_rejects_conflicting_metadata(tmp_path) -> None:
    results = ResultsService(tmp_path / "results")
    first = ModelRecord(
        model_id="org/model",
        revision="v1",
        display_name="Stable model",
        registered_at="2026-09-10T00:00:00Z",
    )
    repeated = ModelRecord(
        model_id="org/model",
        revision="v1",
        display_name="Stable model",
        registered_at="2026-09-10T00:01:00Z",
    )

    stored = results.register_model(first)
    assert results.register_model(repeated) == stored
    with pytest.raises(ConflictingRecordError, match="model"):
        results.register_model(
            ModelRecord(model_id="org/model", revision="v2", display_name="Changed")
        )


def test_prediction_listing_honors_dataset_filter(tmp_path) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")

    assert (
        len(results.list_predictions(run_id=run_id, dataset_id="fx-arabic-demo")) == 4
    )
    assert results.list_predictions(run_id=run_id, dataset_id="other-dataset") == []


def test_metric_store_rejects_a_different_value_for_the_same_identity(
    tmp_path,
) -> None:
    results = ResultsService(tmp_path / "results")
    run = results.create_run(
        model_id="model", dataset_id="dataset", dataset_version="v1", created_at="t"
    )
    store = results.store
    page = build_fixture_pages(dataset_id="dataset")[0]
    results.add_page(run.run_id, page)
    original = EvaluationRecord(
        record_id="metric-id",
        run_id=run.run_id,
        page_id=page.page_id,
        metric_name="cer@raw",
        value=0.25,
        computed_at="2026-09-10T00:00:00Z",
    )
    conflicting = EvaluationRecord(
        record_id="metric-id",
        run_id=run.run_id,
        page_id=page.page_id,
        metric_name="cer@raw",
        value=0.75,
        computed_at="2026-09-10T00:00:01Z",
    )

    store.append_metrics(run.run_id, [original])
    store.append_metrics(run.run_id, [original])
    with pytest.raises(ConflictingRecordError, match="metric"):
        store.append_metrics(run.run_id, [conflicting])

    wrong_run = EvaluationRecord(
        record_id="wrong-run-metric",
        run_id="other-run",
        page_id="page-2",
        metric_name="cer@raw",
        value=0.5,
    )
    with pytest.raises(ValueError, match="run id"):
        store.append_metrics(run.run_id, [wrong_run])

    assert list(store.iter_metrics(run.run_id))[0].value == 0.25


def test_run_updates_cannot_rebind_a_run_to_another_model(tmp_path) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")
    payload = results.store.load_run_metadata(run_id)

    with pytest.raises(ConflictingRecordError, match="immutable run identity"):
        results.store.save_run_metadata({**payload, "model_id": "other-model"})

    assert results.store.load_run_metadata(run_id)["model_id"] == "fake-model-a"


def test_lab_metrics_match_canonical_results_metric_policies(tmp_path) -> None:
    results, run_id = _create_stored_run(tmp_path / "results")
    page_id = "fx_page_002"
    ground_truth = results.get_ground_truth(run_id, page_id)
    prediction = results.get_predictions(run_id, page_id)[0]
    raw = compute_page_metrics(ground_truth, prediction, normalization="raw")
    normalized = compute_page_metrics(
        ground_truth,
        prediction,
        normalization="comparison_arabic_fold_digits",
    )

    analysis = StoredResultsAnalysisService(results).analyze_page(run_id, page_id)

    assert analysis.cer == raw["cer"]
    assert analysis.wer == raw["wer"]
    assert analysis.normalized_cer == normalized["cer"]
