"""Store behavior: duplicates, conflicts, integrity, queries, read-only."""

from __future__ import annotations

import pytest

from clouda_data.results.ground_truth import build_ground_truth_record
from clouda_data.results.models import (
    InferenceRunStatus,
    OCRPrediction,
    PageRecord,
)
from clouda_data.results.service import NewPrediction, ResultsService
from clouda_data.results.store import (
    ConflictingRecordError,
    ResultsStore,
    UnknownRecordError,
)
from tests.results.fixtures_e2e import (
    build_fixture_ground_truth,
    build_fixture_pages,
    build_fixture_predictions,
)


@pytest.fixture()
def populated(tmp_path):
    svc = ResultsService(tmp_path / "store")
    pages = build_fixture_pages(dataset_id="fx")
    run = svc.create_run(
        model_id="fake-model-a",
        model_revision="1.0",
        dataset_id="fx",
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
    return svc, run.run_id


class TestDuplicateHandling:
    def test_identical_page_reingestion_is_idempotent(self, populated) -> None:
        svc, run_id = populated
        pages = build_fixture_pages(dataset_id="fx")
        assert svc.add_pages(run_id, pages) == 0  # nothing new appended

    def test_conflicting_page_rejected(self, populated) -> None:
        svc, run_id = populated
        pages = build_fixture_pages(dataset_id="fx")
        conflicting = [pages[0].__class__(**{**pages[0].to_dict(), "split": "test"})]
        with pytest.raises(ConflictingRecordError):
            svc.add_pages(run_id, conflicting)

    def test_conflicting_prediction_rejected(self, populated) -> None:
        svc, run_id = populated
        with pytest.raises(ConflictingRecordError):
            svc.add_prediction(
                run_id,
                NewPrediction(
                    page_id="fx_page_001", text="different", model_id="fake-model-a"
                ),
            )

    def test_conflicting_ground_truth_rejected(self, populated) -> None:
        svc, run_id = populated
        page = svc.get_page(run_id, "fx_page_001")
        gt = build_ground_truth_record(page=page, raw_text="مختلف")
        with pytest.raises(ConflictingRecordError):
            svc.store.append_ground_truth(run_id, [gt])


class TestQueries:
    def test_get_page(self, populated) -> None:
        svc, run_id = populated
        page = svc.get_page(run_id, "fx_page_001")
        assert page.document_id == "fx_doc_1"

    def test_get_page_missing(self, populated) -> None:
        svc, run_id = populated
        with pytest.raises(UnknownRecordError):
            svc.get_page(run_id, "nope")

    def test_get_ground_truth(self, populated) -> None:
        svc, run_id = populated
        text = svc.get_ground_truth_text(run_id, "fx_page_001")
        assert "القَاهِرَة" in text  # raw form with diacritics preserved

    def test_list_pages_filters(self, populated) -> None:
        svc, run_id = populated
        assert len(svc.list_pages(run_id, split="train")) == 2
        assert len(svc.list_pages(run_id, split="holdout")) == 1
        assert len(svc.list_pages(run_id, profile="clean")) == 1
        assert len(svc.list_pages(run_id, document_id="fx_doc_2")) == 2

    def test_list_runs_filters(self, populated) -> None:
        svc, run_id = populated
        assert svc.list_runs(model_id="fake-model-a")
        assert not svc.list_runs(model_id="other")
        assert svc.list_runs(status="COMPLETED")

    def test_predictions_by_page_and_model(self, populated) -> None:
        svc, run_id = populated
        preds = svc.get_predictions(run_id, "fx_page_001")
        assert len(preds) == 1
        assert preds[0].model_id == "fake-model-a"
        empty = svc.get_predictions(run_id, "fx_page_001", model_id="nope")
        assert empty == []

    def test_worst_pages_sorted_descending(self, populated) -> None:
        svc, run_id = populated
        rows = svc.get_worst_pages(run_id, limit=4)
        values = [row["value"] for row in rows]
        assert values == sorted(values, reverse=True)
        # The empty prediction (page 003, model B) is not in this run, but
        # page 002's digit/tatweel differences should rank above zero-CER pages.
        assert rows[0]["value"] >= rows[-1]["value"]

    def test_metrics_scope_filter(self, populated) -> None:
        svc, run_id = populated
        page_metrics = svc.get_metrics(run_id, scope="page")
        summary_metrics = svc.get_metrics(run_id, scope="run_summary")
        assert page_metrics and summary_metrics
        assert all(record.page_id for record in page_metrics)
        assert all(record.page_id is None for record in summary_metrics)


class TestReadOnly:
    def test_read_only_refuses_writes(self, tmp_path) -> None:
        store = ResultsStore(tmp_path / "ro", read_only=True)
        with pytest.raises(PermissionError):
            store.save_model({"model_id": "m"})

    def test_read_only_allows_reads(self, populated) -> None:
        svc, run_id = populated
        root = svc.store.root
        ro = ResultsStore(root, read_only=True)
        assert ro.load_run_metadata(run_id)["run_id"] == run_id


class TestIntegrity:
    def test_verify_ok(self, populated) -> None:
        svc, run_id = populated
        report = svc.verify(run_id)
        assert report["ok"], report["issues"]

    def test_verify_detects_unknown_run(self, populated) -> None:
        svc, _ = populated
        with pytest.raises(UnknownRecordError):
            svc.verify("run_does_not_exist")

    def test_verify_detects_corrupted_gt(self, populated, tmp_path) -> None:
        svc, run_id = populated
        gt_path = svc.store.ground_truth_path(run_id)
        lines = gt_path.read_text(encoding="utf-8").splitlines()
        import json

        row = json.loads(lines[0])
        row["raw_text"] = row["raw_text"] + "تمرير"
        lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
        gt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = svc.verify(run_id)
        assert not report["ok"]
        assert any("ground truth hash mismatch" in issue for issue in report["issues"])

    def test_verify_detects_unknown_page_reference(self, populated) -> None:
        svc, run_id = populated
        from clouda_data.results.identity import prediction_identity
        from clouda_data.ground_truth.checksums import sha256_text

        ghost = OCRPrediction(
            prediction_id=prediction_identity(run_id=run_id, page_id="ghost"),
            run_id=run_id,
            page_id="ghost",
            model_id="fake-model-a",
            model_revision="1.0",
            text="x",
            text_sha256=sha256_text("x"),
            dataset_id="fx",
        )
        with pytest.raises(ValueError, match="unknown page"):
            svc.store.append_predictions(run_id, [ghost])

    def test_verify_detects_prediction_run_mismatch(self, populated) -> None:
        svc, run_id = populated
        from clouda_data.results.identity import prediction_identity
        from clouda_data.ground_truth.checksums import sha256_text

        foreign = OCRPrediction(
            prediction_id=prediction_identity(run_id="other", page_id="fx_page_001"),
            run_id="other",
            page_id="fx_page_001",
            model_id="m",
            model_revision="r",
            text="x",
            text_sha256=sha256_text("x"),
        )
        with pytest.raises(ValueError):
            svc.store.append_predictions(run_id, [foreign])

    def test_write_boundaries_require_a_registered_matching_run(
        self, populated
    ) -> None:
        svc, run_id = populated
        page = build_fixture_pages(dataset_id="fx")[0]
        with pytest.raises(UnknownRecordError, match="Unknown run"):
            svc.store.append_pages("unknown-run", [page])
        assert not svc.store.run_dir("unknown-run").exists()

        wrong_page = PageRecord.from_dict({**page.to_dict(), "dataset_id": "other"})
        with pytest.raises(ValueError, match="dataset"):
            svc.add_page(run_id, wrong_page)
        with pytest.raises(ValueError, match="model"):
            svc.add_prediction(
                run_id,
                NewPrediction(page_id=page.page_id, text="x", model_id="other"),
            )

    def test_verify_detects_cross_record_identity_corruption(self, populated) -> None:
        svc, run_id = populated
        path = svc.store.predictions_path(run_id)
        import json

        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["model_id"] = "foreign-model"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

        report = svc.verify(run_id)
        assert not report["ok"]
        assert any("prediction model mismatch" in issue for issue in report["issues"])


class TestIndexAndExport:
    def test_index_rebuildable(self, populated) -> None:
        svc, run_id = populated
        import shutil

        index_dir = svc.store.rebuild_index(run_id)
        assert (index_dir / "predictions_by_page.json").exists()
        shutil.rmtree(index_dir)
        svc.store.rebuild_index(run_id)
        assert (index_dir / "predictions_by_page.json").exists()

    def test_export_copy_is_portable(self, populated, tmp_path) -> None:
        svc, run_id = populated
        target = tmp_path / "export"
        svc.store.export_run_copy(run_id, target)
        exported = ResultsStore(target, read_only=True)
        assert exported.load_run_metadata(run_id)["run_id"] == run_id

    def test_exported_bundle_has_no_absolute_paths(self, populated, tmp_path) -> None:
        svc, run_id = populated
        target = tmp_path / "export2"
        svc.store.export_run_copy(run_id, target)
        for path in target.rglob("*.json*"):
            text = path.read_text(encoding="utf-8")
            assert "F:\\" not in text
            assert str(svc.store.root) not in text

    def test_export_refuses_destination_inside_source_run(self, populated) -> None:
        svc, run_id = populated
        nested = svc.store.run_dir(run_id) / "nested-export"

        with pytest.raises(ValueError, match="overlap"):
            svc.store.export_run_copy(run_id, nested)

        assert not nested.exists()

    def test_dataset_path_matches_saved_registry_record(self, tmp_path) -> None:
        store = ResultsStore(tmp_path / "store")
        payload = {"dataset_id": "dataset", "version": "v1"}

        assert store.save_dataset(payload) == store.dataset_path("dataset", "v1")


class TestRunLifecycle:
    def test_status_transitions(self, tmp_path) -> None:
        svc = ResultsService(tmp_path / "store")
        run = svc.create_run(
            model_id="m", dataset_id="d", created_at="t", manifest_sha256=None
        )
        assert run.status is InferenceRunStatus.CREATED
        svc.mark_run(run.run_id, InferenceRunStatus.RUNNING)
        payload = svc.get_run(run.run_id)
        assert payload["status"] == "RUNNING"
        assert payload["started_at"]
        svc.mark_run(run.run_id, InferenceRunStatus.FAILED)
        payload = svc.get_run(run.run_id)
        assert payload["status"] == "FAILED"
        assert payload["ended_at"]

    def test_unknown_run(self, tmp_path) -> None:
        svc = ResultsService(tmp_path / "store", read_only=True)
        with pytest.raises(UnknownRecordError):
            svc.get_run("missing")
