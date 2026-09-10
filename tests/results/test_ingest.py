"""Ingestion adapter tests against real repository formats + provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_data.results.ingest import (
    benchmark_manifest_to_pages,
    foundation_record_to_canonical,
    iter_benchmark_manifest,
    ocr_arabic_results_to_runs,
    training_run_to_model_record,
)
from clouda_data.results.metrics import compute_page_metrics
from clouda_data.results.models import GroundTruthRecord, OCRPrediction

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_MANIFEST = REPO_ROOT / "benchmarks/ocr_arabic/benchmark_manifest.jsonl"
RESULTS_CSV = REPO_ROOT / "benchmarks/ocr_arabic/results.csv"
CANONICAL_MANIFEST_SHA256 = (
    "2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893"
)


class TestBenchmarkManifestAdapter:
    def test_converts_all_177_rows(self) -> None:
        pages = benchmark_manifest_to_pages(BENCHMARK_MANIFEST)
        assert len(pages) == 177
        assert len({page.page_id for page in pages}) == 177

    def test_preserves_page_identity_and_hashes(self) -> None:
        pages = benchmark_manifest_to_pages(BENCHMARK_MANIFEST)
        page = next(p for p in pages if p.document_id == "arabic_ocr_000026")
        assert page.document_id == "arabic_ocr_000026"  # source identity kept
        assert page.page_id.startswith("dist_")  # unique per-page key
        assert page.ground_truth_sha256 is not None
        assert len(page.ground_truth_sha256) == 64

    def test_portable_uris_only(self) -> None:
        pages = benchmark_manifest_to_pages(BENCHMARK_MANIFEST)
        for page in pages:
            uri = page.image_artifact.uri
            assert uri.startswith("dataset://")
            assert not Path(uri).is_absolute()
            assert "F:" not in uri and "\\" not in uri

    def test_distortion_metadata_preserved(self) -> None:
        rows = list(iter_benchmark_manifest(BENCHMARK_MANIFEST))
        pages = benchmark_manifest_to_pages(BENCHMARK_MANIFEST)
        profile_rows = [row for row in rows if row.get("kind") == "profile"]
        profile_pages = [p for p in pages if p.profile is not None]
        assert len(profile_pages) == len(profile_rows) == 36
        combined = profile_pages[0]
        assert combined.distortions, "profile rows keep their step list"

    def test_provenance_survives(self) -> None:
        pages = benchmark_manifest_to_pages(BENCHMARK_MANIFEST)
        provenance = pages[0].provenance
        assert provenance.source_format == "benchmarks.ocr_arabic.benchmark_manifest.v1"
        assert provenance.source_sha256 is None or len(provenance.source_sha256) == 64
        assert provenance.adapter_version == "clouda.results.ingest.v1"

    def test_protection_markers_survive(self) -> None:
        rows = list(iter_benchmark_manifest(BENCHMARK_MANIFEST))
        # Synthetic check: inject a holdout split marker into one row.
        rows[0]["source"]["source_split"] = "holdout"
        from clouda_data.results.ingest import benchmark_manifest_row_to_page

        page = benchmark_manifest_row_to_page(rows[0])
        assert page.protection.protected is True
        assert page.is_training_eligible is False


class TestResultsCsvAdapter:
    def test_converts_all_rows(self) -> None:
        runs = ocr_arabic_results_to_runs(
            RESULTS_CSV, manifest_sha256=CANONICAL_MANIFEST_SHA256
        )
        assert len(runs) == 8  # 6 complete + 1 partial + 1 failed smoke

    def test_status_mapping(self) -> None:
        runs = ocr_arabic_results_to_runs(RESULTS_CSV)
        by_model = {run.model_id: run for run in runs}
        assert by_model["HunyuanOCR-1.5"].status.value == "COMPLETED"
        assert by_model["dots.mocr"].status.value == "INTERRUPTED"
        assert by_model["PaddleOCR-VL-1.6"].status.value == "FAILED"

    def test_deterministic_run_ids(self) -> None:
        first = ocr_arabic_results_to_runs(RESULTS_CSV)
        second = ocr_arabic_results_to_runs(RESULTS_CSV)
        assert [run.run_id for run in first] == [run.run_id for run in second]

    def test_models_metadata_only(self) -> None:
        models: list = []
        ocr_arabic_results_to_runs(RESULTS_CSV, model_records_out=models)
        assert len(models) == 8
        assert all(not model.metadata.get("weights") for model in models)

    def test_run_links_manifest_hash(self) -> None:
        runs = ocr_arabic_results_to_runs(
            RESULTS_CSV, manifest_sha256=CANONICAL_MANIFEST_SHA256
        )
        assert all(run.manifest_sha256 == CANONICAL_MANIFEST_SHA256 for run in runs)


class TestFoundationAdapter:
    def test_minimal_row(self) -> None:
        row = {
            "page_id": "pg1",
            "reference_text": "نص",
            "prediction_text": "نص",
        }
        page, gt, prediction, metrics = foundation_record_to_canonical(
            row, run_id="run1"
        )
        assert page.page_id == "pg1"
        assert gt.raw_text == "نص"
        assert prediction.text == "نص"
        assert len(metrics) == 4

    def test_holdout_row_stays_protected(self) -> None:
        row = {
            "page_id": "pg2",
            "reference_text": "نص",
            "prediction_text": "نص",
            "split": "holdout",
        }
        page, gt, _, _ = foundation_record_to_canonical(row, run_id="run1")
        assert page.protection.protected
        assert not page.is_training_eligible
        assert gt.protection.protected

    def test_generated_page_id_fallback(self) -> None:
        row = {
            "generated_page_id": "gen_9",
            "reference_text": "",
            "prediction_text": "",
        }
        page, _, _, _ = foundation_record_to_canonical(row, run_id="run1")
        assert page.page_id == "gen_9"

    def test_malformed_row_rejected(self) -> None:
        with pytest.raises(ValueError):
            foundation_record_to_canonical({"reference_text": "x"}, run_id="run1")


class TestTrainingLineageAdapter:
    def test_model_record_from_training_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "exp1" / "runX"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": "runX",
                    "experiment_name": "exp1",
                    "config_hash": "c" * 64,
                    "dataset_manifest_hash": "d" * 64,
                    "model": {"model_id": "clouda-tiny", "revision": "ck9"},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps({"best_metrics": {"cer": 0.1}}), encoding="utf-8"
        )
        record = training_run_to_model_record(run_dir)
        assert record.training_lineage is not None
        assert record.training_lineage.experiment_name == "exp1"
        assert record.training_lineage.training_run_id == "runX"
        assert record.training_lineage.config_hash == "c" * 64
        assert record.model_id == "clouda-tiny"
        assert record.metadata["best_metrics"] == {"cer": 0.1}

    def test_missing_metadata_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            training_run_to_model_record(tmp_path)


class TestMetricConsistencyWithFoundation:
    def test_matches_foundation_cer(self) -> None:
        from clouda_data.evaluation.cer import cer as foundation_cer
        from clouda_data.evaluation.normalization import normalize_ocr_text

        gt = GroundTruthRecord(
            page_id="p",
            raw_text="كِتَاب",
            raw_text_sha256="0" * 64,
        )
        prediction = OCRPrediction(
            prediction_id="x",
            run_id="r",
            page_id="p",
            model_id="m",
            model_revision="rev",
            text="كتاب",
            text_sha256="0" * 64,
        )
        raw = compute_page_metrics(gt, prediction, normalization="raw")
        normalized = compute_page_metrics(
            gt, prediction, normalization="comparison_arabic_fold_digits"
        )
        assert raw["cer"] == foundation_cer("كِتَاب", "كتاب")
        assert normalized["cer"] == foundation_cer(
            normalize_ocr_text("كِتَاب"), normalize_ocr_text("كتاب")
        )
        # Diacritics fold away under the comparison policy.
        assert normalized["cer"] == 0.0
        assert raw["cer"] > 0.0

    def test_unknown_normalization_rejected(self) -> None:
        gt = GroundTruthRecord(page_id="p", raw_text="س", raw_text_sha256="0" * 64)
        prediction = OCRPrediction(
            prediction_id="x",
            run_id="r",
            page_id="p",
            model_id="m",
            model_revision="rev",
            text="س",
            text_sha256="0" * 64,
        )
        with pytest.raises(ValueError):
            compute_page_metrics(gt, prediction, normalization="nope")
