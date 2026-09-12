from __future__ import annotations

import csv
import json
from pathlib import Path

from .test_catalog import _catalog


def _services(tmp_path: Path):
    from clouda_lab.dashboard.observability import ObservabilityService
    from clouda_lab.dashboard.training import TrainingService

    catalog = _catalog(tmp_path)
    training = TrainingService(catalog.settings, catalog)
    return catalog, training, ObservabilityService(catalog.settings, catalog, training)


def test_results_and_benchmarks_read_only_canonical_local_metadata(tmp_path: Path):
    from clouda_data.results.service import ResultsService

    catalog, _training, service = _services(tmp_path)
    writer = ResultsService(catalog.settings.results_root)
    writer.register_dataset(dataset_id="bench-set", version="1", name="Benchmark")
    writer.register_model({"model_id": "model-a", "display_name": "Model A"})
    run = writer.create_run(
        model_id="model-a",
        dataset_id="bench-set",
        created_at="2026-09-12T12:00:00+00:00",
        metadata={"experiment_id": "exp-a", "benchmark_id": "arabic-v1"},
    )

    bench = catalog.settings.benchmarks_root / "ocr_arabic"
    bench.mkdir(parents=True)
    (bench / "release.json").write_text(
        json.dumps({"benchmark_id": "arabic-v1", "manifest_sha256": "a" * 64}),
        encoding="utf-8",
    )
    with (bench / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=["model", "status", "cer", "wer", "rankable", "gpu"],
        )
        writer_csv.writeheader()
        writer_csv.writerow(
            {
                "model": "Model A",
                "status": "COMPLETE",
                "cer": "0.1",
                "wer": "0.2",
                "rankable": "true",
                "gpu": "fixture",
            }
        )
        writer_csv.writerow(
            {
                "model": "Model B",
                "status": "PARTIAL",
                "cer": "",
                "wer": "",
                "rankable": "false",
                "gpu": "fixture",
            }
        )

    results = service.results(
        {
            "model": "model-a",
            "experiment": "exp-a",
            "benchmark": "arabic-v1",
            "date": "2026-09-12",
        }
    )
    assert [item["run_id"] for item in results["runs"]] == [run.run_id]
    assert results["datasets"][0]["dataset_id"] == "bench-set"
    assert str(tmp_path) not in repr(results)

    detail = service.result_detail(run.run_id, metric_limit=10)
    assert detail["run"]["run_id"] == run.run_id
    assert detail["summary"]["status"] == "NOT AVAILABLE"
    assert detail["metrics"] == []
    assert detail["integrity"]["valid"] is True

    benchmarks = service.benchmarks({})
    assert benchmarks["manifest"]["benchmark_id"] == "arabic-v1"
    assert [row["status"] for row in benchmarks["results"]] == [
        "COMPLETE",
        "PARTIAL",
    ]
    assert benchmarks["results"][1]["rankable"] is False


def test_doctor_hardware_and_overview_preserve_canonical_status(tmp_path: Path):
    _catalog_service, _training, service = _services(tmp_path)

    latest = service.latest_doctor()
    assert latest["status"] == "NOT RUN"

    report = service.run_doctor(deep=False)
    assert report["overall_status"] in {"PASS", "WARN", "FAIL", "INFO", "SKIP"}
    assert str(tmp_path) not in repr(report)

    hardware = service.hardware()
    assert hardware["gpu"]["available"] is False
    assert hardware["gpu"]["status"] in {"UNAVAILABLE", "DEFERRED"}
    assert hardware["logical_validation_separate"] is True

    overview = service.overview()
    assert overview["datasets"] == 2
    assert overview["gpu"] in {"UNAVAILABLE", "AVAILABLE", "DEFERRED"}
    assert overview["offline"] == "ACTIVE"
    assert overview["doctor"] == report["overall_status"]


def test_overview_hardware_probe_does_not_implicitly_run_full_doctor(tmp_path: Path):
    _catalog_service, _training, service = _services(tmp_path)

    overview = service.overview()

    assert overview["doctor"] == "NOT RUN"
    assert service.latest_doctor()["status"] == "NOT RUN"
