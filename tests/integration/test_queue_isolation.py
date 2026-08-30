from __future__ import annotations

from pathlib import Path

from clouda_contracts.queues import (
    DATASET_PIPELINE_QUEUE,
    MODEL_EVALUATION_QUEUE,
    PDF_CONVERSION_QUEUE,
    QUEUE_POLICIES,
    TRAINING_PREPARATION_QUEUE,
    WORKER_CAPABILITIES,
    capability_report,
)

ROOT = Path(__file__).resolve().parents[2]


def test_required_queues_have_independent_policies() -> None:
    assert set(QUEUE_POLICIES) == {
        PDF_CONVERSION_QUEUE,
        DATASET_PIPELINE_QUEUE,
        TRAINING_PREPARATION_QUEUE,
        MODEL_EVALUATION_QUEUE,
    }
    assert (
        len(
            {
                (policy.timeout_seconds, policy.retry_count)
                for policy in QUEUE_POLICIES.values()
            }
        )
        == 4
    )


def test_runtime_worker_cannot_prepare_datasets_or_train() -> None:
    runtime = WORKER_CAPABILITIES["runtime"]
    assert runtime.queues == (PDF_CONVERSION_QUEUE,)
    assert "dataset" not in runtime.allowed_storage_roots
    assert runtime.may_train is False


def test_capability_report_is_serializable() -> None:
    report = capability_report("dataset")
    assert report["queues"] == [DATASET_PIPELINE_QUEUE]
    assert report["policies"][0]["storage_capability"] == "dataset"


def test_request_entrypoints_do_not_import_training_subsystem() -> None:
    for relative in ("app.py", "pdfword/worker_api.py"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "clouda_training" not in content
