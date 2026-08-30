from pathlib import Path
import threading

import pytest
from pypdf import PdfWriter

from pdfword import conversion_service
from pdfword.models import PageResult


def _write_one_page_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


def test_execute_worker_conversion_records_cloud_attempts_and_outputs(
    monkeypatch, tmp_path
):
    pdf_path = tmp_path / "input.pdf"
    docx_path = tmp_path / "output.docx"
    _write_one_page_pdf(pdf_path)

    def fake_process_pdf(**kwargs):
        assert kwargs["page_numbers"] == [1]
        assert kwargs["cloud_attempt_allowed"](0.01) is True
        kwargs["cloud_attempt_callback"](
            {"model": "cloud-model", "engine_name": "openrouter"}
        )
        kwargs["cloud_attempt_allowed"](99.0)
        page = PageResult(
            page_no=1,
            model_used="local:pypdf",
            markdown="bad text",
            quality_score=98.0,
            text_quality_score=97.0,
            layout_quality_score=96.0,
        )
        conversion_service.capture_openrouter_telemetry(lambda event: None)
        return [page], {}

    emitted = []

    def fake_capture(callback):
        class Manager:
            def __enter__(self):
                callback(
                    {
                        "model": "cloud-model",
                        "prompt_tokens": 4,
                        "completion_tokens": 5,
                        "cost": 0.02,
                        "cost_is_estimated": False,
                    }
                )
                emitted.append(True)

            def __exit__(self, *_args):
                return False

        return Manager()

    monkeypatch.setattr(
        conversion_service, "capture_openrouter_telemetry", fake_capture
    )
    monkeypatch.setattr(conversion_service, "process_pdf", fake_process_pdf)
    monkeypatch.setattr(conversion_service, "markdown_to_docx", lambda pages: b"DOCX")

    request = conversion_service.WorkerConversionRequest(
        job_id="job-1",
        pdf_path=pdf_path,
        docx_path=docx_path,
        page_numbers=[1],
        api_key="",
        fast_model="fast",
        accurate_model="accurate",
        settings={
            "max_pdf_pages": 5,
            "file_cost_limit": 1,
            "daily_cost_limit": 1,
            "current_document_cost": 0,
            "daily_cost_spent": 0,
        },
        correction_rules=[
            {
                "id": 1,
                "wrong_text": "bad",
                "correct_text": "good",
                "approved": 1,
                "enabled": 1,
                "confidence": 1.0,
            }
        ],
    )

    result = conversion_service.execute_worker_conversion(request)

    assert emitted
    assert docx_path.read_bytes() == b"DOCX"
    assert result["status"] == "completed"
    assert result["file_type"] == "digital"
    assert result["text_quality_score"] == pytest.approx(97.0)
    assert result["correction_applications"][0]["before"] == "bad"
    assert result["cloud_attempts"][0]["prompt_tokens"] == 4


def test_worker_conversion_cost_guard_rejects_unknown_and_parallel_over_limit(
    monkeypatch, tmp_path
):
    pdf_path = tmp_path / "input.pdf"
    docx_path = tmp_path / "output.docx"
    _write_one_page_pdf(pdf_path)
    decisions = []

    def fake_process_pdf(**kwargs):
        decisions.append(kwargs["cloud_attempt_allowed"](None))
        barrier = threading.Barrier(3)
        results = []

        def reserve():
            barrier.wait()
            results.append(kwargs["cloud_attempt_allowed"](0.6))

        threads = [threading.Thread(target=reserve) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        decisions.extend(results)
        return [
            PageResult(
                page_no=1,
                model_used="local:pypdf",
                markdown="needs review",
                quality_score=88.0,
                text_quality_score=88.0,
                requires_manual_review=True,
                review_reason="cost_limit_reached",
            )
        ], {}

    monkeypatch.setattr(conversion_service, "process_pdf", fake_process_pdf)
    monkeypatch.setattr(conversion_service, "markdown_to_docx", lambda pages: b"DOCX")

    request = conversion_service.WorkerConversionRequest(
        job_id="job-1",
        pdf_path=pdf_path,
        docx_path=docx_path,
        page_numbers=[1],
        api_key="",
        fast_model="fast",
        accurate_model="accurate",
        settings={
            "max_pdf_pages": 5,
            "file_cost_limit": 1,
            "daily_cost_limit": 0,
            "current_document_cost": 0,
            "daily_cost_spent": 0,
        },
    )

    result = conversion_service.execute_worker_conversion(request)

    assert decisions[0] is False
    assert sorted(decisions[1:]) == [False, True]
    assert result["status"] == "manual_review"
    assert result["cost_limit_reached"] is True


def test_worker_conversion_cost_guard_rejects_non_finite_settings(
    monkeypatch, tmp_path
):
    pdf_path = tmp_path / "input.pdf"
    docx_path = tmp_path / "output.docx"
    _write_one_page_pdf(pdf_path)
    decisions = []

    def fake_process_pdf(**kwargs):
        decisions.append(kwargs["cloud_attempt_allowed"](0.01))
        return [
            PageResult(
                page_no=1,
                model_used="local:pypdf",
                markdown="needs review",
                quality_score=88.0,
                text_quality_score=88.0,
                requires_manual_review=True,
                review_reason="cost_limit_reached",
            )
        ], {}

    monkeypatch.setattr(conversion_service, "process_pdf", fake_process_pdf)
    monkeypatch.setattr(conversion_service, "markdown_to_docx", lambda pages: b"DOCX")

    request = conversion_service.WorkerConversionRequest(
        job_id="job-1",
        pdf_path=pdf_path,
        docx_path=docx_path,
        page_numbers=[1],
        api_key="",
        fast_model="fast",
        accurate_model="accurate",
        settings={
            "max_pdf_pages": 5,
            "file_cost_limit": "nan",
            "daily_cost_limit": 1,
            "current_document_cost": 0,
            "daily_cost_spent": 0,
        },
    )

    result = conversion_service.execute_worker_conversion(request)

    assert decisions == [False]
    assert result["status"] == "manual_review"


class FakeDatabase:
    instances: list["FakeDatabase"] = []

    def __init__(self):
        self.updates = []
        self.attempts = []
        self.instances.append(self)

    def update_conversion(self, conversion_id, values):
        self.updates.append((conversion_id, values))

    def list_attempts(self, _conversion_id):
        return []

    def get_conversion(self, _job_id):
        return {"total_cost": 0.0}

    def daily_cost(self):
        return 0.0

    def record_attempt(self, values):
        self.attempts.append(values)
        return len(self.attempts)

    def enabled_correction_rules(self):
        return [{"pattern": "bad", "replacement": "good"}]


def test_execute_conversion_success_and_failure_paths(monkeypatch, tmp_path):
    pdf_path = tmp_path / "input.pdf"
    docx_path = tmp_path / "out.docx"
    job_root = tmp_path / "job"
    job_root.mkdir()
    _write_one_page_pdf(pdf_path)
    FakeDatabase.instances = []

    def fake_process_pdf(**kwargs):
        assert kwargs["cloud_attempt_allowed"](0.01) is True
        assert kwargs["cancellation_check"]() is False
        kwargs["checkpoint_callback"]([])
        return [
            PageResult(
                page_no=1,
                model_used="local:pypdf",
                markdown="bad page",
                quality_score=91.0,
                text_quality_score=92.0,
                layout_quality_score=93.0,
            )
        ], {}

    monkeypatch.setattr(conversion_service, "Database", FakeDatabase)
    monkeypatch.setattr(
        conversion_service, "load_settings", lambda _db: {"max_pdf_pages": 5}
    )
    monkeypatch.setattr(conversion_service, "load_checkpoint", lambda _root: [])
    monkeypatch.setattr(conversion_service, "save_checkpoint", lambda *_args: None)
    monkeypatch.setattr(conversion_service, "process_pdf", fake_process_pdf)
    monkeypatch.setattr(conversion_service, "markdown_to_docx", lambda pages: b"DOCX")

    request = conversion_service.ConversionRequest(
        conversion_id=7,
        job_id="job-7",
        username="user",
        pdf_path=pdf_path,
        job_root=str(job_root),
        docx_path=str(docx_path),
        page_numbers=[1],
        api_key="",
        fast_model="fast",
        accurate_model="accurate",
    )
    conversion_service.execute_conversion(request, cancellation_check=lambda: False)

    db = FakeDatabase.instances[-1]
    assert docx_path.read_bytes() == b"DOCX"
    assert db.attempts[0]["engine_name"] == "pypdf"
    assert db.updates[-1][1]["status"] == "completed"
    assert db.updates[-1][1]["winning_engine"] == "local:pypdf"

    def failing_process_pdf(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(conversion_service, "process_pdf", failing_process_pdf)
    with pytest.raises(RuntimeError):
        conversion_service.execute_conversion(request, cancellation_check=lambda: False)
    failed_db = FakeDatabase.instances[-1]
    assert failed_db.updates[-1][1]["status"] == "failed"
