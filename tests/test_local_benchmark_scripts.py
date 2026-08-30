from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path

from pdfword.docx_export import markdown_to_docx
from pdfword.models import PageResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "benchmarks" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calculate_metrics import (
    calculate_text_metrics,
    quality_gate_decision,
)  # noqa: E402
from validate_docx import validate_docx_bytes  # noqa: E402


def test_benchmark_quality_gate_uses_real_accuracy_boundaries() -> None:
    assert quality_gate_decision(
        character_accuracy=89.99,
        has_ground_truth=True,
        estimated_quality=99.0,
    ) == ("retry", "below_90_percent_accuracy")
    assert quality_gate_decision(
        character_accuracy=90.0,
        has_ground_truth=True,
        estimated_quality=None,
    ) == ("accepted", None)
    assert quality_gate_decision(
        character_accuracy=90.01,
        has_ground_truth=True,
        estimated_quality=10.0,
    ) == ("accepted", None)


def test_benchmark_quality_gate_rejects_unmeasured_or_exhausted_local_routes() -> None:
    assert quality_gate_decision(
        character_accuracy=None,
        has_ground_truth=True,
        estimated_quality=100.0,
    ) == ("manual_review", "missing_ground_truth_or_unmeasured_accuracy")
    assert quality_gate_decision(
        character_accuracy=70.0,
        has_ground_truth=True,
        estimated_quality=100.0,
        local_routes_exhausted=True,
    ) == ("manual_review", "local_routes_exhausted")


def test_benchmark_metrics_keep_punctuation_and_report_edit_operations() -> None:
    metrics = calculate_text_metrics("abc, مرحبا", "axc مرحبا")
    assert metrics["cer"] > 0
    assert metrics["wer"] > 0
    assert metrics["char_substitutions"] >= 1
    assert metrics["char_deletions"] >= 1
    assert metrics["reference_characters"] == len("abc, مرحبا")


def test_benchmark_docx_validation_rejects_media_and_tables_and_accepts_text_only() -> (
    None
):
    payload = markdown_to_docx(
        [
            PageResult(
                page_no=1,
                model_used="local:pypdf",
                markdown="مرحبا بالعالم\nEnglish text",
                accepted=True,
                requires_manual_review=False,
            ),
            PageResult(
                page_no=2,
                model_used="local:pypdf",
                markdown="Second page",
                accepted=True,
                requires_manual_review=False,
            ),
        ]
    )
    validation = validate_docx_bytes(payload, expected_pages=2)
    assert validation["valid"] is True
    assert validation["has_page_break"] is True
    assert validation["has_rtl"] is True
    assert validation["table_count"] == 0
    assert validation["media_count"] == 0
