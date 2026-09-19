from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).parents[2] / "clouda_lab" / "dashboard" / "static"


def test_dashboard_shell_contains_complete_accessible_navigation():
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    for label in (
        "Overview",
        "Datasets",
        "Quality &amp; Dedup",
        "Experiment Planner",
        "Preflight",
        "Model Adapters",
        "Downloads",
        "Tasks",
        "Model Catalog",
        "Benchmark Workspace",
        "Storage",
        "Training Runs",
        "Results Store",
        "Benchmarks",
        "Doctor",
        "Hardware",
        "System &amp; Network Policy",
    ):
        assert label in html
    assert 'aria-label="Primary navigation"' in html
    assert 'aria-live="polite"' in html
    assert 'data-view="loading"' in html
    assert 'data-view="empty"' in html
    assert 'data-view="error"' in html


def test_client_covers_required_pages_and_honest_disabled_actions():
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    for route in (
        "overview",
        "datasets",
        "quality",
        "planner",
        "preflight",
        "models",
        "runs",
        "results",
        "benchmarks",
        "doctor",
        "hardware",
        "offline",
        "downloads",
        "tasks",
        "model-catalog",
        "benchmark-workspace",
        "storage",
    ):
        assert f'"{route}"' in script
    for phrase in (
        "Protected Holdout",
        "Training Allowed",
        "Start Real Training",
        "Resume Blocked",
        "Run Quality Check",
        "Analyze Duplicates",
        "Create Derived Dataset",
        "Confirm Download",
        "Verify Assets",
        "Create Benchmark Plan",
        "Refresh Tasks",
    ):
        assert phrase in script
    assert "innerHTML" not in script
    assert "https://" not in script
    assert "http://" not in script
    assert '"data-runtime-loading": "true"' in script
    assert 'querySelector("[data-runtime-loading]")?.remove()' in script
    assert "new AbortController()" in script
    assert "setTimeout" in script
    assert "window.confirm" in script
    assert "requestController?.abort()" in script
    assert '["Raw Source", "Training Dataset"]' not in script
    assert '"Checkpoint", "Resume"' not in script
    assert 'value: "clouda-lab-plan"' not in script
    assert 'value: "clean-derived"' not in script
    assert '"Model weights not installed"' not in script
    assert 'title: "No compatible GPU is currently available"' not in script
    assert 'text: "GPU unavailable is a valid operating state.' not in script
    assert "No datasets found." in script
    assert "No training runs recorded." in script
    assert "No benchmark result is available for this filter." in script
    assert "No approved model download manifest" in script


def test_styles_define_statuses_tables_lineage_and_responsive_layout():
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    for selector in (
        ".status-pass",
        ".status-warn",
        ".status-fail",
        ".table-wrap",
        ".lineage",
        ".sidebar",
        "@media",
    ):
        assert selector in css


def test_document_intelligence_page_is_explicit_and_categorical_only():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'href="#/document-intelligence"' in html
    assert '"document-intelligence": renderDocumentIntelligence' in script
    assert "rawBody" in script
    assert "formData" not in script
    assert 'init.headers["X-Clouda-Lab-Action"]' in script
    assert 'type: "file"' in script
    assert 'accept: "application/pdf,.pdf"' in script
    assert "Gate verdict" in script
    assert "Reason codes" in script
    assert "OCR pending" in script
    assert "textContent" in script
    assert "accuracy percentage" not in script.lower()
    assert "confidence percentage" not in script.lower()
    assert "quality percentage" not in script.lower()
