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
        "Training Runs",
        "Results Store",
        "Benchmarks",
        "Doctor",
        "Hardware",
        "Offline Status",
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
    ):
        assert f'"{route}"' in script
    for phrase in (
        "Protected Holdout",
        "Training Allowed",
        "Model weights not installed",
        "No compatible GPU is currently available",
        "Start Real Training",
        "Resume Blocked",
        "Run Quality Check",
        "Analyze Duplicates",
        "Create Derived Dataset",
    ):
        assert phrase in script
    assert "innerHTML" not in script
    assert "https://" not in script
    assert "http://" not in script
    assert '"data-runtime-loading": "true"' in script
    assert 'querySelector("[data-runtime-loading]")?.remove()' in script
    assert "new AbortController()" in script
    assert "requestController?.abort()" in script


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
