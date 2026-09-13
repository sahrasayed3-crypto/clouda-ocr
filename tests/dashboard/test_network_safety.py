from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).parents[2] / "clouda_lab" / "dashboard" / "static"


def test_navigation_has_no_remote_urls_or_automatic_mutation():
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert "https://" not in script + html
    assert "http://" not in script + html
    assert "WebSocket" not in script
    assert "EventSource" not in script
    assert "navigator.sendBeacon" not in script
    assert "route().then" not in script
    assert '"Automatic downloads": data.network_policy?.automatic_downloads' in script
    assert "fetch(`/api/lab${path}`" in script
    assert "window.confirm" in script


def test_every_operational_action_targets_a_closed_lab_api_route():
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    for endpoint in (
        "/dataset-downloads",
        "/model-removals",
        "/benchmark-plans",
        "/training/${encodeURIComponent(select.value)}/start-plan",
        "/training-starts",
        "/runs/${encoded}/resume",
    ):
        assert endpoint in script
    for forbidden in ("/shell", "/exec", "/files", "/install", "/model-download"):
        assert forbidden not in script
