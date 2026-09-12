from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_serve_defaults_to_loopback_and_rejects_public_bindings():
    from clouda_lab.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000

    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--host", "0.0.0.0"])


def test_serve_builds_standalone_app_without_reload(monkeypatch, tmp_path):
    from clouda_lab.cli import _cmd_serve, build_parser

    called = {}
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda app, **kwargs: called.update(app=app, **kwargs)),
    )
    args = build_parser().parse_args(
        ["serve", "--repo-root", str(tmp_path), "--port", "8123"]
    )

    assert _cmd_serve(args) == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8123
    assert called["reload"] is False
    assert called["app"].title == "Clouda Lab"
