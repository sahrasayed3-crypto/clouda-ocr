"""Render backends: weasyprint (System B) and raqm (System A)."""

from __future__ import annotations

from .base import RenderBackend, RenderResult

__all__ = ["RenderBackend", "RenderResult", "get_backend"]

_BACKENDS: dict[str, type] = {}


def _load():
    from .weasyprint_backend import WeasyPrintBackend
    from .raqm_page_backend import RqmPageBackend

    _BACKENDS["weasyprint"] = WeasyPrintBackend
    _BACKENDS["raqm"] = RqmPageBackend


def get_backend(name: str, **kwargs) -> RenderBackend:
    if not _BACKENDS:
        _load()
    if name not in _BACKENDS:
        raise ValueError(f"unknown render backend '{name}' (have: {sorted(_BACKENDS)})")
    return _BACKENDS[name](**kwargs)


def available_backends() -> list[str]:
    if not _BACKENDS:
        _load()
    return sorted(_BACKENDS)
