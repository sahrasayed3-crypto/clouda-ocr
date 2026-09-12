"""Bridge from quality-gate runs to the results store.

``persist_quality_summary`` writes one quality-run summary into any store that
exposes ``save_summary(run_id, summary)`` (duck-typed via
:class:`SummaryStoreProtocol`, so tests can pass a fake). The store schema is
never changed: the summary is a plain JSON dict whose ``metadata`` block marks
it as a dataset-quality run. Stores lacking ``save_summary`` raise
:class:`ResultsBridgeError` instead of failing with an ``AttributeError``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class ResultsBridgeError(RuntimeError):
    """Raised when a store cannot accept a quality-run summary."""


@runtime_checkable
class SummaryStoreProtocol(Protocol):
    """Narrow structural view of the results store used by this bridge."""

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> Any:
        """Persist ``summary`` for ``run_id``; return value is ignored."""
        ...


def quality_summary(
    run_id: str,
    summary: dict[str, Any],
    *,
    config_identity: str,
    verdict: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shape the summary payload with the dataset-quality metadata block."""

    extra = dict(metadata or {})
    merged = {
        "kind": "dataset_quality_run",
        "config_identity": config_identity,
        "verdict": verdict,
    }
    merged.update(extra)
    payload = dict(summary)
    payload["run_id"] = run_id
    payload["metadata"] = merged
    return payload


def persist_quality_summary(
    store: SummaryStoreProtocol,
    run_id: str,
    summary: dict[str, Any],
    *,
    config_identity: str = "",
    verdict: str = "",
) -> None:
    """Persist ``summary`` for ``run_id`` via ``store.save_summary``.

    ``summary`` should already carry its metadata block (see
    :func:`quality_summary`); this function validates the store capability and
    delegates. No store schema is created or changed.
    """

    save_summary = getattr(store, "save_summary", None)
    if not callable(save_summary):
        raise ResultsBridgeError(
            "Results store does not support save_summary(); "
            f"got store of type {type(store).__name__!r}."
        )
    save_summary(run_id, summary)
