"""Canonical Ground Truth access.

Rules (from ``docs/data_foundation/GROUND_TRUTH_POLICY.md``):
- original text is immutable; raw and normalized views are distinct;
- no silent normalization during ingestion;
- SHA-256 proves the reference text did not change;
- protected pages stay protected (fail closed).
"""

from __future__ import annotations

from typing import Any

from clouda_data.ground_truth.normalization import normalize_for_comparison

from .identity import sha256_text
from .models import GroundTruthRecord, PageRecord, ProtectionInfo


def normalized_view(
    record: GroundTruthRecord,
    *,
    fold_digits: bool = False,
) -> str:
    """Explicitly derive a normalized comparison view (never overwrites raw)."""

    return normalize_for_comparison(record.raw_text, fold_digits=fold_digits)


def verify_ground_truth(record: GroundTruthRecord) -> bool:
    """Return True when the stored text still matches its recorded hash."""

    return sha256_text(record.raw_text) == record.raw_text_sha256


def build_ground_truth_record(
    *,
    page: PageRecord,
    raw_text: str,
    source_uri: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GroundTruthRecord:
    """Build a canonical GT record from a page and its exact raw text.

    The text is stored verbatim — this function never normalizes, strips, or
    re-encodes the input. Protection state is inherited from the page and
    cannot be relaxed here.
    """

    protection = page.protection or ProtectionInfo()
    return GroundTruthRecord(
        page_id=page.page_id,
        raw_text=raw_text,
        raw_text_sha256=sha256_text(raw_text),
        dataset_id=page.dataset_id,
        split=page.split,
        provenance=(
            page.provenance
            if source_uri is None
            else (
                page.provenance
                if page.provenance is None
                else type(page.provenance)(
                    source_format=page.provenance.source_format,
                    source_uri=source_uri,
                    source_sha256=page.provenance.source_sha256,
                    license_or_permission=page.provenance.license_or_permission,
                    adapter_version=page.provenance.adapter_version,
                    ingested_at=page.provenance.ingested_at,
                    extra=page.provenance.extra,
                    source_private=page.provenance.source_private,
                )
            )
        ),
        protection=protection,
        metadata=dict(metadata or {}),
    )
