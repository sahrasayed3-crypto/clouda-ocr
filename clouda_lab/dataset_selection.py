"""Dataset Selection Engine.

Selects pages/samples from canonical pre-training manifests into reproducible,
provenance-preserving derived subsets. Every selection:

- filters through the holdout guard (protected rows can never be selected);
- is deterministic for a given (criteria, seed) pair;
- records the source manifest identity (path + SHA-256), selection criteria,
  seed, timestamp, and lineage in the derived manifest header;
- writes derived manifests with the canonical
  ``clouda_data.pretraining.manifest.write_manifest`` writer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from clouda_contracts.checksums import sha256_file
from clouda_data.pretraining.manifest import read_manifest, write_manifest

from .holdout_guard import filter_protected_rows, row_is_protected

SELECTION_SCHEMA_VERSION = "clouda.lab.selection.v1"


@dataclass(frozen=True)
class SelectionCriteria:
    """Declarative selection query. ``None`` means "no constraint".

    Deterministic random sampling, top-N hardest/easiest and percentile
    windows require per-sample scores; pass ``scores`` (sample_id -> float,
    higher = harder) to :func:`select_samples` for those modes.
    """

    sample_ids: tuple[str, ...] | None = None
    dataset_id: str | None = None
    split: str | None = None
    document_type: str | None = None
    profile: str | None = None
    distortion: str | None = None
    error_type: str | None = None
    cer_min: float | None = None
    cer_max: float | None = None
    wer_min: float | None = None
    wer_max: float | None = None
    model_id: str | None = None
    run_id: str | None = None
    failure_bucket: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    random_sample_size: int | None = None
    top_n: str | None = None  # "hardest" | "easiest"
    top_n_count: int | None = None
    percentile_range: tuple[float, float] | None = None  # e.g. (80, 100) = hardest 20%
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if value is None or value == {} or value == ():
                continue
            if isinstance(value, tuple):
                payload[key] = list(value)
            else:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class SelectionResult:
    sample_ids: tuple[str, ...]
    excluded_protected: int
    criteria: dict[str, Any]
    seed: int
    created_utc: str
    selection_id: str
    source_manifest: str
    source_manifest_sha256: str
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "sample_ids": list(self.sample_ids),
            "excluded_protected": self.excluded_protected,
            "criteria": dict(self.criteria),
            "seed": self.seed,
            "created_utc": self.created_utc,
            "source_manifest": self.source_manifest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "schema_version": SELECTION_SCHEMA_VERSION,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _row_matches(row: dict[str, Any], criteria: SelectionCriteria) -> bool:
    def get(*fields: str) -> Any:
        for name in fields:
            if name in row and row[name] is not None:
                return row[name]
        return None

    if (
        criteria.dataset_id is not None
        and str(get("dataset_id")) != criteria.dataset_id
    ):
        return False
    if criteria.split is not None:
        # ``target_split`` is the assignment field; ``split``/``source_split``
        # are checked only when no explicit target_split exists (e.g. the
        # header-level split convention). Never treat differing fallback
        # fields as conflicting — first present wins.
        split_value = row.get("target_split")
        if split_value is None:
            split_value = get("split", "source_split")
        if (
            split_value is None
            or str(split_value).strip().casefold() != criteria.split.strip().casefold()
        ):
            return False
    if criteria.document_type is not None:
        candidates = [
            get("document_type"),
            (
                (row.get("metadata") or {}).get("document_type")
                if isinstance(row.get("metadata"), dict)
                else None
            ),
            (
                (row.get("provenance") or {}).get("document_type")
                if isinstance(row.get("provenance"), dict)
                else None
            ),
        ]
        if not any(
            value is not None and str(value) == criteria.document_type
            for value in candidates
        ):
            return False
    if criteria.profile is not None:
        candidates = [
            get("profile", "profile_id"),
            (
                (row.get("provenance") or {}).get("profile")
                if isinstance(row.get("provenance"), dict)
                else None
            ),
        ]
        if not any(
            value is not None and str(value) == criteria.profile for value in candidates
        ):
            return False
    if criteria.distortion is not None:
        candidates = [get("distortion")]
        provenance = row.get("provenance")
        if isinstance(provenance, dict):
            transform = provenance.get("transform_steps") or provenance.get(
                "distortions"
            )
            if isinstance(transform, list):
                candidates.extend(
                    step.get("distortion")
                    for step in transform
                    if isinstance(step, dict)
                )
        if not any(
            value is not None and str(value) == criteria.distortion
            for value in candidates
        ):
            return False
    if criteria.source is not None:
        candidates = [get("source_id", "source"), get("source_document_id")]
        if not any(
            value is not None and str(value) == criteria.source for value in candidates
        ):
            return False
    if criteria.tags:
        # Tags may live at row level or inside the metadata block.
        row_tags = row.get("tags")
        if row_tags is None and isinstance(row.get("metadata"), dict):
            row_tags = row["metadata"].get("tags")
        if isinstance(row_tags, str):
            row_tags = {row_tags}
        if not isinstance(row_tags, (list, tuple, set, frozenset)):
            return False
        if not set(criteria.tags).issubset({str(tag) for tag in row_tags}):
            return False
    if criteria.metadata:
        row_metadata = row.get("metadata")
        if not isinstance(row_metadata, dict):
            return False
        for key, value in criteria.metadata.items():
            if row_metadata.get(key) != value:
                return False
    return True


def _score_matches(
    row: dict[str, Any],
    criteria: SelectionCriteria,
    scores: dict[str, float] | None,
    sample_id: str,
) -> bool:
    """Filter by error type / CER / WER / model / run / failure bucket.

    These criteria can be satisfied either from inline row metrics
    (``cer``/``wer``/``model_id``/... present in analysis manifests) or from
    the supplied ``scores`` map (sample_id -> CER).
    """
    if criteria.error_type is not None:
        row_errors = row.get("error_types")
        if isinstance(row_errors, dict):
            if criteria.error_type not in row_errors:
                return False
        elif row.get("error_type") is not None:
            if str(row["error_type"]) != criteria.error_type:
                return False
        elif scores is None:
            # Cannot verify error_type without analysis data.
            return False
    if criteria.cer_min is not None or criteria.cer_max is not None:
        cer_value = _numeric(row, "cer", scores, sample_id)
        if cer_value is None:
            return False
        if criteria.cer_min is not None and cer_value < criteria.cer_min:
            return False
        if criteria.cer_max is not None and cer_value > criteria.cer_max:
            return False
    if criteria.wer_min is not None or criteria.wer_max is not None:
        wer_value = _numeric(row, "wer", None, sample_id)
        if wer_value is None:
            return False
        if criteria.wer_min is not None and wer_value < criteria.wer_min:
            return False
        if criteria.wer_max is not None and wer_value > criteria.wer_max:
            return False
    if criteria.model_id is not None:
        value = row.get("model_id")
        if value is None or str(value) != criteria.model_id:
            return False
    if criteria.run_id is not None:
        value = row.get("run_id")
        if value is None or str(value) != criteria.run_id:
            return False
    if criteria.failure_bucket is not None:
        value = row.get("failure_bucket")
        if value is None or str(value) != criteria.failure_bucket:
            return False
    return True


def _numeric(
    row: dict[str, Any],
    key: str,
    scores: dict[str, float] | None,
    sample_id: str,
) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if scores is not None and sample_id in scores:
        return float(scores[sample_id])
    return None


def _deterministic_rank(sample_id: str, seed: int) -> tuple:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return (digest, sample_id)


def _apply_sampling(
    rows: list[dict[str, Any]],
    criteria: SelectionCriteria,
    scores: dict[str, float] | None,
    seed: int,
) -> list[dict[str, Any]]:
    if criteria.random_sample_size is not None:
        ranked = sorted(
            rows, key=lambda row: _deterministic_rank(str(row.get("sample_id")), seed)
        )
        return ranked[: criteria.random_sample_size]
    if criteria.top_n and scores is not None:
        hardest = criteria.top_n == "hardest"
        # Missing scores sink to the bottom for both directions.
        missing = float("-inf") if hardest else float("inf")
        ranked = sorted(
            rows,
            key=lambda row: (
                scores.get(str(row.get("sample_id")), missing),
                str(row.get("sample_id")),
            ),
            reverse=hardest,
        )
        count = criteria.top_n_count or len(ranked)
        return ranked[:count]
    if criteria.percentile_range is not None and scores is not None:
        low, high = criteria.percentile_range
        scored = sorted(
            (
                (
                    scores.get(str(row.get("sample_id")), 0.0),
                    str(row.get("sample_id")),
                    row,
                )
                for row in rows
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not scored:
            return []
        values = [item[0] for item in scored]
        low_cut = _percentile_value(values, low)
        high_cut = _percentile_value(values, high)
        window = [row for score, _sid, row in scored if low_cut <= score <= high_cut]
        return window
    if criteria.limit is not None:
        return rows[: criteria.limit]
    return rows


def _percentile_value(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return (
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def select_samples(
    manifest_path: str,
    criteria: SelectionCriteria,
    *,
    seed: int = 20260723,
    scores: dict[str, float] | None = None,
) -> SelectionResult:
    """Select rows from a canonical manifest according to ``criteria``.

    Protected rows are never returned and never count toward sampling limits;
    they are tallied in ``excluded_protected``.
    """
    _header, rows = read_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    safe_rows, protected_count = filter_protected_rows(rows)

    matched: list[dict[str, Any]] = []
    if criteria.sample_ids is not None:
        wanted = set(criteria.sample_ids)
        for row in safe_rows:
            if str(row.get("sample_id")) in wanted:
                matched.append(row)
    else:
        for row in safe_rows:
            if _row_matches(row, criteria) and _score_matches(
                row, criteria, scores, str(row.get("sample_id"))
            ):
                matched.append(row)

    sampled = _apply_sampling(matched, criteria, scores, seed)
    selected_ids = tuple(str(row.get("sample_id")) for row in sampled)

    created = _utc_now()
    selection_material = json.dumps(
        {
            "manifest_sha256": manifest_sha,
            "criteria": criteria.to_dict(),
            "seed": seed,
            "sample_ids": selected_ids,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    selection_id = (
        "sel_" + hashlib.sha256(selection_material.encode("utf-8")).hexdigest()[:20]
    )

    return SelectionResult(
        sample_ids=selected_ids,
        excluded_protected=protected_count,
        criteria=criteria.to_dict(),
        seed=seed,
        created_utc=created,
        selection_id=selection_id,
        source_manifest=str(manifest_path),
        source_manifest_sha256=manifest_sha,
        rows=tuple(sampled),
    )


def write_selection_manifest(
    result: SelectionResult,
    output_path: str,
    *,
    lineage: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Write the derived subset manifest; returns its header.

    The header preserves source identity, source manifest hash, criteria,
    seed, timestamp, and lineage so the subset is fully auditable.
    """
    header = {
        "manifest_role": "lab_selection",
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "selection_id": result.selection_id,
        "selection_criteria": result.criteria,
        "selection_seed": result.seed,
        "selection_created_utc": result.created_utc,
        "source_manifest": result.source_manifest,
        "source_manifest_sha256": result.source_manifest_sha256,
        "source_sample_count": len(result.sample_ids),
        "selection_lineage": list(lineage),
    }
    write_manifest(output_path, list(result.rows), metadata=header)
    written_header, _rows = read_manifest(output_path)
    return written_header


def validate_derived_manifest_for_training(manifest_path: str) -> dict[str, Any]:
    """Verify a derived manifest passes the framework's own dataset guard.

    Delegates to ``validate_training_dataset``-equivalent checks by re-reading
    the manifest and applying the lab holdout guard plus split sanity. Returns
    a small report dict; raises ``PermissionError`` on protected content.
    """
    header, rows = read_manifest(manifest_path)
    if row_is_protected(header):
        raise PermissionError("Derived manifest header is marked protected")
    protected = sum(1 for row in rows if row_is_protected(row))
    if protected:
        raise PermissionError(f"Derived manifest contains {protected} protected rows")
    splits = {
        str(row.get("target_split", row.get("split", ""))).strip().casefold()
        for row in rows
    }
    return {
        "rows": len(rows),
        "splits": sorted(s for s in splits if s),
        "header_protected": False,
        "protected_rows": 0,
    }


__all__ = [
    "SELECTION_SCHEMA_VERSION",
    "SelectionCriteria",
    "SelectionResult",
    "select_samples",
    "validate_derived_manifest_for_training",
    "write_selection_manifest",
]
