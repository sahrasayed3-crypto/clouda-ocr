"""Near-duplicate candidate indexing for the dataset quality gate (Agent E).

Two stages over the per-image fingerprints produced by
:mod:`clouda_data.quality.image_fp`:

``build_fingerprints``
    Safe-load every sample image exactly once (via
    :func:`image_fp.fingerprint_path`) and project the result into a
    :class:`~clouda_data.quality.models.SampleFingerprint`. Blankish pages
    are fingerprinted but excluded from the index (recorded in the run
    summary); undecodable images are recorded as decode errors.

``candidate_pairs``
    O(N) LSH banding over the 192-bit concatenation
    ``phash || dhash || ahash`` (12 bands x 16 bits) with a same
    aspect-bucket requirement and a per-bucket overflow guard
    (``max_bucket`` from :class:`ImageFingerprintPolicy`): a bucket larger
    than the cap is skipped entirely and flagged in the summary instead of
    degrading to pairwise comparison. An intra-document adjacent-page pass
    (same ``document_id``, ``page_index`` differing by 1, no bucket
    requirement) is applied on top when page metadata is supplied.

Per-pair Hamming distances are verified only for emitted candidates
(O(candidates), never O(N^2)). Levels follow the Agent F triple
conjunction: CONFIRMED iff ``d_p <= 8 and d_d <= 10 and d_a <= 10``,
LIKELY iff ``d_p <= 12``, else CANDIDATE.

v1 is in-memory only (fine to ~10k samples); the optional SQLite-backed
index described in DESIGN_DECISIONS (B8 schema) is deliberately deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from clouda_data.pretraining.schema import DatasetSample
from clouda_data.quality.config import QualityGateConfig
from clouda_data.quality.image_fp import (
    CONFIRMED_A_MAX,
    CONFIRMED_D_MAX,
    CONFIRMED_P_MAX,
    FINGERPRINT_VERSION,
    fingerprint_path_cached,
)
from clouda_data.quality.models import (
    NEAR_DUPLICATE_LEVELS,
    NearDuplicateCandidate,
    SampleFingerprint,
)

# Level thresholds: the CONFIRMED triple conjunction lives in image_fp
# (Agent D owns those constants). The LIKELY pHash ceiling is defined here.
LIKELY_P_MAX = 12

LEVEL_CONFIRMED = "CONFIRMED_NEAR_DUPLICATE"
LEVEL_LIKELY = "LIKELY_DUPLICATE"
LEVEL_CANDIDATE = "CANDIDATE"
assert (LEVEL_CANDIDATE, LEVEL_LIKELY, LEVEL_CONFIRMED) == NEAR_DUPLICATE_LEVELS

#: LSH geometry over the 192-bit concatenation.
LSH_BANDS = 12
LSH_BAND_BITS = 16
LSH_TOTAL_BITS = LSH_BANDS * LSH_BAND_BITS  # 192

#: Default overflow cap (mirrors ImageFingerprintPolicy.max_bucket default).
DEFAULT_MAX_BUCKET = 4096


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FingerprintStageSummary:
    """Bookkeeping for :func:`build_fingerprints` (IDs and codes only)."""

    fingerprinted: int = 0
    skipped_blankish: tuple[str, ...] = ()
    decode_errors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FingerprintStage:
    """Fingerprints plus their stage summary."""

    fingerprints: tuple[SampleFingerprint, ...]
    summary: FingerprintStageSummary


def build_fingerprints_with_summary(
    samples: Iterable[DatasetSample],
    root: str | Path,
    config: QualityGateConfig,
) -> FingerprintStage:
    """Fingerprint every sample with an image, recording skips and errors.

    Blankish pages are fingerprinted but kept out of the returned index
    (their IDs land in ``summary.skipped_blankish``); undecodable images are
    recorded in ``summary.decode_errors``. Deterministic: input order does
    not affect the output (sorted by sample_id).
    """

    del config  # fingerprint policy is fully owned by image_fp in v1
    base = Path(root)
    fingerprints: list[SampleFingerprint] = []
    skipped: list[str] = []
    errors: dict[str, str] = {}
    count = 0
    for sample in sorted(samples, key=lambda s: s.sample_id):
        if not sample.image_path:
            continue
        path = Path(sample.image_path)
        if not path.is_absolute():
            path = base / path
        try:
            result = fingerprint_path_cached(path)
        except Exception as exc:  # decode/IO failure -> recorded, never fatal
            errors[sample.sample_id] = f"{type(exc).__name__}: {exc}"
            continue
        count += 1
        if result.blankish:
            skipped.append(sample.sample_id)
            continue
        fingerprints.append(
            SampleFingerprint(
                sample_id=sample.sample_id,
                source_id=sample.source_id,
                ahash=result.ahash,
                dhash=result.dhash,
                phash=result.phash,
                aspect_bucket=result.aspect_bucket,
                blankish=False,
                blank_stats=dict(result.blank_stats),
                fingerprint_version=result.fingerprint_version or FINGERPRINT_VERSION,
            )
        )
    return FingerprintStage(
        fingerprints=tuple(fingerprints),
        summary=FingerprintStageSummary(
            fingerprinted=count,
            skipped_blankish=tuple(skipped),
            decode_errors=dict(errors),
        ),
    )


def build_fingerprints(
    samples: Iterable[DatasetSample],
    root: str | Path,
    config: QualityGateConfig,
) -> list[SampleFingerprint]:
    """Index fingerprints for all non-blankish samples (see *_with_summary)."""

    return list(build_fingerprints_with_summary(samples, root, config).fingerprints)


def page_index_meta(
    samples: Iterable[DatasetSample],
) -> dict[str, tuple[str, int]]:
    """Map ``sample_id -> (document_id, page_index)`` for the adjacent pass."""

    meta: dict[str, tuple[str, int]] = {}
    for sample in samples:
        if sample.document_id is None or sample.page_index is None:
            continue
        meta[sample.sample_id] = (sample.document_id, int(sample.page_index))
    return meta


# ---------------------------------------------------------------------------
# Candidate pairs
# ---------------------------------------------------------------------------


def _concat_int(fp: SampleFingerprint) -> int:
    """192-bit concat ``phash || dhash || ahash`` as a single integer."""

    return (int(fp.phash, 16) << 128) | (int(fp.dhash, 16) << 64) | int(fp.ahash, 16)


def _band_key(concat: int, band: int) -> int:
    return (concat >> (band * LSH_BAND_BITS)) & ((1 << LSH_BAND_BITS) - 1)


def _classify(d_p: int, d_d: int, d_a: int) -> str:
    if d_p <= CONFIRMED_P_MAX and d_d <= CONFIRMED_D_MAX and d_a <= CONFIRMED_A_MAX:
        return LEVEL_CONFIRMED
    if d_p <= LIKELY_P_MAX:
        return LEVEL_LIKELY
    return LEVEL_CANDIDATE


@dataclass(frozen=True)
class CandidateStageSummary:
    """Counts + overflow flags for :func:`candidate_pairs_with_summary`."""

    indexed: int = 0
    overflowed_buckets: tuple[str, ...] = ()
    overflowed_entries: int = 0
    pairs_before_dedupe: int = 0
    adjacent_pairs: int = 0


@dataclass(frozen=True)
class CandidateStage:
    """Candidates plus their stage summary."""

    candidates: tuple[NearDuplicateCandidate, ...]
    summary: CandidateStageSummary


def candidate_pairs_with_summary(
    fingerprints: Iterable[SampleFingerprint],
    config: QualityGateConfig,
    page_meta: Mapping[str, tuple[str, int]] | None = None,
) -> CandidateStage:
    """LSH-banded near-duplicate candidates with a run summary.

    Deterministic: output is sorted by ``(sample_id_a, sample_id_b)`` and
    buckets are iterated in sorted key order. Buckets exceeding
    ``config.image_fingerprint.max_bucket`` entries are skipped and flagged
    (``summary.overflowed_buckets``) rather than exploding into pairs.
    """

    fps = [fp for fp in fingerprints if not fp.blankish]
    fps = sorted(fps, key=lambda f: f.sample_id)
    max_bucket = config.image_fingerprint.max_bucket
    concat_by_id = {fp.sample_id: _concat_int(fp) for fp in fps}
    fps_by_id = {fp.sample_id: fp for fp in fps}

    pair_ids: set[tuple[str, str]] = set()
    overflowed: list[tuple[str, int]] = []
    pairs_before = 0
    for band in range(LSH_BANDS):
        buckets: dict[tuple[int, int, str], list[str]] = {}
        for fp in fps:
            key = (band, _band_key(concat_by_id[fp.sample_id], band), fp.aspect_bucket)
            buckets.setdefault(key, []).append(fp.sample_id)
        for key in sorted(buckets):
            members = buckets[key]
            # >= : a bucket AT the cap can still emit ~8M pairs (R3-H2).
            if len(members) >= max_bucket:
                overflowed.append(
                    (f"band_{key[0]}:bucket_{key[1]:04x}:{key[2]}", len(members))
                )
                continue
            ordered_members = sorted(members)
            for i, sid_a in enumerate(ordered_members):
                for sid_b in ordered_members[i + 1 :]:
                    pair_ids.add((sid_a, sid_b))
                    pairs_before += 1

    adjacent_pairs = 0
    if page_meta:
        by_doc: dict[str, list[tuple[int, str]]] = {}
        for sid, (doc_id, page_index) in sorted(page_meta.items()):
            if sid in fps_by_id:
                by_doc.setdefault(doc_id, []).append((page_index, sid))
        for doc_id in sorted(by_doc):
            entries = sorted(by_doc[doc_id])
            for (idx_a, sid_a), (idx_b, sid_b) in zip(entries, entries[1:]):
                if idx_b - idx_a != 1:
                    continue
                pair = (min(sid_a, sid_b), max(sid_a, sid_b))
                if pair not in pair_ids:
                    adjacent_pairs += 1
                pair_ids.add(pair)

    candidates: list[NearDuplicateCandidate] = []
    for sid_a, sid_b in sorted(pair_ids):
        fp_a = fps_by_id[sid_a]
        fp_b = fps_by_id[sid_b]
        d_p = (int(fp_a.phash, 16) ^ int(fp_b.phash, 16)).bit_count()
        d_d = (int(fp_a.dhash, 16) ^ int(fp_b.dhash, 16)).bit_count()
        d_a = (int(fp_a.ahash, 16) ^ int(fp_b.ahash, 16)).bit_count()
        candidates.append(
            NearDuplicateCandidate(
                sample_id_a=sid_a,
                sample_id_b=sid_b,
                level=_classify(d_p, d_d, d_a),
                distances={"d_p": d_p, "d_d": d_d, "d_a": d_a},
            )
        )
    overflowed.sort()
    return CandidateStage(
        candidates=tuple(candidates),
        summary=CandidateStageSummary(
            indexed=len(fps),
            overflowed_buckets=tuple(key for key, _ in overflowed),
            overflowed_entries=sum(n for _, n in overflowed),
            pairs_before_dedupe=pairs_before,
            adjacent_pairs=adjacent_pairs,
        ),
    )


def candidate_pairs(
    fingerprints: Iterable[SampleFingerprint],
    config: QualityGateConfig,
    page_meta: Mapping[str, tuple[str, int]] | None = None,
) -> list[NearDuplicateCandidate]:
    """Near-duplicate candidates (see :func:`candidate_pairs_with_summary`)."""

    return list(
        candidate_pairs_with_summary(fingerprints, config, page_meta).candidates
    )


__all__ = [
    "DEFAULT_MAX_BUCKET",
    "LIKELY_P_MAX",
    "LEVEL_CANDIDATE",
    "LEVEL_CONFIRMED",
    "LEVEL_LIKELY",
    "LSH_BAND_BITS",
    "LSH_BANDS",
    "LSH_TOTAL_BITS",
    "CandidateStage",
    "CandidateStageSummary",
    "FingerprintStage",
    "FingerprintStageSummary",
    "build_fingerprints",
    "build_fingerprints_with_summary",
    "candidate_pairs",
    "candidate_pairs_with_summary",
    "page_index_meta",
]
