"""Structured failure buckets.

Assigns samples to failure buckets using **only measured metrics and
metadata actually present** — visual/document properties that are not in the
metadata are never inferred.

Bucket assignment is deterministic: a sample may match multiple buckets
(e.g. high CER *and* digit-heavy); all matching buckets are returned, with
``primary`` computed from the first rule in the fixed rule order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import FailureBucket

BUCKET_RULES: tuple[tuple[str, str], ...] = (
    # (bucket, description) — order defines primary-bucket precedence.
    ("high_cer", "CER at or above the high-CER threshold"),
    ("high_wer", "WER at or above the high-WER threshold"),
    ("whitespace_heavy", "whitespace error rate dominates"),
    ("digit_heavy", "digit error categories dominate"),
    ("punctuation_heavy", "punctuation error categories dominate"),
    ("diacritics_heavy", "diacritic error categories dominate"),
    ("deletion_heavy", "deletions dominate the error mix"),
    ("insertion_heavy", "insertions dominate the error mix"),
    ("severe_substitution", "substitutions dominate the error mix"),
    ("distorted_blur", "metadata reports blur-type distortion"),
    ("distorted_skew", "metadata reports skew/rotation distortion"),
    ("distorted_compression", "metadata reports compression artifacts"),
    ("mixed_arabic_english", "metadata marks mixed Arabic/English content"),
    ("small_text", "metadata marks small text (when present)"),
    ("table_form", "metadata marks table/form document (when present)"),
    ("unknown", "no rule matched"),
)

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "high_cer": 0.4,
    "high_wer": 0.5,
    "dominant_share": 0.4,  # share of errors a category needs to dominate
}

_BLUR_DISTORTIONS = frozenset(
    {"gaussian_blur", "motion_blur", "low_resolution", "defocus"}
)
_SKEW_DISTORTIONS = frozenset({"rotation", "skew", "perspective"})
_COMPRESSION_DISTORTIONS = frozenset({"jpeg_compression", "low_contrast"})


def bucket_sample(
    *,
    cer: float = 0.0,
    wer: float = 0.0,
    error_type_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> list[FailureBucket]:
    """Return all matching buckets for one sample (deterministic order)."""
    resolved = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        resolved.update(thresholds)
    counts = dict(error_type_counts or {})
    meta = dict(metadata or {})
    evidence: dict[str, Any] = {"cer": cer, "wer": wer}
    buckets: list[FailureBucket] = []

    total_errors = sum(counts.values())

    def dominant(*categories: str) -> bool:
        if total_errors == 0:
            return False
        share = sum(counts.get(c, 0) for c in categories) / total_errors
        return share >= resolved["dominant_share"]

    if cer >= resolved["high_cer"]:
        buckets.append(
            FailureBucket("high_cer", "", {"cer": cer})
        )
    if wer >= resolved["high_wer"]:
        buckets.append(
            FailureBucket("high_wer", "", {"wer": wer})
        )
    if dominant("whitespace"):
        buckets.append(
            FailureBucket(
                "whitespace_heavy", "", {"whitespace": counts.get("whitespace", 0)}
            )
        )
    if dominant(
        "arabic_digit", "latin_digit", "digit_system_confusion", "digit_word"
    ):
        buckets.append(
            FailureBucket(
                "digit_heavy",
                "",
                {
                    "digit_errors": sum(
                        counts.get(c, 0)
                        for c in (
                            "arabic_digit",
                            "latin_digit",
                            "digit_system_confusion",
                            "digit_word",
                        )
                    )
                },
            )
        )
    if dominant("punctuation"):
        buckets.append(
            FailureBucket("punctuation_heavy", "", {"punctuation": counts.get("punctuation", 0)})
        )
    if dominant("diacritic", "diacritics_heavy"):
        buckets.append(
            FailureBucket("diacritics_heavy", "", {"diacritic": counts.get("diacritic", 0)})
        )
    if dominant("missing_arabic_character", "missing_latin_character", "missing_character", "missing_word"):
        buckets.append(
            FailureBucket(
                "deletion_heavy",
                "",
                {"deletions": sum(counts.get(c, 0) for c in (
                    "missing_arabic_character", "missing_latin_character",
                    "missing_character", "missing_word"))},
            )
        )
    if dominant("extra_arabic_character", "extra_latin_character", "extra_character", "extra_word"):
        buckets.append(
            FailureBucket(
                "insertion_heavy",
                "",
                {"insertions": sum(counts.get(c, 0) for c in (
                    "extra_arabic_character", "extra_latin_character",
                    "extra_character", "extra_word"))},
            )
        )
    if dominant(
        "character_substitution", "hamza_alef_variant", "ya_alef_maqsura",
        "ta_marbuta_ha", "arabic_latin_script", "word_substitution",
    ):
        buckets.append(
            FailureBucket("severe_substitution", "", {"substitution_share": True})
        )

    distortion = _distortion_values(meta)
    if distortion & _BLUR_DISTORTIONS:
        buckets.append(
            FailureBucket("distorted_blur", "", {"distortion": sorted(distortion & _BLUR_DISTORTIONS)})
        )
    if distortion & _SKEW_DISTORTIONS:
        buckets.append(
            FailureBucket("distorted_skew", "", {"distortion": sorted(distortion & _SKEW_DISTORTIONS)})
        )
    if distortion & _COMPRESSION_DISTORTIONS:
        buckets.append(
            FailureBucket(
                "distorted_compression",
                "",
                {"distortion": sorted(distortion & _COMPRESSION_DISTORTIONS)},
            )
        )

    text_blob = json_text(meta)
    if "mixed" in text_blob or meta.get("language") in ("ar+en", "ar-en", "mixed"):
        buckets.append(FailureBucket("mixed_arabic_english", "", {"language": meta.get("language", "mixed")}))
    if meta.get("small_text") is True or meta.get("text_size") == "small":
        buckets.append(FailureBucket("small_text", "", {"small_text": True}))
    if str(meta.get("document_type", "")).casefold() in {"table", "form", "invoice", "receipt"}:
        buckets.append(
            FailureBucket("table_form", "", {"document_type": meta.get("document_type")})
        )

    if not buckets:
        buckets.append(FailureBucket("unknown", "", dict(evidence)))
    return buckets


def _distortion_values(meta: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("distortion", "distortions"):
        value = meta.get(key)
        if isinstance(value, str):
            values.add(value)
        elif isinstance(value, (list, tuple)):
            values.update(str(item) for item in value)
    provenance = meta.get("provenance")
    if isinstance(provenance, Mapping):
        transform = provenance.get("transform_steps")
        if isinstance(transform, Sequence):
            for step in transform:
                if isinstance(step, Mapping) and step.get("distortion"):
                    values.add(str(step["distortion"]))
    profile = meta.get("profile")
    if isinstance(profile, str):
        values.add(f"profile:{profile}")
    return {v.replace("profile:", "") if v.startswith("profile:") else v for v in values}


def json_text(meta: Mapping[str, Any]) -> str:
    return str(meta.get("content_type", "")).casefold()


def primary_bucket(buckets: Sequence[FailureBucket]) -> str:
    """First matching bucket per the fixed rule order, else 'unknown'."""
    return buckets[0].bucket if buckets else "unknown"


def assign_buckets(
    samples: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, list[FailureBucket]]:
    """Assign buckets to many samples.

    Each sample mapping needs ``sample_id`` and any of ``cer``/``wer``/
    ``error_type_counts``/``metadata``.
    """
    result: dict[str, list[FailureBucket]] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        result[sample_id] = bucket_sample(
            cer=float(sample.get("cer", 0.0) or 0.0),
            wer=float(sample.get("wer", 0.0) or 0.0),
            error_type_counts=sample.get("error_type_counts"),
            metadata=sample.get("metadata"),
            thresholds=thresholds,
        )
    return result


__all__ = [
    "BUCKET_RULES",
    "assign_buckets",
    "bucket_sample",
    "primary_bucket",
]
