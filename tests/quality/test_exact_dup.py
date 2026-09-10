"""Tests for :mod:`clouda_data.quality.exact_dup`."""

from __future__ import annotations

import hashlib
import random

import pytest

from clouda_data.pretraining.schema import DatasetSample
from clouda_data.quality.exact_dup import (
    RAW_TEXT_HASH_DOMAIN,
    classify_exact_duplicates,
    raw_text_sha256,
)
from tests.quality.conftest import sha256_bytes


def _sample(
    sample_id: str,
    *,
    file_sha256: str | None = None,
    raw_text: str | None = None,
    **overrides: object,
) -> DatasetSample:
    kwargs: dict[str, object] = {"sample_id": sample_id, "source_id": "src1"}
    if file_sha256 is not None:
        kwargs["file_sha256"] = file_sha256
    if raw_text is not None:
        kwargs["raw_text"] = raw_text
    kwargs.update(overrides)  # type: ignore[arg-type]
    return DatasetSample(**kwargs)  # type: ignore[arg-type]


def _states(samples: list[DatasetSample]) -> dict[str, str]:
    return {s.sample_id: s.duplicate_state.value for s in samples}


def _by_id(samples: list[DatasetSample]) -> dict[str, DatasetSample]:
    return {s.sample_id: s for s in samples}


def test_byte_identical_images_are_duplicates_with_link() -> None:
    sha = sha256_bytes(b"same-png-bytes")
    samples = [
        _sample("smp_a", file_sha256=sha),
        _sample("smp_b", file_sha256=sha),
        _sample("smp_c", file_sha256=sha256_bytes(b"other-png-bytes")),
    ]
    result, report = classify_exact_duplicates(samples)
    states = _states(result)
    by_id = _by_id(result)

    assert states["smp_c"] == "unique"
    assert sorted(s for s, v in states.items() if v == "canonical") == ["smp_a"]
    assert sorted(s for s, v in states.items() if v == "duplicate") == ["smp_b"]
    assert by_id["smp_b"].duplicate_of == "smp_a"
    assert report.duplicate_file_hash == 1
    assert report.conflicting_duplicate == 0
    counts = report.to_dict()["counts"]
    assert counts["duplicate_file_hash"] == 1
    assert counts["unique"] == 1


def test_same_text_different_images_conflicting_both_kept() -> None:
    text = "نص عربي متطابق تمامًا بين العينتين"
    samples = [
        _sample("smp_x", file_sha256=sha256_bytes(b"image-one"), raw_text=text),
        _sample("smp_y", file_sha256=sha256_bytes(b"image-two"), raw_text=text),
    ]
    result, report = classify_exact_duplicates(samples)
    states = _states(result)

    # Pretraining semantics: conflicting-duplicate canonicals stay UNIQUE
    # (both samples are kept as real data; only the twin is marked).
    assert set(states.values()) == {"unique", "conflicting_duplicate"}
    assert states["smp_x"] != states["smp_y"]
    assert report.conflicting_duplicate == 1
    # Both samples are kept in the output (never deleted).
    assert len(result) == 2


def test_same_text_same_image_hash_stays_duplicate() -> None:
    sha = sha256_bytes(b"identical-artifact")
    samples = [
        _sample("smp_a", file_sha256=sha, raw_text="نص واحد"),
        _sample("smp_b", file_sha256=sha, raw_text="نص واحد"),
    ]
    result, report = classify_exact_duplicates(samples)
    states = _states(result)

    assert set(states.values()) == {"canonical", "duplicate"}
    assert report.conflicting_duplicate == 0
    assert report.duplicate_file_hash == 1


def test_raw_text_hash_domain_separated() -> None:
    text = "بعض النص"
    expected = hashlib.sha256(
        RAW_TEXT_HASH_DOMAIN + b"\x00" + text.encode()
    ).hexdigest()
    assert raw_text_sha256(text) == expected
    # Domain separation: must differ from a plain sha256 of the text.
    assert raw_text_sha256(text) != hashlib.sha256(text.encode()).hexdigest()
    assert (
        raw_text_sha256(text)
        != hashlib.sha256(RAW_TEXT_HASH_DOMAIN + text.encode()).hexdigest()
    )


def test_confirmed_near_pair_unions_with_near_reason() -> None:
    samples = [
        _sample("smp_a", file_sha256=sha256_bytes(b"img-a")),
        _sample("smp_b", file_sha256=sha256_bytes(b"img-b-near")),
        _sample("smp_z", file_sha256=sha256_bytes(b"img-unrelated")),
    ]
    result, report = classify_exact_duplicates(samples, [("smp_b", "smp_a")])
    states = _states(result)
    by_id = _by_id(result)

    assert states["smp_a"] == "canonical"
    assert states["smp_b"] == "duplicate"
    assert by_id["smp_b"].duplicate_of == "smp_a"
    assert by_id["smp_b"].exclusion_reason == "near_duplicate_image"
    assert states["smp_z"] == "unique"
    assert report.near_duplicate_image == 1
    counts = report.to_dict()["counts"]
    assert counts["near_duplicate_image"] == 1
    assert counts["unique"] == 1


def test_deterministic_across_shuffled_input_order() -> None:
    samples = [
        _sample("smp_a", file_sha256=sha256_bytes(b"d0")),
        _sample("smp_a2", file_sha256=sha256_bytes(b"d0"), raw_text="same words"),
        _sample("smp_b", file_sha256=sha256_bytes(b"d1"), raw_text="same words"),
        _sample("smp_c", file_sha256=sha256_bytes(b"d2")),
        _sample("smp_d", file_sha256=sha256_bytes(b"d3")),
    ]
    pairs = [("smp_c", "smp_d")]

    baseline: tuple[list[dict[str, object]], dict[str, object]] | None = None
    for seed in range(6):
        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)
        result, report = classify_exact_duplicates(shuffled, pairs)
        payload = ([s.to_dict() for s in result], report.to_dict())
        snapshot: tuple[list[dict[str, object]], dict[str, object]] = payload
        if baseline is None:
            baseline = snapshot
        else:
            assert snapshot == baseline
    assert baseline is not None


@pytest.mark.parametrize(
    "pair",
    [
        ("smp_b", "smp_a"),
        ("smp_a", "smp_b"),
    ],
)
def test_near_pair_order_irrelevant(pair: tuple[str, str]) -> None:
    samples = [
        _sample("smp_a", file_sha256=sha256_bytes(b"p1")),
        _sample("smp_b", file_sha256=sha256_bytes(b"p2")),
    ]
    result, report = classify_exact_duplicates(samples, [pair])
    states = _states(result)
    assert states["smp_a"] == "canonical"
    assert states["smp_b"] == "duplicate"
    assert report.near_duplicate_image == 1


def test_report_schema_version() -> None:
    _result, report = classify_exact_duplicates([_sample("smp_only")])
    payload = report.to_dict()
    assert payload["schema_version"] == "clouda.quality.dedupe.v2"
    assert set(payload["counts"]) == {
        "unique",
        "duplicate_file_hash",
        "duplicate_sample_id",
        "duplicate_source_record",
        "conflicting_duplicate",
        "near_duplicate_image",
    }
