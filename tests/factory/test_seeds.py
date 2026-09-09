"""Seed derivation: unified v1 + legacy mode vectors and field sensitivity.

Migrated from the standalone clouda-data-factory repository (tests/test_seeds.py);
imports retargeted to the integrated ``clouda_data.factory`` package.
"""

from __future__ import annotations

import json
from pathlib import Path


from clouda_data.factory.seed.derive import derive_seed, page_seed
from clouda_data.factory.seed.legacy import (
    derive_seed_ocr_benchmark,
    page_seed_arabic_scan_factory,
    variant_seed_arabic_scan_factory,
)

FIXTURES = Path(__file__).parent
SOURCE = "40d13d4ed530ab2271b7d16db20ab4209cec0feab7c491e3e038686a4984fbea"


def test_unified_v1_is_deterministic_and_field_sensitive():
    base = derive_seed(
        20260831,
        SOURCE,
        document_id="doc",
        page_index=0,
        variant_index=0,
        profile="05_old_book_medium",
        distortion_stage="paper_degradation",
        severity="medium",
    )
    assert base == derive_seed(
        20260831,
        SOURCE,
        document_id="doc",
        page_index=0,
        variant_index=0,
        profile="05_old_book_medium",
        distortion_stage="paper_degradation",
        severity="medium",
    )
    variants = [
        derive_seed(
            20260831,
            SOURCE,
            document_id="doc",
            page_index=1,
            variant_index=0,
            profile="05_old_book_medium",
            distortion_stage="paper_degradation",
            severity="medium",
        ),
        derive_seed(
            20260831,
            SOURCE,
            document_id="doc",
            page_index=0,
            variant_index=1,
            profile="05_old_book_medium",
            distortion_stage="paper_degradation",
            severity="medium",
        ),
        derive_seed(
            20260831,
            SOURCE,
            document_id="doc",
            page_index=0,
            variant_index=0,
            profile="12_hard_composite",
            distortion_stage="paper_degradation",
            severity="medium",
        ),
        derive_seed(
            20260831,
            SOURCE,
            document_id="doc",
            page_index=0,
            variant_index=0,
            profile="05_old_book_medium",
            distortion_stage="gaussian_blur",
            severity="medium",
        ),
        derive_seed(
            20260831,
            SOURCE,
            document_id="doc",
            page_index=0,
            variant_index=0,
            profile="05_old_book_medium",
            distortion_stage="paper_degradation",
            severity="heavy",
        ),
        derive_seed(
            20260831,
            "0" * 64,
            document_id="doc",
            page_index=0,
            variant_index=0,
            profile="05_old_book_medium",
            distortion_stage="paper_degradation",
            severity="medium",
        ),
    ]
    assert len(set(variants)) == len(variants), "every field must change the seed"


def test_unified_v1_is_63_bit():
    for n in range(50):
        seed = derive_seed(n, SOURCE, document_id="d", page_index=n)
        assert 0 <= seed < (1 << 63)


def test_legacy_arabic_scan_factory_known_vectors():
    # Computed with the original arabic_scan_factory.profiles.variant_seed.
    assert (
        variant_seed_arabic_scan_factory(SOURCE, 0, "05_old_book_medium", 20260831)
        == 3497116117959753694
    )
    assert (
        variant_seed_arabic_scan_factory(SOURCE, 1, "05_old_book_medium", 20260831)
        == 8248077015845791739
    )
    # per-page convention: variant seed + page index
    assert (
        page_seed_arabic_scan_factory(SOURCE, 0, "05_old_book_medium", 3, 20260831)
        == 3497116117959753697
    )


def test_legacy_ocr_benchmark_reproduces_recorded_manifest_seeds():
    """The vendored derivation must reproduce seeds recorded in the REAL
    historical ocr_benchmark manifest (global seed 20260825)."""
    fixture = FIXTURES / "legacy_a_manifest_sample.jsonl"
    rows = [
        json.loads(line)
        for line in fixture.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    assert rows, "fixture manifest sample must not be empty"
    for row in rows:
        derived = derive_seed_ocr_benchmark(
            20260825,
            row["distorted_id"],
            distortion=row.get("distortion", row.get("profile", "")),
            severity=row["severity"],
        )
        assert derived == row["seed"], f"seed mismatch for {row['distorted_id']}"


def test_modes_do_not_collide():
    a = derive_seed_ocr_benchmark(20260825, SOURCE)
    b = variant_seed_arabic_scan_factory(SOURCE, 0, "", 0)
    v = derive_seed(20260831, SOURCE)
    assert len({a, b, v}) == 3


def test_page_seed_is_stable():
    assert page_seed(1, SOURCE, "doc", 0, 0, "p") == page_seed(
        1, SOURCE, "doc", 0, 0, "p"
    )


def test_scan_family_profile_module_matches_legacy_vectors():
    """The first-class scan_families module keeps the legacy variant_seed contract."""
    from clouda_data.factory.profiles import scan_families

    assert (
        scan_families.variant_seed(SOURCE, 0, "05_old_book_medium", 20260831)
        == 3497116117959753694
    )
    assert len(scan_families.PROFILES) == 12
    assert set(scan_families.PROFILES) >= {
        "01_high_quality_flatbed",
        "05_old_book_medium",
        "12_hard_composite",
    }
    severe = scan_families.PROFILES["06_old_book_heavy"]
    assert severe["recoverability"] == "SEVERE"
