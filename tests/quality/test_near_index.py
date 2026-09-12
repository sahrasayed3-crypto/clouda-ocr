"""Tests for clouda_data.quality.near_index."""

from __future__ import annotations

import pytest
from PIL import Image

from clouda_data.quality.config import ImageFingerprintPolicy, QualityGateConfig
from clouda_data.quality.models import NearDuplicateCandidate, SampleFingerprint
from clouda_data.quality.near_index import (
    LEVEL_CANDIDATE,
    LEVEL_CONFIRMED,
    LEVEL_LIKELY,
    build_fingerprints,
    build_fingerprints_with_summary,
    candidate_pairs,
    candidate_pairs_with_summary,
    page_index_meta,
)
from tests.quality.conftest import (  # noqa: E402
    blank_like,
    brightness,
    make_row,
    recompress,
    render_arabic_page,
    resize_img,
    save_png,
)


def _config(max_bucket: int = 4096) -> QualityGateConfig:
    return QualityGateConfig(
        image_fingerprint=ImageFingerprintPolicy(max_bucket=max_bucket)
    )


def _fp(
    sample_id: str,
    image: Image.Image,
    *,
    source_id: str = "src1",
) -> SampleFingerprint:
    from clouda_data.quality.image_fp import fingerprint_image

    result = fingerprint_image(image)
    return SampleFingerprint(
        sample_id=sample_id,
        source_id=source_id,
        ahash=result.ahash,
        dhash=result.dhash,
        phash=result.phash,
        aspect_bucket=result.aspect_bucket,
        blankish=result.blankish,
        blank_stats=dict(result.blank_stats),
        fingerprint_version=result.fingerprint_version,
    )


def _pairs(
    fps: list[SampleFingerprint], **kwargs: object
) -> list[NearDuplicateCandidate]:
    return candidate_pairs(fps, _config(), **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Identical images
# ---------------------------------------------------------------------------


class TestIdenticalImages:
    def test_identical_pages_produce_confirmed_pair(self, tmp_path) -> None:
        base = render_arabic_page(21, "plain", "صفحة")
        save_png(base, tmp_path / "a.png")
        save_png(base, tmp_path / "b.png")
        rows = [
            make_row("smp_a", image_path="a.png"),
            make_row("smp_b", image_path="b.png"),
        ]
        fps = build_fingerprints(rows, tmp_path, _config())
        assert len(fps) == 2
        pairs = _pairs(fps)
        assert len(pairs) == 1
        pair = pairs[0]
        assert (pair.sample_id_a, pair.sample_id_b) == ("smp_a", "smp_b")
        assert pair.level == LEVEL_CONFIRMED
        assert pair.distances["d_p"] == 0
        assert pair.distances["d_d"] == 0
        assert pair.distances["d_a"] == 0


# ---------------------------------------------------------------------------
# Mutations of the same page
# ---------------------------------------------------------------------------


class TestMutations:
    @pytest.mark.parametrize(
        ("mutate", "label"),
        [
            (lambda img: recompress(img, quality=40), "recompressed"),
            (lambda img: resize_img(img, 0.5), "resized"),
            (lambda img: brightness(img, 20), "brightened"),
        ],
    )
    def test_mutation_stays_confirmed_or_likely(self, tmp_path, mutate, label) -> None:
        base = render_arabic_page(7, "plain", "نص عربي")
        save_png(base, tmp_path / "orig.png")
        save_png(mutate(base).convert("L"), tmp_path / "mut.png")
        rows = [
            make_row("smp_orig", image_path="orig.png"),
            make_row("smp_mut", image_path="mut.png"),
        ]
        fps = build_fingerprints(rows, tmp_path, _config())
        assert len(fps) == 2, f"{label}: both pages must be indexed"
        pairs = _pairs(fps)
        assert len(pairs) == 1, f"{label}: expected exactly one pair"
        assert pairs[0].level in {
            LEVEL_CONFIRMED,
            LEVEL_LIKELY,
        }, f"{label}: level was {pairs[0].level} with {pairs[0].distances}"


# ---------------------------------------------------------------------------
# Distinct content
# ---------------------------------------------------------------------------


class TestDistinctContent:
    def test_two_distinct_pages_produce_no_pair(self, tmp_path) -> None:
        a = render_arabic_page(21, "plain", "صفحة")
        b = render_arabic_page(99, "dense", "مختلف تماما")
        save_png(a, tmp_path / "a.png")
        save_png(b, tmp_path / "b.png")
        rows = [
            make_row("smp_a", image_path="a.png"),
            make_row("smp_b", image_path="b.png"),
        ]
        fps = build_fingerprints(rows, tmp_path, _config())
        assert len(fps) == 2
        assert _pairs(fps) == []


# ---------------------------------------------------------------------------
# Aspect-bucket gating
# ---------------------------------------------------------------------------


class TestAspectBucket:
    def test_different_aspect_buckets_never_pair(self) -> None:
        base = render_arabic_page(21, "plain", "صفحة")
        squashed = base.resize((base.width // 2, base.height))
        fp_a = _fp("smp_a", base)
        fp_b = _fp("smp_b", squashed)
        assert fp_a.aspect_bucket != fp_b.aspect_bucket
        assert _pairs([fp_a, fp_b]) == []


# ---------------------------------------------------------------------------
# Blankish exclusion
# ---------------------------------------------------------------------------


class TestBlankishExclusion:
    def test_blankish_pages_fingerprinted_but_excluded(self, tmp_path) -> None:
        blank = blank_like(render_arabic_page(21, "plain", "صفحة"))
        save_png(blank, tmp_path / "blank.png")
        rows = [
            make_row("smp_blank", image_path="blank.png"),
            make_row("smp_blank2", image_path="blank.png"),
        ]
        stage = build_fingerprints_with_summary(rows, tmp_path, _config())
        assert stage.summary.skipped_blankish == ("smp_blank", "smp_blank2")
        assert stage.fingerprints == ()
        assert _pairs(list(stage.fingerprints)) == []


# ---------------------------------------------------------------------------
# Overflow guard
# ---------------------------------------------------------------------------


class TestOverflowGuard:
    def test_oversized_bucket_skipped_and_flagged_without_crash(self) -> None:
        cfg = _config(max_bucket=3)
        base = render_arabic_page(21, "plain", "صفحة")
        fps = [_fp(f"smp_{i:02d}", base) for i in range(5)]
        stage = candidate_pairs_with_summary(fps, cfg)
        assert stage.candidates == ()
        assert stage.summary.overflowed_buckets, "overflow must be flagged"
        assert stage.summary.overflowed_entries > 0

    def test_overflow_cap_respected(self) -> None:
        cfg = _config(max_bucket=4)
        base = render_arabic_page(3, "plain", "حروف")
        fps = [_fp(f"smp_{i:02d}", base) for i in range(6)]
        stage = candidate_pairs_with_summary(fps, cfg)
        assert stage.candidates == ()
        assert len(stage.summary.overflowed_buckets) >= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_twice_identical_candidates(self, tmp_path) -> None:
        base = render_arabic_page(21, "plain", "صفحة")
        mutated = recompress(base, quality=60)
        save_png(base, tmp_path / "a.png")
        save_png(mutated, tmp_path / "b.png")
        save_png(render_arabic_page(99, "dense", "مختلف"), tmp_path / "c.png")
        rows = [
            make_row("smp_c", image_path="c.png"),
            make_row("smp_b", image_path="b.png"),
            make_row("smp_a", image_path="a.png"),
        ]
        cfg = _config()
        first = build_fingerprints(rows, tmp_path, cfg)
        second = build_fingerprints(list(reversed(rows)), tmp_path, cfg)
        assert first == second
        meta = page_index_meta(rows)
        pairs_1 = _pairs(first, page_meta=meta)
        pairs_2 = _pairs(second, page_meta=meta)
        assert pairs_1 == pairs_2
        ids = [(p.sample_id_a, p.sample_id_b) for p in pairs_1]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Intra-document adjacent-page pass
# ---------------------------------------------------------------------------


class TestIntraDocumentAdjacency:
    def test_adjacent_pages_of_same_document_become_candidates(self) -> None:
        base = render_arabic_page(21, "plain", "صفحة")
        # Same content family but distinct enough that banding may not fire:
        # adjacency alone must surface the pair.
        fp_a = _fp("smp_p1", base)
        fp_b = _fp("smp_p2", brightness(base, 4))
        meta = {
            "smp_p1": ("doc-1", 0),
            "smp_p2": ("doc-1", 1),
        }
        pairs = _pairs([fp_a, fp_b], page_meta=meta)
        assert len(pairs) == 1
        assert (pairs[0].sample_id_a, pairs[0].sample_id_b) == ("smp_p1", "smp_p2")
        assert pairs[0].level in {LEVEL_CONFIRMED, LEVEL_LIKELY, LEVEL_CANDIDATE}

    def test_non_adjacent_pages_not_paired(self, tmp_path) -> None:
        a = render_arabic_page(21, "plain", "صفحة")
        b = render_arabic_page(55, "plain", "نص آخر")
        save_png(a, tmp_path / "a.png")
        save_png(b, tmp_path / "b.png")
        rows = [
            make_row("smp_p1", image_path="a.png", page_index=0),
            make_row("smp_p5", image_path="b.png", page_index=4),
        ]
        fps = build_fingerprints(rows, tmp_path, _config())
        assert _pairs(fps, page_meta=page_index_meta(rows)) == []

    def test_adjacent_pages_of_different_documents_not_paired(self, tmp_path) -> None:
        # Two near-identical pages that banding WOULD pair when adjacent in
        # the same document; different documents must suppress the adjacency
        # pass even at identical page indices.
        base = render_arabic_page(21, "plain", "صفحة")
        save_png(base, tmp_path / "a.png")
        save_png(recompress(base, quality=85), tmp_path / "b.png")
        rows = [
            make_row("smp_p1", image_path="a.png", document_id="doc-1", page_index=0),
            make_row("smp_p2", image_path="b.png", document_id="doc-2", page_index=1),
        ]
        fps = build_fingerprints(rows, tmp_path, _config())
        pairs = _pairs(fps, page_meta=page_index_meta(rows))
        if pairs:
            # Banding may still legitimately pair them on content alone; the
            # adjacency pass itself must add nothing beyond the banding pairs.
            stage_plain = candidate_pairs_with_summary(fps, _config())
            assert stage_plain.candidates == tuple(pairs)


# ---------------------------------------------------------------------------
# Candidate dataclass invariants
# ---------------------------------------------------------------------------


class TestCandidateInvariants:
    def test_level_values_match_model_vocabulary(self) -> None:
        from clouda_data.quality.models import NEAR_DUPLICATE_LEVELS

        assert LEVEL_CANDIDATE in NEAR_DUPLICATE_LEVELS
        assert LEVEL_LIKELY in NEAR_DUPLICATE_LEVELS
        assert LEVEL_CONFIRMED in NEAR_DUPLICATE_LEVELS

    def test_near_duplicate_candidate_rejects_self_pair(self) -> None:
        with pytest.raises(ValueError):
            NearDuplicateCandidate(
                sample_id_a="smp_a",
                sample_id_b="smp_a",
                level=LEVEL_CANDIDATE,
            )
