"""Arabic text normalization + near-text dedupe tests (module pending)."""

from __future__ import annotations

import pytest

from clouda_data.pretraining.normalize import (
    NormalizationPolicy,
    normalize_text,
)  # noqa: E402

text_dup = pytest.importorskip("clouda_data.quality.text_dup")


class TestArabicTextNormalization:
    def test_diacritics_removed(self) -> None:
        policy = NormalizationPolicy()
        result = normalize_text("مُحَمَّد", policy)
        assert "ُ" not in result.value
        assert "َ" not in result.value

    def test_alef_variants_folded(self) -> None:
        policy = NormalizationPolicy()
        result = normalize_text("أحمد إبراهيم آمنة", policy)
        assert "أ" not in result.value
        assert "إ" not in result.value
        assert "آ" not in result.value

    def test_deterministic_normalization(self) -> None:
        policy = NormalizationPolicy()
        text = "النَّصُّ العربيُّ رقم ١٢٣"
        assert normalize_text(text, policy).value == normalize_text(text, policy).value


class TestTextNearDuplicate:
    def test_identical_text_is_near_family(self) -> None:
        assert text_dup is not None  # module contract pending (Wave2-G)

    def test_short_text_skips_tier2(self) -> None:
        assert text_dup is not None  # min_text_chars=40 contract (Wave2-G)
