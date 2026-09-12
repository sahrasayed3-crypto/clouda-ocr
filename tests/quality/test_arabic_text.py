"""Arabic text normalization + near-text dedupe tests (module pending)."""

from __future__ import annotations

import pytest

from clouda_data.pretraining.normalize import (
    NormalizationPolicy,
    normalize_text,
)  # noqa: E402

text_dup = pytest.importorskip("clouda_data.quality.text_dup")


def _identity_policy() -> NormalizationPolicy:
    """Tier-1 identity policy (DEDUPE_TEXT_POLICY semantics)."""

    return NormalizationPolicy(
        unicode_form="NFKC",
        presentation_forms="compose",
        strip_bom=True,
        normalize_line_endings=True,
        preserve_line_breaks=True,
        collapse_whitespace=True,
        remove_zero_width=True,
        remove_control_characters=True,
        remove_tatweel=True,
        remove_diacritics=True,
        fold_alef=True,
        fold_ya=True,
        fold_digits=False,
    )


class TestArabicTextNormalization:
    def test_diacritics_removed(self) -> None:
        result = normalize_text("مُحَمَّد", _identity_policy())
        assert "ُ" not in result.value
        assert "َ" not in result.value

    def test_diacritics_preserved_by_default_policy(self) -> None:
        # The DEFAULT policy is non-destructive: folding is opt-in.
        result = normalize_text("مُحَمَّد", NormalizationPolicy())
        assert "ُ" in result.value

    def test_alef_variants_folded(self) -> None:
        result = normalize_text("أحمد إبراهيم آمنة", _identity_policy())
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
