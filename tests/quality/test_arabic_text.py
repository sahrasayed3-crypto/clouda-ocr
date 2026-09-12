"""Arabic text normalization + near-text dedupe tests."""

from __future__ import annotations

from clouda_data.pretraining.normalize import (
    NormalizationPolicy,
    normalize_text,
)  # noqa: E402

from clouda_data.quality import text_dup


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
        text = "هذا نص عربي طويل لاختبار كشف الصفحات النصية المتشابهة بدقة"
        result = text_dup.classify_text_pairs({"smp_a": text, "smp_b": text})
        assert result.near_text == (("smp_a", "smp_b"),)

    def test_short_text_skips_tier2(self) -> None:
        result = text_dup.classify_text_pairs({"smp_a": "قصير", "smp_b": "نص"})
        assert result.near_text == ()
        assert result.skipped_ids == ("smp_a", "smp_b")
