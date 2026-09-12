"""Tests for tiered text duplicate detection (Tier0/1/2)."""

from __future__ import annotations

import hashlib
import unicodedata

from clouda_data.quality.text_dup import (
    DEDUPE_TEXT_POLICY,
    MINHASH_SEED,
    TEXT_POLICY_VERSION,
    char_ngrams,
    classify_text_pairs,
    jaccard,
    lsh_candidates,
    minhash_signature,
    normalized_hash,
    raw_hash,
)

LONG_BASE = (
    "هذا نص طويل نسبيا يستخدم لاختبار بصمات النصوص في بوابة الجودة، "
    "ويتضمن جملًا كافية لتجاوز الحد الأدنى لعدد الأحرف المطلوب للتوقيع."
)

OTHER_TEXT = (
    "نص مختلف تماما عن النص الاساسي ولا يشبهه في المحتوى او المعنى "
    "الذي يحمله فهو يتحدث عن موضوع منفصل تماما ولا يشاركه اي كلمات"
)


def _drop_words(text: str, drops: list[str]) -> str:
    """Deterministically drop the first occurrence of each token."""

    out = text
    for d in drops:
        out = out.replace(d, " ", 1)
    return out.replace("  ", " ").strip()


# ~8% of characters removed -> J ~0.92 (near family).
NOISY = _drop_words(LONG_BASE, ["نسبيا "])
# ~15% removed -> J ~0.79 (review band).
REVIEW_VARIANT = _drop_words(LONG_BASE, ["نسبيا ", "بصمات ", "الأحرف "])


# ---------------------------------------------------------------------------
# Tier 0
# ---------------------------------------------------------------------------


def test_raw_hash_domain_separated_sha256() -> None:
    text = "مرحبا بالعالم"
    expected = hashlib.sha256(
        b"clouda.text.raw.v1\x00" + text.encode("utf-8")
    ).hexdigest()
    assert raw_hash(text) == expected
    assert raw_hash(text) != hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert raw_hash("") == hashlib.sha256(b"clouda.text.raw.v1\x00").hexdigest()


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


def test_policy_folds_diacritics_only_variants() -> None:
    plain = (
        "التعليم في مصر يشهد تطورا كبيرا في السنوات الاخيرة وباتجه نحو "
        "الاعتماد على التقنية الحديثة في ادارة الصفوف الدراسية بفاعلية"
    )
    vocalized = (
        "التَعلِيم في مصر يَشهد تَطوُّرًا كَبِيرًا في السَّنَوَاتِ الأخِيرَة وبَاتَجَه "
        "نَحو الاعتمَاد على التَّقنِيَة الحَدِيثَة في اِدارة الصُّفُوفِ الدِّرَاسِيَّةِ بِفَاعِلِيَّة"
    )
    assert normalized_hash(plain) == normalized_hash(vocalized)
    assert normalized_hash(plain) != raw_hash(plain)


def test_policy_strips_tatweel() -> None:
    plain = (
        "مرحبا بكم في هذا النص الطويل الذي يحتوي على كلمات كافية "
        "لاختبار التطبيع العربي الشامل في بوابة الجودة"
    )
    stretched = (
        "مرحــــبا بـــــكم في هــــذا النــــص الطــــويــــل الذي يحتوي على كلمات كافية "
        "لاختبار التطبيع العربي الشامل في بوابة الجودة"
    )
    assert normalized_hash(plain) == normalized_hash(stretched)


def test_policy_folds_alef_and_ya_variants() -> None:
    # أحمد/احمد and على/علي must hash identically (fold_alef + fold_ya).
    assert normalized_hash("أحمد علي") == normalized_hash("احمد علي")
    assert normalized_hash("أحمد") == normalized_hash("احمد")
    assert normalized_hash("على") == normalized_hash("علي")
    # Full-sentence with mixed hamza carriers and alef maqsura.
    base = (
        "احمد علي هذا نص طويل بما يكفي لتجاوز الحد الادنى لعدد الاحرف "
        "المطلوب للتوقيع الرقمي في بوابة الجودة"
    )
    hamza = (
        "أحمد على هذا نص طويل بما يكفي لتجاوز الحد الأدنى لعدد الأحرف "
        "المطلوب للتوقيع الرقمي في بوابة الجودة"
    )
    assert normalized_hash(base) == normalized_hash(hamza)


def test_digits_are_not_folded() -> None:
    assert normalized_hash("50 ريال") != normalized_hash("٥٠ ريال")
    assert DEDUPE_TEXT_POLICY.fold_digits is False


def test_mixed_ar_en_nfc_vs_nfd_same_hash() -> None:
    mixed = "ChatGPT والذكاء الاصطناعي AI في التعليم Machine Learning 2026 é"
    nfc = unicodedata.normalize("NFC", mixed)
    nfd = unicodedata.normalize("NFD", mixed)
    assert nfc != nfd
    assert normalized_hash(nfc) == normalized_hash(nfd)


def test_text_policy_version_is_stable_string() -> None:
    assert TEXT_POLICY_VERSION == DEDUPE_TEXT_POLICY.version()
    prefix = "clouda.pretraining.normalize.v1+"
    assert TEXT_POLICY_VERSION.startswith(prefix)
    assert len(TEXT_POLICY_VERSION) == len(prefix) + 12


def test_policy_matches_expected_configuration() -> None:
    policy = DEDUPE_TEXT_POLICY
    assert policy.unicode_form == "NFKC"
    assert policy.presentation_forms == "compose"
    assert policy.strip_bom is True
    assert policy.normalize_line_endings is True
    assert policy.preserve_line_breaks is True
    assert policy.collapse_whitespace is True
    assert policy.remove_zero_width is True
    assert policy.remove_control_characters is True
    assert policy.remove_tatweel is True
    assert policy.remove_diacritics is True
    assert policy.fold_alef is True
    assert policy.fold_ya is True
    assert policy.fold_digits is False


# ---------------------------------------------------------------------------
# Tier 2: MinHash
# ---------------------------------------------------------------------------


def test_short_text_returns_none() -> None:
    assert minhash_signature("قصير") is None
    assert minhash_signature("ا" * 39) is None


def test_signature_length_boundary() -> None:
    sig_39 = minhash_signature("ا" * 39)
    sig_40 = minhash_signature("ا" * 40)
    assert sig_39 is None
    assert sig_40 is not None
    assert len(sig_40) == 128


def test_same_text_twice_identical_signature() -> None:
    a = minhash_signature(LONG_BASE)
    b = minhash_signature(LONG_BASE)
    assert a is not None and b is not None
    assert a == b


def test_signature_values_in_mersenne_range() -> None:
    sig = minhash_signature(LONG_BASE)
    assert sig is not None
    for v in sig:
        assert 0 <= v < 2**61 - 1


def test_similar_texts_agree_on_most_permutations() -> None:
    sig_a = minhash_signature(LONG_BASE)
    sig_b = minhash_signature(NOISY)
    assert sig_a is not None and sig_b is not None
    agree = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    assert agree / len(sig_a) > 0.5


# ---------------------------------------------------------------------------
# Tier 2: LSH + Jaccard + classification
# ---------------------------------------------------------------------------


def test_lsh_finds_similar_pair() -> None:
    sig_a = minhash_signature(LONG_BASE)
    sig_b = minhash_signature(NOISY)
    assert sig_a is not None and sig_b is not None
    sigs: dict[str, tuple[int, ...]] = {
        "a": sig_a,
        "b": sig_b,
    }
    pairs, skipped = lsh_candidates(sigs, bands=16, rows=8, owner_cap=20)
    assert skipped == {}
    assert ("a", "b") in pairs


def test_lsh_owner_cap_skips_large_buckets() -> None:
    sig = minhash_signature(LONG_BASE)
    assert sig is not None
    ids = {f"s{i}": sig for i in range(25)}
    pairs, skipped = lsh_candidates(ids, bands=16, rows=8, owner_cap=20)
    assert pairs == set()
    assert len(skipped) == 16  # every band's bucket exceeded the cap
    assert all(count == 25 for count in skipped.values())


def test_lsh_owner_cap_allows_under_cap() -> None:
    sig = minhash_signature(LONG_BASE)
    assert sig is not None
    ids = {f"s{i}": sig for i in range(20)}
    pairs, skipped = lsh_candidates(ids, bands=16, rows=8, owner_cap=20)
    assert skipped == {}
    assert len(pairs) == 190  # C(20,2)


def test_jaccard_exact() -> None:
    a = char_ngrams("abcd efgh ijkl")
    b = char_ngrams("abcd efgh ijkl")
    assert jaccard(a, b) == 1.0
    assert jaccard(a, []) == 0.0
    assert jaccard([], []) == 0.0
    c = char_ngrams("abcd efgh ijkm")
    inter = len(set(a) & set(c))
    union = len(set(a) | set(c))
    assert jaccard(a, c) == inter / union


def test_classify_near_family() -> None:
    texts = {"base": LONG_BASE, "noisy": NOISY, "other": OTHER_TEXT}
    result = classify_text_pairs(texts)
    assert ("base", "noisy") in result.near_text
    assert ("base", "noisy") not in result.review
    assert ("base", "other") not in result.near_text
    assert ("base", "other") not in result.review


def test_classify_review_band() -> None:
    direct = jaccard(
        char_ngrams(LONG_BASE),
        char_ngrams(REVIEW_VARIANT),
    )
    assert 0.70 <= direct < 0.85
    result = classify_text_pairs({"a": LONG_BASE, "b": REVIEW_VARIANT})
    assert ("a", "b") in result.review
    assert ("a", "b") not in result.near_text


def test_classify_skips_short_texts() -> None:
    texts = {"short": "قصير", "long": LONG_BASE}
    result = classify_text_pairs(texts)
    assert result.skipped_ids == ("short",)
    assert result.near_text == ()
    assert result.review == ()


def test_classify_deterministic_across_runs() -> None:
    texts = {"a": LONG_BASE, "b": NOISY, "c": OTHER_TEXT}
    r1 = classify_text_pairs(texts)
    r2 = classify_text_pairs(texts)
    assert r1 == r2
    assert MINHASH_SEED == "clouda.quality.textdup.v1"


def test_distinct_pages_sharing_header_only_no_candidate() -> None:
    header = (
        "التقرير السنوي لوزارة التعليم - الفصل الثالث: تقرير عام "
        "عن حالة التعليم في الوطن"
    )
    page_a = header + (
        " يتناول هذا التقرير تطور المناهج الدراسية واضافة مواد جديدة للحاسوب والعلوم"
    )
    page_b = header + (
        " يتناول هذا التقرير ارتفاع تكاليف النقل المدرسي وتوزيع الكتب الدراسية مجانا"
    )
    result = classify_text_pairs({"page_a": page_a, "page_b": page_b})
    assert ("page_a", "page_b") not in result.near_text
    assert ("page_a", "page_b") not in result.review
    direct = jaccard(char_ngrams(page_a), char_ngrams(page_b))
    assert direct < 0.70
