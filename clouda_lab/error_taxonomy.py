"""Arabic-aware OCR error classification.

Deterministic, testable classifiers that map aligned character/word edits to
practical error categories. Classification is purely rule-based on the paired
strings — no linguistic overclaiming, no model calls.

Category order matters: the first matching rule wins, so specific categories
(diacritic, hamza variants, digits) are tested before the generic ones.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Character classes (Unicode Arabic block knowledge, kept minimal & explicit)
# ---------------------------------------------------------------------------

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"

ALEF_VARIANTS = frozenset("أإآٱ")
ALEF_BASE = "ا"
YA = "ي"
ALEF_MAQSURA = "ى"
TA_MARBUTA = "ة"
HA = "ه"
HAMZA = "ء"

ARABIC_DIGITS = frozenset("٠١٢٣٤٥٦٧٨٩")  # U+0660..U+0669
ARABIC_DIGITS_EXT = frozenset("۰۱۲۳۴۵۶۷۸۹")  # U+06F0..U+06F9 (Persian/Urdu)
LATIN_DIGITS = frozenset("0123456789")

ARABIC_LETTERS_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
LATIN_LETTERS_RE = re.compile(r"[A-Za-z]")

PUNCTUATION_RE = re.compile(
    r"[\u060C\u061B\u061F\u066A-\u066D\u06D4.,;:!?\"'()\[\]{}«»…\-–—/\\|@#&*%$+=<>~^_]"
)
WHITESPACE_RE = re.compile(r"\s")


def _single(character: str | None) -> bool:
    return character is not None and len(character) == 1


def classify_char_error(gt: str | None, pred: str | None) -> str:
    """Classify one character-level edit deterministically.

    ``gt`` is ``None`` for insertions, ``pred`` is ``None`` for deletions,
    and exactly one of them is ``None`` — substitutions carry both.
    """
    if gt is None and _single(pred):
        character = pred or ""
        if WHITESPACE_RE.match(character):
            return "whitespace"
        if PUNCTUATION_RE.match(character):
            return "punctuation"
        if character in ARABIC_DIGITS or character in ARABIC_DIGITS_EXT:
            return "arabic_digit"
        if character in LATIN_DIGITS:
            return "latin_digit"
        if ARABIC_DIACRITICS_RE.match(character):
            return "diacritic"
        if character == TATWEEL:
            return "tatweel"
        if ARABIC_LETTERS_RE.match(character):
            return "extra_arabic_character"
        if LATIN_LETTERS_RE.match(character):
            return "extra_latin_character"
        return "extra_character"
    if pred is None and _single(gt):
        character = gt or ""
        if WHITESPACE_RE.match(character):
            return "whitespace"
        if PUNCTUATION_RE.match(character):
            return "punctuation"
        if character in ARABIC_DIGITS or character in ARABIC_DIGITS_EXT:
            return "arabic_digit"
        if character in LATIN_DIGITS:
            return "latin_digit"
        if ARABIC_DIACRITICS_RE.match(character):
            return "diacritic"
        if character == TATWEEL:
            return "tatweel"
        if ARABIC_LETTERS_RE.match(character):
            return "missing_arabic_character"
        if LATIN_LETTERS_RE.match(character):
            return "missing_latin_character"
        return "missing_character"
    if _single(gt) and _single(pred):
        return _classify_char_substitution(gt or "", pred or "")
    return "unknown"


def _classify_char_substitution(gt: str, pred: str) -> str:
    if gt == pred:
        return "match"
    if WHITESPACE_RE.match(gt) or WHITESPACE_RE.match(pred):
        return "whitespace"
    if PUNCTUATION_RE.match(gt) and PUNCTUATION_RE.match(pred):
        return "punctuation"
    # Digits (before script confusion: ٣ vs 3 is digit confusion)
    gt_digit = gt in ARABIC_DIGITS or gt in ARABIC_DIGITS_EXT or gt in LATIN_DIGITS
    pred_digit = (
        pred in ARABIC_DIGITS or pred in ARABIC_DIGITS_EXT or pred in LATIN_DIGITS
    )
    if gt_digit and pred_digit:
        if (gt in LATIN_DIGITS) != (pred in LATIN_DIGITS):
            return "digit_system_confusion"  # Arabic-Indic <-> Latin
        return "arabic_digit"
    if ARABIC_DIACRITICS_RE.match(gt) or ARABIC_DIACRITICS_RE.match(pred):
        return "diacritic"
    if gt == TATWEEL or pred == TATWEEL:
        return "tatweel"
    # Alef variants fold to the same base letter
    if (gt in ALEF_VARIANTS and pred == ALEF_BASE) or (
        pred in ALEF_VARIANTS and gt == ALEF_BASE
    ):
        return "hamza_alef_variant"
    if gt in ALEF_VARIANTS and pred in ALEF_VARIANTS:
        return "hamza_alef_variant"
    if {gt, pred} == {YA, ALEF_MAQSURA}:
        return "ya_alef_maqsura"
    if {gt, pred} == {TA_MARBUTA, HA}:
        return "ta_marbuta_ha"
    if gt == HAMZA or pred == HAMZA:
        return "hamza"
    # Script confusion: Arabic letter vs Latin letter
    if ARABIC_LETTERS_RE.match(gt) and LATIN_LETTERS_RE.match(pred):
        return "arabic_latin_script"
    if LATIN_LETTERS_RE.match(gt) and ARABIC_LETTERS_RE.match(pred):
        return "arabic_latin_script"
    if ARABIC_LETTERS_RE.match(gt) and ARABIC_LETTERS_RE.match(pred):
        return "character_substitution"
    if LATIN_LETTERS_RE.match(gt) and LATIN_LETTERS_RE.match(pred):
        return "character_substitution"
    return "character_substitution"


# ---------------------------------------------------------------------------
# Word-level classification
# ---------------------------------------------------------------------------


def classify_word_error(gt: str | None, pred: str | None) -> str:
    """Classify one word-level edit deterministically.

    Word classification is structural (missing/extra word) or, for
    substitutions, derived from the character mix of the pair. Diacritic-only
    differences are detected explicitly so normalization-sensitive words are
    not mislabeled as plain substitutions.
    """
    if gt is None:
        return "extra_word"
    if pred is None:
        return "missing_word"
    if gt == pred:
        return "match"
    stripped_gt = ARABIC_DIACRITICS_RE.sub("", unicodedata.normalize("NFC", gt))
    stripped_pred = ARABIC_DIACRITICS_RE.sub("", unicodedata.normalize("NFC", pred))
    if stripped_gt == stripped_pred:
        return "diacritic"
    if _contains_digit(gt) or _contains_digit(pred):
        return "digit_word"
    if _is_punctuation_only(gt) or _is_punctuation_only(pred):
        return "punctuation"
    # Mixed-script word pair (e.g. "Python" vs "بايثون")
    if _is_latin_word(gt) != _is_latin_word(pred):
        return "mixed_script_word"
    return "word_substitution"


def _contains_digit(word: str) -> bool:
    return any(
        ch in LATIN_DIGITS or ch in ARABIC_DIGITS or ch in ARABIC_DIGITS_EXT
        for ch in word
    )


def _is_punctuation_only(word: str) -> bool:
    return bool(word) and all(PUNCTUATION_RE.match(ch) for ch in word)


def _is_latin_word(word: str) -> bool:
    return (
        bool(word)
        and all(
            LATIN_LETTERS_RE.match(ch) or not ARABIC_LETTERS_RE.match(ch) for ch in word
        )
        and any(LATIN_LETTERS_RE.match(ch) for ch in word)
    )


# Canonical category list (for stable exports and tests)
CHAR_ERROR_CATEGORIES: tuple[str, ...] = (
    "match",
    "whitespace",
    "punctuation",
    "arabic_digit",
    "latin_digit",
    "digit_system_confusion",
    "diacritic",
    "tatweel",
    "hamza_alef_variant",
    "ya_alef_maqsura",
    "ta_marbuta_ha",
    "hamza",
    "arabic_latin_script",
    "character_substitution",
    "missing_arabic_character",
    "missing_latin_character",
    "missing_character",
    "extra_arabic_character",
    "extra_latin_character",
    "extra_character",
    "unknown",
)

WORD_ERROR_CATEGORIES: tuple[str, ...] = (
    "match",
    "missing_word",
    "extra_word",
    "diacritic",
    "digit_word",
    "punctuation",
    "mixed_script_word",
    "word_substitution",
)
