"""Configurable, versioned Arabic/multilingual text normalization.

Normalization never destroys information: the raw text is preserved
untouched and the normalized form is stored separately. Every behavior that
could lose Arabic information (diacritics, tatweel, digit folding, alef/ya
folding, presentation forms) is opt-in via :class:`NormalizationPolicy`.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Literal, cast

from clouda_data.ground_truth.normalization import (
    ALEF_VARIANTS,
    ARABIC_DIACRITICS_RE,
    DIGIT_VARIANTS,
)

NORMALIZATION_VERSION = "clouda.pretraining.normalize.v1"

TATWEEL = "\u0640"
BOM = "\ufeff"

# Zero-width characters and bidirectional control marks.
ZERO_WIDTH_CHARS = (
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",  # bidi embedding
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",  # bidi isolate
    "\ufeff",  # zero width no-break space (BOM in text)
)

# C0/C1 control characters except tab and newline.
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")


@dataclass(frozen=True)
class NormalizationPolicy:
    """Explicit, serializable text normalization policy."""

    unicode_form: str = "NFC"
    strip_bom: bool = True
    normalize_line_endings: bool = True
    preserve_line_breaks: bool = True
    collapse_whitespace: bool = False
    remove_zero_width: bool = False
    remove_control_characters: bool = False
    remove_tatweel: bool = False
    remove_diacritics: bool = False
    fold_alef: bool = False
    fold_ya: bool = False
    fold_digits: bool = False
    presentation_forms: str = "preserve"  # or "compose"

    def __post_init__(self) -> None:
        boolean_fields = (
            "strip_bom",
            "normalize_line_endings",
            "preserve_line_breaks",
            "collapse_whitespace",
            "remove_zero_width",
            "remove_control_characters",
            "remove_tatweel",
            "remove_diacritics",
            "fold_alef",
            "fold_ya",
            "fold_digits",
        )
        if any(not isinstance(getattr(self, name), bool) for name in boolean_fields):
            raise TypeError("Normalization switches must be booleans.")
        if self.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(f"Unsupported unicode form: {self.unicode_form}")
        if self.presentation_forms not in {"preserve", "compose"}:
            raise ValueError("presentation_forms must be 'preserve' or 'compose'")
        if (
            self.unicode_form in {"NFKC", "NFKD"}
            and self.presentation_forms != "compose"
        ):
            raise ValueError(
                "Compatibility normalization requires presentation_forms='compose'"
            )
        if self.presentation_forms == "compose" and self.unicode_form != "NFKC":
            raise ValueError(
                "presentation_forms='compose' requires unicode_form='NFKC'"
            )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def version(self) -> str:
        return f"{NORMALIZATION_VERSION}+{self.fingerprint()}"


@dataclass(frozen=True)
class NormalizedText:
    value: str
    applied: tuple[str, ...]


def normalize_text(text: str, policy: NormalizationPolicy) -> NormalizedText:
    """Return the normalized text plus the ordered list of applied steps.

    The input string is never modified; callers keep the original as
    ``raw_text``.
    """

    applied: list[str] = []
    value = text

    if policy.strip_bom and value.startswith(BOM):
        value = value.lstrip(BOM)
        applied.append("strip_bom")

    if policy.normalize_line_endings and ("\r" in value):
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        applied.append("normalize_line_endings")

    form = cast("Literal['NFC', 'NFD', 'NFKC', 'NFKD']", policy.unicode_form)
    if not unicodedata.is_normalized(form, value):
        value = unicodedata.normalize(form, value)
        applied.append(f"unicode_{policy.unicode_form.lower()}")

    if policy.remove_zero_width:
        removed = "".join(ZERO_WIDTH_CHARS)
        if any(ch in value for ch in removed):
            value = value.translate(str.maketrans("", "", removed))
            applied.append("remove_zero_width_and_bidi_marks")

    if policy.remove_control_characters:
        if CONTROL_CHARS_RE.search(value):
            value = CONTROL_CHARS_RE.sub("", value)
            applied.append("remove_control_characters")

    if policy.remove_tatweel and TATWEEL in value:
        value = value.replace(TATWEEL, "")
        applied.append("remove_tatweel")

    if policy.remove_diacritics and ARABIC_DIACRITICS_RE.search(value):
        value = ARABIC_DIACRITICS_RE.sub("", value)
        applied.append("remove_arabic_diacritics")

    if policy.fold_alef and any(ch in value for ch in "\u0623\u0625\u0622\u0671"):
        value = value.translate(ALEF_VARIANTS)
        applied.append("fold_alef_variants")

    if policy.fold_ya and "\u0649" in value:
        value = value.replace("\u0649", "\u064a")
        applied.append("fold_alef_maqsura_to_ya")

    if policy.fold_digits and any(
        ch in value
        for ch in "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
        "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
    ):
        value = value.translate(DIGIT_VARIANTS)
        applied.append("fold_arabic_indic_digits")

    if policy.collapse_whitespace:
        value = re.sub(r"[^\S\n]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = value.strip()
        applied.append("collapse_whitespace")

    if not policy.preserve_line_breaks and "\n" in value:
        value = value.replace("\n", " ")
        value = re.sub(r" +", " ", value).strip()
        applied.append("flatten_line_breaks")

    return NormalizedText(value=value, applied=tuple(applied))
