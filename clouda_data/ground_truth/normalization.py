from __future__ import annotations

import re
import unicodedata
from typing import Literal

TATWEEL = "\u0640"
ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
DIGIT_VARIANTS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def preserve_original(text: str) -> str:
    return text


def normalize_unicode(
    text: str,
    form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC",
) -> str:
    return unicodedata.normalize(form, text)


def normalize_for_comparison(
    text: str,
    *,
    fold_alef: bool = True,
    fold_ya: bool = True,
    fold_ta_marbuta: bool = False,
    remove_tatweel: bool = True,
    remove_diacritics: bool = True,
    fold_digits: bool = False,
    collapse_spaces: bool = True,
) -> str:
    value = normalize_unicode(text)
    if remove_diacritics:
        value = ARABIC_DIACRITICS_RE.sub("", value)
    if remove_tatweel:
        value = value.replace(TATWEEL, "")
    if fold_alef:
        value = value.translate(ALEF_VARIANTS)
    if fold_ya:
        value = value.replace("ى", "ي")
    if fold_ta_marbuta:
        value = value.replace("ة", "ه")
    if fold_digits:
        value = value.translate(DIGIT_VARIANTS)
    if collapse_spaces:
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\s*\n\s*", "\n", value).strip()
    return value
