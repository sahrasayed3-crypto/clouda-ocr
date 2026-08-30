from __future__ import annotations

from clouda_data.ground_truth.normalization import normalize_for_comparison


def normalize_ocr_text(text: str) -> str:
    return normalize_for_comparison(text, fold_digits=True)
