from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence

ACCEPTANCE_THRESHOLD = 90.0


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.insertions + self.deletions


def normalize_reference_text(text: str) -> str:
    """Normalize only encoding/newlines; do not remove punctuation or diacritics."""
    return unicodedata.normalize(
        "NFC", (text or "").replace("\r\n", "\n").replace("\r", "\n")
    )


def _edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)
    ]
    for i in range(1, rows):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, cols):
        dp[0][j] = (j, 0, j, 0)

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                keep = dp[i - 1][j - 1]
            else:
                d, s, ins, dele = dp[i - 1][j - 1]
                keep = (d + 1, s + 1, ins, dele)
            d, s, ins, dele = dp[i][j - 1]
            insert = (d + 1, s, ins + 1, dele)
            d, s, ins, dele = dp[i - 1][j]
            delete = (d + 1, s, ins, dele + 1)
            dp[i][j] = min(
                keep,
                insert,
                delete,
                key=lambda item: (item[0], item[1], item[2], item[3]),
            )

    _, substitutions, insertions, deletions = dp[-1][-1]
    return EditCounts(
        substitutions=substitutions, insertions=insertions, deletions=deletions
    )


def calculate_text_metrics(reference_text: str, hypothesis_text: str) -> dict:
    reference = normalize_reference_text(reference_text)
    hypothesis = normalize_reference_text(hypothesis_text)
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    char_edits = _edit_counts(ref_chars, hyp_chars)
    word_edits = _edit_counts(ref_words, hyp_words)
    ref_char_count = len(ref_chars)
    ref_word_count = len(ref_words)
    cer = (char_edits.distance / max(1, ref_char_count)) * 100.0
    wer = (word_edits.distance / max(1, ref_word_count)) * 100.0
    return {
        "cer": cer,
        "wer": wer,
        "character_accuracy": max(0.0, 100.0 - cer),
        "word_accuracy": max(0.0, 100.0 - wer),
        "char_substitutions": char_edits.substitutions,
        "char_insertions": char_edits.insertions,
        "char_deletions": char_edits.deletions,
        "word_substitutions": word_edits.substitutions,
        "word_insertions": word_edits.insertions,
        "word_deletions": word_edits.deletions,
        "reference_characters": ref_char_count,
        "reference_words": ref_word_count,
    }


def quality_gate_decision(
    *,
    character_accuracy: float | None,
    has_ground_truth: bool,
    estimated_quality: float | None,
    local_routes_exhausted: bool = False,
) -> tuple[str, str | None]:
    """Accept only real accuracy at or above 90%; heuristic quality is informational."""
    del estimated_quality
    if not has_ground_truth or character_accuracy is None:
        return "manual_review", "missing_ground_truth_or_unmeasured_accuracy"
    if character_accuracy >= ACCEPTANCE_THRESHOLD:
        return "accepted", None
    if local_routes_exhausted:
        return "manual_review", "local_routes_exhausted"
    return "retry", "below_90_percent_accuracy"
