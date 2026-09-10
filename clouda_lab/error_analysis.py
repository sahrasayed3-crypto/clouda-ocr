"""OCR Error Analysis Engine.

Character- and word-level alignment with Arabic-aware error classification.

Metric policy — the engine composes the existing project metric functions and
does not reimplement them:

- ``clouda_data.evaluation.cer.cer`` / ``wer.wer`` for raw CER/WER.
- ``clouda_data.ground_truth.normalization.normalize_for_comparison`` for the
  normalized (N-CER) variant, mirroring the canonical
  ``normalize_ocr_text`` policy (fold digits + alef + ya, strip diacritics
  and tatweel, collapse spaces).

Alignment is computed with a Levenshtein backtrace whose tie-breaking is
deterministic (substitution preferred over insertion over deletion at equal
cost), so repeated analysis of the same input always yields the same records.
"""

from __future__ import annotations

from typing import Iterable

from clouda_data.evaluation.cer import cer as project_cer
from clouda_data.evaluation.wer import wer as project_wer
from clouda_data.ground_truth.normalization import normalize_for_comparison

from .error_taxonomy import classify_char_error, classify_word_error
from .models import ErrorAnalysis, ErrorRecord, OCRSample

_OP_SUB = "substitution"
_OP_INS = "insertion"
_OP_DEL = "deletion"
_OP_MATCH = "match"

# Backtrace tie-break priority: match/substitution, then deletion, then
# insertion. Kept as explicit tuples so the DP recurrence is auditable.
_TRACE_SUB = 0
_TRACE_DEL = 1
_TRACE_INS = 2

_CONTEXT_RADIUS = 12


def normalize_for_ncer(text: str) -> str:
    """Canonical normalized text used for N-CER (project policy)."""
    return normalize_for_comparison(text, fold_digits=True)


def normalized_cer(reference: str, hypothesis: str) -> float:
    return project_cer(normalize_for_ncer(reference), normalize_for_ncer(hypothesis))


def _align(reference: list[str], hypothesis: list[str]) -> list[tuple[str, int, int]]:
    """Return aligned ops ``[(op, ref_idx, hyp_idx)]`` with deterministic ties.

    ``ref_idx``/``hyp_idx`` are ``-1`` when the token is absent on that side.
    """
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    trace = [[_TRACE_SUB] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = i
        trace[i][0] = _TRACE_DEL
    for j in range(1, cols):
        cost[0][j] = j
        trace[0][j] = _TRACE_INS
    for i in range(1, rows):
        ref_item = reference[i - 1]
        for j in range(1, cols):
            sub_cost = cost[i - 1][j - 1] + (ref_item != hypothesis[j - 1])
            del_cost = cost[i - 1][j] + 1
            ins_cost = cost[i][j - 1] + 1
            if sub_cost <= del_cost and sub_cost <= ins_cost:
                cost[i][j] = sub_cost
                trace[i][j] = _TRACE_SUB
            elif del_cost <= ins_cost:
                cost[i][j] = del_cost
                trace[i][j] = _TRACE_DEL
            else:
                cost[i][j] = ins_cost
                trace[i][j] = _TRACE_INS
    ops: list[tuple[str, int, int]] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            op = trace[i][j]
            if op == _TRACE_SUB:
                kind = _OP_MATCH if reference[i - 1] == hypothesis[j - 1] else _OP_SUB
                ops.append((kind, i - 1, j - 1))
                i, j = i - 1, j - 1
            elif op == _TRACE_DEL:
                ops.append((_OP_DEL, i - 1, -1))
                i -= 1
            else:
                ops.append((_OP_INS, -1, j - 1))
                j -= 1
        elif i > 0:
            ops.append((_OP_DEL, i - 1, -1))
            i -= 1
        else:
            ops.append((_OP_INS, -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _context(reference: list[str], index: int, radius: int = _CONTEXT_RADIUS) -> str:
    start = max(0, index - radius)
    end = min(len(reference), index + radius + 1)
    return "".join(reference[start:end])


def _char_records(
    reference: str, hypothesis: str, *, normalized: bool
) -> tuple[ErrorRecord, ...]:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    records: list[ErrorRecord] = []
    for op, ref_idx, hyp_idx in _align(ref_chars, hyp_chars):
        if op == _OP_MATCH:
            records.append(
                ErrorRecord(
                    level="char",
                    operation=_OP_MATCH,
                    error_type="match",
                    gt=ref_chars[ref_idx],
                    predicted=hyp_chars[hyp_idx],
                    position=ref_idx,
                    predicted_position=hyp_idx,
                    word_index=None,
                    context="",
                    normalized=normalized,
                )
            )
        elif op == _OP_SUB:
            gt_char = ref_chars[ref_idx]
            pred_char = hyp_chars[hyp_idx]
            records.append(
                ErrorRecord(
                    level="char",
                    operation=_OP_SUB,
                    error_type=classify_char_error(gt_char, pred_char),
                    gt=gt_char,
                    predicted=pred_char,
                    position=ref_idx,
                    predicted_position=hyp_idx,
                    word_index=None,
                    context=_context(ref_chars, ref_idx),
                    normalized=normalized,
                )
            )
        elif op == _OP_DEL:
            gt_char = ref_chars[ref_idx]
            records.append(
                ErrorRecord(
                    level="char",
                    operation=_OP_DEL,
                    error_type=classify_char_error(gt_char, None),
                    gt=gt_char,
                    predicted=None,
                    position=ref_idx,
                    predicted_position=None,
                    word_index=None,
                    context=_context(ref_chars, ref_idx),
                    normalized=normalized,
                )
            )
        else:
            pred_char = hyp_chars[hyp_idx]
            records.append(
                ErrorRecord(
                    level="char",
                    operation=_OP_INS,
                    error_type=classify_char_error(None, pred_char),
                    gt=None,
                    predicted=pred_char,
                    position=ref_idx,  # -1: no gt counterpart
                    predicted_position=hyp_idx,
                    word_index=None,
                    context=_context(hyp_chars, hyp_idx),
                    normalized=normalized,
                )
            )
    return tuple(records)


def _word_char_index(reference: str) -> list[tuple[int, int]]:
    """Map each whitespace-delimited word to its (start, end) char range."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, ch in enumerate(reference):
        if ch.isspace():
            if start is not None:
                spans.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append((start, len(reference)))
    return spans


def _word_records(reference: str, hypothesis: str) -> tuple[ErrorRecord, ...]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    spans = _word_char_index(reference)
    records: list[ErrorRecord] = []
    for op, ref_idx, hyp_idx in _align(ref_words, hyp_words):
        if op == _OP_MATCH:
            records.append(
                ErrorRecord(
                    level="word",
                    operation=_OP_MATCH,
                    error_type="match",
                    gt=ref_words[ref_idx],
                    predicted=hyp_words[hyp_idx],
                    position=spans[ref_idx][0] if ref_idx < len(spans) else ref_idx,
                    predicted_position=None,
                    word_index=ref_idx,
                    context="",
                    normalized=False,
                )
            )
        elif op == _OP_SUB:
            gt_word = ref_words[ref_idx]
            pred_word = hyp_words[hyp_idx]
            records.append(
                ErrorRecord(
                    level="word",
                    operation=_OP_SUB,
                    error_type=classify_word_error(gt_word, pred_word),
                    gt=gt_word,
                    predicted=pred_word,
                    position=spans[ref_idx][0] if ref_idx < len(spans) else ref_idx,
                    predicted_position=None,
                    word_index=ref_idx,
                    context=gt_word,
                    normalized=False,
                )
            )
        elif op == _OP_DEL:
            gt_word = ref_words[ref_idx]
            records.append(
                ErrorRecord(
                    level="word",
                    operation=_OP_DEL,
                    error_type="missing_word",
                    gt=gt_word,
                    predicted=None,
                    position=spans[ref_idx][0] if ref_idx < len(spans) else ref_idx,
                    predicted_position=None,
                    word_index=ref_idx,
                    context=gt_word,
                    normalized=False,
                )
            )
        else:
            records.append(
                ErrorRecord(
                    level="word",
                    operation=_OP_INS,
                    error_type="extra_word",
                    gt=None,
                    predicted=hyp_words[hyp_idx],
                    position=-1,
                    predicted_position=None,
                    word_index=hyp_idx,
                    context=hyp_words[hyp_idx],
                    normalized=False,
                )
            )
    return tuple(records)


def _error_type_counts(records: Iterable[ErrorRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.operation == _OP_MATCH:
            continue
        counts[record.error_type] = counts.get(record.error_type, 0) + 1
    return dict(sorted(counts.items()))


def _substitution_pairs(
    records: Iterable[ErrorRecord],
) -> tuple[tuple[str, str, int], ...]:
    pairs: dict[tuple[str, str], int] = {}
    for record in records:
        if record.operation == _OP_SUB and record.gt and record.predicted:
            key = (record.gt, record.predicted)
            pairs[key] = pairs.get(key, 0) + 1
    return tuple(
        (gt, pred, count)
        for (gt, pred), count in sorted(
            pairs.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    )


def _rates(counts: dict[str, int], total_chars: int) -> dict[str, float]:
    if total_chars <= 0:
        return {key: 0.0 for key in counts}
    return {key: round(count / total_chars, 6) for key, count in counts.items()}


def analyze_sample(sample: OCRSample) -> ErrorAnalysis:
    """Analyze one OCR sample: metrics, alignments, and Arabic-aware errors."""
    reference = sample.ground_truth
    hypothesis = sample.prediction
    ncer = normalized_cer(reference, hypothesis)
    char_records = _char_records(reference, hypothesis, normalized=False)
    # Second, normalized alignment pass: catches errors invisible to the raw
    # alignment (e.g. diacritic-only or alef-variant mismatches) and labels
    # them with ``normalized=True`` so the UI can distinguish the two layers.
    norm_reference = normalize_for_ncer(reference)
    norm_hypothesis = normalize_for_ncer(hypothesis)
    normalized_records = (
        _char_records(norm_reference, norm_hypothesis, normalized=True)
        if norm_reference != norm_hypothesis
        else ()
    )
    word_records = _word_records(reference, hypothesis)

    char_errors = [r for r in char_records if r.operation != _OP_MATCH]
    word_errors = [r for r in word_records if r.operation != _OP_MATCH]
    counts = dict(sorted(_error_type_counts(char_records).items()))
    # Fold word-level structural categories into the summary counts as well
    for record in word_errors:
        counts[record.error_type] = counts.get(record.error_type, 0) + 1

    return ErrorAnalysis(
        page_id=sample.page_id or sample.sample_id,
        model_id=sample.model_id,
        run_id=sample.run_id,
        dataset_id=sample.dataset_id,
        cer=project_cer(reference, hypothesis),
        wer=project_wer(reference, hypothesis),
        normalized_cer=ncer,
        exact_match=reference == hypothesis,
        character_count=len(reference),
        word_count=len(reference.split()),
        char_records=char_records,
        word_records=word_records,
        error_type_counts=counts,
        error_type_rates=_rates(counts, len(reference)),
        substitutions=_substitution_pairs(char_records),
        metadata={
            "normalized_char_errors": [r.to_dict() for r in normalized_records],
            "word_error_count": len(word_errors),
            "char_error_count": len(char_errors),
        },
    )


__all__ = [
    "analyze_sample",
    "normalized_cer",
    "normalize_for_ncer",
]
