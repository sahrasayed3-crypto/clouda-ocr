from __future__ import annotations

import re
import unicodedata
from .config import TeacherPipelineConfig
from .enums import DatasetDecision
from .models import AgreementMetrics, DecisionRecord

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_for_agreement(text: str) -> str:
    value = unicodedata.normalize("NFC", text)
    value = _DIACRITICS.sub("", value).replace("\u0640", "")
    value = re.sub(r"[ \t]+", " ", value)
    return re.sub(r"\s*\n\s*", "\n", value).strip()


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for index, first in enumerate(left, 1):
        current = [index]
        for offset, second in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (first != second),
                )
            )
        previous = current
    return previous[-1]


def _agreement(left: list[str], right: list[str]) -> float:
    denominator = max(len(left), len(right))
    return (
        1.0
        if denominator == 0
        else max(0.0, 1.0 - _edit_distance(left, right) / denominator)
    )


def score_pair(
    left: str,
    right: str,
    *,
    region_coverage: float = 1.0,
    main_text_coverage: float = 1.0,
    margin_coverage: float = 1.0,
    footnote_coverage: float = 1.0,
    uncertain_span_ratio: float = 0.0,
    teacher_confidence: float = 1.0,
    judge_confidence: float = 1.0,
    source_image_coverage: float = 1.0,
    missing_line_indicators: int = 0,
    duplicate_text_indicators: int = 0,
    hallucination_indicators: int = 0,
    reading_order_disagreement: float = 0.0,
) -> AgreementMetrics:
    first = normalize_for_agreement(left)
    second = normalize_for_agreement(right)
    first_lines = first.splitlines()
    second_lines = second.splitlines()
    line_denominator = max(len(first_lines), len(second_lines), 1)
    return AgreementMetrics(
        normalized_character_agreement=_agreement(list(first), list(second)),
        normalized_word_agreement=_agreement(first.split(), second.split()),
        line_count_disagreement=abs(len(first_lines) - len(second_lines))
        / line_denominator,
        missing_line_indicators=missing_line_indicators,
        duplicate_text_indicators=duplicate_text_indicators,
        reading_order_disagreement=reading_order_disagreement,
        region_coverage=region_coverage,
        main_text_coverage=main_text_coverage,
        margin_coverage=margin_coverage,
        footnote_coverage=footnote_coverage,
        uncertain_span_ratio=uncertain_span_ratio,
        teacher_confidence=teacher_confidence,
        judge_confidence=judge_confidence,
        source_image_coverage=source_image_coverage,
        hallucination_indicators=hallucination_indicators,
    )


def decide_dataset(
    metrics: AgreementMetrics,
    config: TeacherPipelineConfig,
    *,
    rights_verified: bool = True,
    input_usable: bool = True,
    provenance_complete: bool = True,
    human_verified: bool = False,
    trusted_reference: bool = False,
    holdout: bool = False,
) -> DecisionRecord:
    reasons: list[str] = []
    if holdout:
        return DecisionRecord(
            DatasetDecision.HOLDOUT,
            config.acceptance_policy_version,
            metrics,
            ("manually_designated_holdout",),
        )
    if not rights_verified:
        reasons.append("rights_not_verified")
    if not input_usable:
        reasons.append("input_unusable")
    if metrics.hallucination_indicators:
        reasons.append("hallucination_detected")
    if not provenance_complete:
        reasons.append("provenance_incomplete")
    if reasons:
        return DecisionRecord(
            DatasetDecision.REJECTED,
            config.acceptance_policy_version,
            metrics,
            tuple(reasons),
        )
    if human_verified or trusted_reference:
        return DecisionRecord(
            DatasetDecision.GOLD,
            config.acceptance_policy_version,
            metrics,
            ("human_verified" if human_verified else "trusted_reference",),
            human_verified=human_verified,
            trusted_reference=trusted_reference,
        )
    gates = {
        "character_agreement_low": metrics.normalized_character_agreement
        < config.character_agreement_min,
        "word_agreement_low": metrics.normalized_word_agreement
        < config.word_agreement_min,
        "line_disagreement_high": metrics.line_count_disagreement
        > config.line_disagreement_max,
        "teacher_confidence_low": metrics.teacher_confidence
        < config.teacher_confidence_min,
        "judge_confidence_low": metrics.judge_confidence < config.judge_confidence_min,
        "source_image_coverage_low": metrics.source_image_coverage
        < config.source_image_coverage_min,
        "uncertainty_high": metrics.uncertain_span_ratio
        > config.uncertain_span_ratio_max,
        "region_coverage_incomplete": metrics.region_coverage < 1.0,
        "main_text_coverage_incomplete": metrics.main_text_coverage < 1.0,
        "margin_coverage_incomplete": metrics.margin_coverage < 1.0,
        "footnote_coverage_incomplete": metrics.footnote_coverage < 1.0,
        "missing_lines": bool(metrics.missing_line_indicators),
        "duplicate_text": bool(metrics.duplicate_text_indicators),
        "reading_order_disagreement": bool(metrics.reading_order_disagreement),
    }
    reasons = [name for name, failed in gates.items() if failed]
    return DecisionRecord(
        DatasetDecision.REVIEW if reasons else DatasetDecision.SILVER,
        config.acceptance_policy_version,
        metrics,
        tuple(reasons or ["silver_policy_satisfied"]),
    )
