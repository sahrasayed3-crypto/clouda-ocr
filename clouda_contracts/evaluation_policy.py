from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MetricScale(StrEnum):
    RATIO = "ratio_0_to_1"
    PERCENT = "percent_0_to_100"


@dataclass(frozen=True)
class EvaluationPolicy:
    policy_id: str
    normalization: str
    error_metric_scale: MetricScale
    heuristic_quality_scale: MetricScale | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.policy_id or not self.normalization:
            raise ValueError("Evaluation policy identifiers cannot be blank.")
        if self.schema_version != 1:
            raise ValueError("Unsupported evaluation policy schema version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "normalization": self.normalization,
            "error_metric_scale": self.error_metric_scale.value,
            "heuristic_quality_scale": (
                self.heuristic_quality_scale.value
                if self.heuristic_quality_scale is not None
                else None
            ),
        }


DATA_FOUNDATION_POLICY = EvaluationPolicy(
    policy_id="data_foundation_v1",
    normalization="comparison_arabic_fold_digits",
    error_metric_scale=MetricScale.RATIO,
    heuristic_quality_scale=None,
)

RUNTIME_POLICY = EvaluationPolicy(
    policy_id="runtime_v1",
    normalization="runtime_ground_truth_exact",
    error_metric_scale=MetricScale.PERCENT,
    heuristic_quality_scale=MetricScale.PERCENT,
)


def convert_error_metric(
    value: float,
    *,
    source: MetricScale,
    target: MetricScale,
) -> float:
    if value < 0:
        raise ValueError("Error metrics cannot be negative.")
    if source is target:
        return value
    if source is MetricScale.RATIO and target is MetricScale.PERCENT:
        return value * 100.0
    return value / 100.0
