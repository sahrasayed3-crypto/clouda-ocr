from __future__ import annotations

import pytest

from clouda_contracts.evaluation_policy import (
    DATA_FOUNDATION_POLICY,
    RUNTIME_POLICY,
    MetricScale,
    convert_error_metric,
)


def test_policies_keep_existing_metric_semantics_explicit() -> None:
    assert DATA_FOUNDATION_POLICY.error_metric_scale is MetricScale.RATIO
    assert DATA_FOUNDATION_POLICY.heuristic_quality_scale is None
    assert RUNTIME_POLICY.error_metric_scale is MetricScale.PERCENT
    assert RUNTIME_POLICY.heuristic_quality_scale is MetricScale.PERCENT


def test_error_metric_conversion_is_explicit_and_reversible() -> None:
    percentage = convert_error_metric(
        0.125,
        source=MetricScale.RATIO,
        target=MetricScale.PERCENT,
    )
    assert percentage == 12.5
    assert (
        convert_error_metric(
            percentage,
            source=MetricScale.PERCENT,
            target=MetricScale.RATIO,
        )
        == 0.125
    )


def test_negative_error_metric_is_rejected() -> None:
    with pytest.raises(ValueError):
        convert_error_metric(
            -1,
            source=MetricScale.PERCENT,
            target=MetricScale.RATIO,
        )
