from __future__ import annotations

import os
from dataclasses import dataclass

from .enums import TeacherPipelineMode


class TeacherPipelineConfigurationError(RuntimeError):
    pass


def _strict_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise TeacherPipelineConfigurationError(f"invalid_{name.lower()}")


def _strict_ratio(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise TeacherPipelineConfigurationError(f"invalid_{name.lower()}") from exc
    if not 0.0 <= value <= 1.0:
        raise TeacherPipelineConfigurationError(f"invalid_{name.lower()}")
    return value


@dataclass(frozen=True)
class TeacherPipelineConfig:
    mode: TeacherPipelineMode = TeacherPipelineMode.DISABLED
    live_dispatch_allowed: bool = False
    training_enabled: bool = False
    cloud_upload_enabled: bool = False
    acceptance_policy_version: str = "teacher-acceptance-v1"
    character_agreement_min: float = 0.97
    word_agreement_min: float = 0.95
    line_disagreement_max: float = 0.02
    teacher_confidence_min: float = 0.85
    judge_confidence_min: float = 0.90
    source_image_coverage_min: float = 0.98
    uncertain_span_ratio_max: float = 0.01

    @classmethod
    def from_env(cls) -> "TeacherPipelineConfig":
        raw_mode = os.getenv("TEACHER_PIPELINE_MODE", "disabled").strip().lower()
        try:
            mode = TeacherPipelineMode(raw_mode)
        except ValueError as exc:
            raise TeacherPipelineConfigurationError(
                "invalid_teacher_pipeline_mode"
            ) from exc
        config = cls(
            mode=mode,
            live_dispatch_allowed=_strict_bool("TEACHER_LIVE_DISPATCH_ALLOWED", False),
            training_enabled=_strict_bool("TRAINING_ENABLED", False),
            cloud_upload_enabled=_strict_bool("CLOUD_UPLOAD_ENABLED", False),
            acceptance_policy_version=(
                os.getenv(
                    "TEACHER_ACCEPTANCE_POLICY_VERSION", "teacher-acceptance-v1"
                ).strip()
                or "teacher-acceptance-v1"
            ),
            character_agreement_min=_strict_ratio(
                "TEACHER_CHARACTER_AGREEMENT_MIN", 0.97
            ),
            word_agreement_min=_strict_ratio("TEACHER_WORD_AGREEMENT_MIN", 0.95),
            line_disagreement_max=_strict_ratio("TEACHER_LINE_DISAGREEMENT_MAX", 0.02),
            teacher_confidence_min=_strict_ratio("TEACHER_CONFIDENCE_MIN", 0.85),
            judge_confidence_min=_strict_ratio("TEACHER_JUDGE_CONFIDENCE_MIN", 0.90),
            source_image_coverage_min=_strict_ratio(
                "TEACHER_SOURCE_IMAGE_COVERAGE_MIN", 0.98
            ),
            uncertain_span_ratio_max=_strict_ratio(
                "TEACHER_UNCERTAIN_SPAN_RATIO_MAX", 0.01
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.live_dispatch_allowed:
            raise TeacherPipelineConfigurationError("live_dispatch_locked_for_phase3")
        if self.training_enabled:
            raise TeacherPipelineConfigurationError("training_locked_for_phase3")
        if self.cloud_upload_enabled:
            raise TeacherPipelineConfigurationError("cloud_upload_locked_for_phase3")
        if self.mode is TeacherPipelineMode.LIVE:
            raise TeacherPipelineConfigurationError("live_teacher_mode_not_implemented")

    @property
    def network_permitted(self) -> bool:
        return False
