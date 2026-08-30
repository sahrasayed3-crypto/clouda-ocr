"""License-gated, framework-neutral training orchestration."""

from .planner import TrainingPlan, plan_training

__all__ = ["TrainingPlan", "plan_training"]
