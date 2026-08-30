from __future__ import annotations

from typing import Protocol

from clouda_training.config.models import TrainingConfig


class TrainingExecutionDisabled(RuntimeError):
    """Raised when an orchestration-only build is asked to train a model."""


class Trainer(Protocol):
    def train(self, config: TrainingConfig) -> None: ...


def training_disabled(*_: object, **__: object) -> None:
    raise TrainingExecutionDisabled(
        "Model training is intentionally disabled in the merged baseline."
    )
