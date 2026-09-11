"""Mock backend: adapts the existing deterministic MockTrainer.

Preserves the legacy dry-run behaviour exactly while exposing it through the
unified :class:`TrainerBackend` seam used by the experiment framework.
"""

from __future__ import annotations

from typing import Any

from clouda_training.experiments.config import ExperimentConfig
from clouda_training.experiments.metrics import MetricLogger
from clouda_training.experiments.trainer import MockTrainer
from clouda_training.runtime.backend import TrainingState


class MockTrainerBackend:
    """Backend adapter around :class:`MockTrainer` (no real optimization)."""

    def __init__(
        self,
        config: ExperimentConfig,
        metrics: MetricLogger,
        checkpoints: Any,
        *,
        fail_at_step: int | None = None,
        interrupt_at_step: int | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self.checkpoints = checkpoints
        self.fail_at_step = fail_at_step
        self.interrupt_at_step = interrupt_at_step
        self._trainer = MockTrainer(
            config,
            metrics,
            checkpoints,
            fail_at_step=fail_at_step,
            interrupt_at_step=interrupt_at_step,
        )

    def train(self, *, start_step: int = 0) -> Any:
        return self._trainer.train(start_step=start_step)

    def capture_state(self) -> TrainingState:
        # MockTrainer keeps no state beyond the step; the framework's own
        # checkpoint metadata already carries it.
        return TrainingState(step=0, epoch=0.0, payload={"mock": True})

    def restore_state(self, state: TrainingState) -> None:
        # Nothing to restore for the stateless mock trainer.
        return None
