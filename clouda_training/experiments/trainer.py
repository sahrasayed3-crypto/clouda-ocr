from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from clouda_data.evaluation.cer import cer
from clouda_data.evaluation.wer import wer

from .checkpoints import CheckpointManager
from .config import ExperimentConfig
from .metrics import MetricLogger


@dataclass(frozen=True)
class TrainerResult:
    final_step: int
    resumed_from_step: int
    duration_seconds: float


class Trainer(Protocol):
    def train(self, *, start_step: int = 0) -> TrainerResult: ...


class MockTrainer:
    """Deterministic CPU-only trainer used to exercise the full lifecycle."""

    def __init__(
        self,
        config: ExperimentConfig,
        metrics: MetricLogger,
        checkpoints: CheckpointManager,
        *,
        fail_at_step: int | None = None,
        interrupt_at_step: int | None = None,
        data_loader: Any | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self.checkpoints = checkpoints
        self.fail_at_step = fail_at_step
        self.interrupt_at_step = interrupt_at_step
        self.data_loader = data_loader
        self._data_batches: Iterator[dict[str, Any]] | None = None

    def _next_training_batch(self) -> dict[str, Any] | None:
        if self.data_loader is None:
            return None
        if self._data_batches is None:
            self._data_batches = iter(self.data_loader.iter_batches())
        try:
            return next(self._data_batches)
        except StopIteration:
            self._data_batches = iter(self.data_loader.iter_batches())
            return next(self._data_batches)

    def train(self, *, start_step: int = 0) -> TrainerResult:
        started = time.monotonic()
        maximum = self.config.training.max_steps or self.config.training.epochs * 5
        for step in range(start_step + 1, maximum + 1):
            if step == self.fail_at_step:
                raise RuntimeError(f"Mock trainer injected failure at step {step}")
            if step == self.interrupt_at_step:
                raise KeyboardInterrupt()
            training_batch = self._next_training_batch()
            epoch = step / maximum * self.config.training.epochs
            noise = random.Random(self.config.training.seed * 1_000_003 + step).random()
            loss = round(1.0 / (step + 1) + noise * 0.01, 8)
            current = {
                "loss": loss,
                "learning_rate": self.config.training.learning_rate,
            }
            if step % self.config.tracking.log_steps == 0:
                for name, value in current.items():
                    self.metrics.append(
                        step=step,
                        epoch=epoch,
                        split="train",
                        metric_name=name,
                        value=value,
                    )
            if (
                self.config.evaluation.enabled
                and step % self.config.evaluation.eval_steps == 0
            ):
                reference = "arabic ocr fixture"
                hypothesis = reference if step == maximum else "arabic ocr fixtur"
                available = {
                    "cer": cer(reference, hypothesis),
                    "wer": wer(reference, hypothesis),
                }
                for name in self.config.evaluation.metrics:
                    if name in available:
                        value = float(available[name])
                        current[name] = value
                        self.metrics.append(
                            step=step,
                            epoch=epoch,
                            split=self.config.evaluation.eval_split,
                            metric_name=name,
                            value=value,
                        )
            if (
                self.config.checkpoint.save_strategy == "steps"
                and step % self.config.checkpoint.save_steps == 0
            ):
                checkpoint_state: dict[str, Any] | None = None
                if self.data_loader is not None:
                    from clouda_data.training_data.checkpoint_bridge import (
                        LoaderCheckpointHook,
                    )

                    checkpoint_state = LoaderCheckpointHook(
                        self.data_loader
                    ).attach_to_state(
                        {
                            "batch_sample_ids": list(
                                training_batch["sample_ids"] if training_batch else []
                            )
                        }
                    )
                self.checkpoints.save(
                    step=step,
                    epoch=epoch,
                    metrics=current,
                    state=checkpoint_state,
                )
        return TrainerResult(maximum, start_step, time.monotonic() - started)
