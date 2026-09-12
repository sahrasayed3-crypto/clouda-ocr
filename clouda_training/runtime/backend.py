"""Trainer backend contracts for the Clouda real-training runtime.

This module defines the seam between the Training Experiment Framework
(orchestration, lifecycle, checkpoints, metrics) and the code that actually
executes optimization steps. Two backends ship today:

- ``MockTrainerBackend`` — deterministic CPU-only dry execution (existing
  ``MockTrainer`` behaviour, preserved verbatim).
- ``TorchTrainerBackend`` — genuine PyTorch optimization loop (optional
  dependency, imported lazily).

The framework must be able to run either backend through the same
:class:`TrainerBackend` protocol without knowing which one is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class StepResult:
    """Outcome of a single optimization step."""

    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: float | None = None
    optimizer_step_taken: bool = True


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    steps: int
    mean_loss: float


@dataclass
class TrainingState:
    """Full resumable execution state for a training run.

    ``payload`` holds backend-specific tensors (model/optimizer/scheduler
    states). Serialization is backend-owned; the framework only persists the
    returned blob plus the bookkeeping fields it already tracks.
    """

    step: int
    epoch: float
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrainerBackend(Protocol):
    """Protocol every training backend must satisfy.

    Mirrors the existing ``Trainer`` protocol used by the experiment
    framework (``train(start_step=...) -> TrainerResult``) while exposing
    the finer-grained hooks needed for real optimization:
    checkpoint capture/restore of full execution state and per-step metrics.
    """

    def train(self, *, start_step: int = 0) -> Any:
        """Run training from ``start_step``; returns TrainerResult-compatible object."""
        ...

    def capture_state(self) -> TrainingState:
        """Return full resumable state (model/optimizer/scheduler/RNG)."""
        ...

    def restore_state(self, state: TrainingState) -> None:
        """Restore a previously captured state exactly."""
        ...


def torch_available() -> bool:
    """Return True when PyTorch is importable in this environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def require_torch() -> None:
    """Raise a helpful error when torch is missing."""
    if not torch_available():
        raise ImportError(
            "PyTorch is required for the real training backend but is not "
            "installed. Install the training extra: "
            "pip install clouda-pdf[training-torch]  (or: pip install torch)"
        )
