"""Clouda real-training runtime package.

Imports cleanly without PyTorch; torch-dependent symbols resolve lazily so
the base install never crashes on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clouda_training.runtime.adapter import (
    ModelAdapter,
    SyntheticLinearAdapter,
)
from clouda_training.runtime.backend import (
    StepResult,
    TrainerBackend,
    TrainingState,
    require_torch,
    torch_available,
)
from clouda_training.runtime.mock_backend import MockTrainerBackend

if TYPE_CHECKING:  # pragma: no cover
    from clouda_training.runtime.torch_backend import TorchTrainerBackend

__all__ = [
    "ModelAdapter",
    "MockTrainerBackend",
    "StepResult",
    "SyntheticLinearAdapter",
    "TrainerBackend",
    "TrainingState",
    "TorchTrainerBackend",
    "require_torch",
    "torch_available",
]


def __getattr__(name: str) -> Any:
    if name == "TorchTrainerBackend":
        if not torch_available():
            raise ImportError(
                "PyTorch is required for TorchTrainerBackend. Install with: "
                "pip install clouda-pdf[training-torch]  (or: pip install torch)"
            )
        from clouda_training.runtime.torch_backend import TorchTrainerBackend

        return TorchTrainerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
