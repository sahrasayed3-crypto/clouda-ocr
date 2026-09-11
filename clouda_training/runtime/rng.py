"""Deterministic RNG capture and restoration for training backends.

Saves/restores python, numpy and torch RNG states so an interrupted run can
be resumed with bit-identical randomness.
"""

from __future__ import annotations

import random
from typing import Any


def capture_rng_state() -> dict[str, Any]:
    """Capture all available RNG states."""

    state: dict[str, Any] = {
        "python": random.getstate(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    try:
        import numpy as _np

        state["numpy"] = _np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states previously captured by :func:`capture_rng_state`."""

    if "python" in state:
        random.setstate(_coerce_python_state(state["python"]))
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(_coerce_torch_state(state["torch"]))
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(_coerce_torch_cuda_state(state["torch_cuda"]))
    except ImportError:
        pass
    try:
        import numpy as _np

        if "numpy" in state:
            _np.random.set_state(_coerce_numpy_state(state["numpy"]))
    except ImportError:
        pass


def _coerce_python_state(value: Any) -> tuple:
    version, internal, gauss = value
    return (int(version), tuple(int(x) for x in internal), gauss)


def _coerce_torch_state(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.clone()
    return torch.tensor(value, dtype=torch.uint8)


def _coerce_torch_cuda_state(value: Any) -> list:
    import torch

    out = []
    for item in value:
        out.append(
            item.clone()
            if isinstance(item, torch.Tensor)
            else torch.tensor(item, dtype=torch.uint8)
        )
    return out


def _coerce_numpy_state(value: Any) -> tuple:
    import numpy as _np

    if isinstance(value, tuple) and isinstance(value[1], _np.ndarray):
        return value
    return value
