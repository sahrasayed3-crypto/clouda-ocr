from __future__ import annotations

import importlib.metadata
import platform
import random
import sys
from typing import Any


def apply_seed(seed: int, *, deterministic: bool) -> dict[str, Any]:
    random.seed(seed)
    applied: dict[str, Any] = {
        "seed": seed,
        "requested_deterministic": deterministic,
        "python_random": True,
        "numpy": False,
        "torch": False,
        "cuda": False,
        "deterministic_algorithms": False,
        "warnings": [],
    }
    try:
        import numpy as np

        np.random.seed(seed)
        applied["numpy"] = True
    except ImportError:
        applied["warnings"].append("NumPy is not installed; its RNG was not seeded.")
    try:
        import torch

        torch.manual_seed(seed)
        applied["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            applied["cuda"] = True
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            applied["deterministic_algorithms"] = True
    except ImportError:
        applied["warnings"].append(
            "PyTorch is not installed; no torch determinism was applied."
        )
    if deterministic:
        applied["warnings"].append(
            "Deterministic settings reduce variability but cannot guarantee bitwise "
            "reproducibility for every future trainer, device, or kernel."
        )
    return applied


def capture_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("clouda-pdf", "numpy", "torch", "PyYAML", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    cuda: dict[str, Any] = {"available": False, "version": None, "gpu_names": []}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["version"] = torch.version.cuda
        if cuda["available"]:
            cuda["gpu_names"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "packages": packages,
        "cuda": cuda,
        "secrets_recorded": False,
    }
