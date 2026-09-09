"""Shim giving the vendored System A render modules their dependencies.

`weighted_choice`/`weighted_choice_entry` are verbatim from
ocrbench/config.py. `LayoutConfigView` exposes the `.layout` attribute that
ocrbench.layouts.pick_layout expects from a BenchmarkConfig without dragging
in the full benchmark config loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class ConfigError(ValueError):
    """Raised when a configuration mapping is missing required structure."""


def weighted_choice(
    options: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> Any:
    """Verbatim ocrbench.config.weighted_choice."""
    if not options:
        raise ConfigError("cannot choose from an empty option pool")
    weights = np.array([float(o.get("weight", 1.0)) for o in options], dtype=np.float64)
    if weights.sum() <= 0:
        raise ConfigError("option pool has non-positive total weight")
    weights /= weights.sum()
    idx = int(rng.choice(len(options), p=weights))
    entry = options[idx]
    if "value" not in entry:
        raise ConfigError(f"option pool entry lacks a 'value' key: {entry!r}")
    return entry["value"]


def weighted_choice_entry(
    options: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> Mapping[str, Any]:
    """Verbatim ocrbench.config.weighted_choice_entry."""
    if not options:
        raise ConfigError("cannot choose from an empty option pool")
    weights = np.array([float(o.get("weight", 1.0)) for o in options], dtype=np.float64)
    if weights.sum() <= 0:
        raise ConfigError("option pool has non-positive total weight")
    weights /= weights.sum()
    return options[int(rng.choice(len(options), p=weights))]


@dataclass
class LayoutConfigView:
    """Minimal stand-in for BenchmarkConfig: only `.layout` is required."""

    layout: Mapping[str, Any] = field(default_factory=dict)
