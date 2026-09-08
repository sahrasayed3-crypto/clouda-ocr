"""Preparation configuration: explicit defaults, safely committable."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .discovery import IMAGE_EXTENSIONS, RECORD_EXTENSIONS, TEXT_EXTENSIONS
from .normalize import NormalizationPolicy
from .splitting import DEFAULT_RATIOS

PREPARATION_CONFIG_VERSION = "clouda.pretraining.config.v1"


class PreparationConfigError(ValueError):
    """Raised when a preparation configuration is invalid."""


@dataclass(frozen=True)
class PreparationConfig:
    """Everything the preparation pipeline needs, with safe defaults."""

    allowed_image_extensions: frozenset[str] = frozenset(IMAGE_EXTENSIONS)
    allowed_record_extensions: frozenset[str] = frozenset(RECORD_EXTENSIONS)
    allowed_text_extensions: frozenset[str] = frozenset(TEXT_EXTENSIONS)
    text_without_image_policy: str = "exclude"  # or "keep"
    normalization: NormalizationPolicy = field(default_factory=NormalizationPolicy)
    split_seed: int = 20260722
    split_ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    holdout_enabled: bool = True
    min_width: int = 8
    min_height: int = 8
    max_pixels: int = 100_000_000
    max_text_chars: int = 100_000
    require_text: bool = True
    require_image: bool = True
    hash_cache_enabled: bool = True
    workers: int = 1
    exporter: str = "jsonl"
    include_holdout_in_export: bool = False
    include_duplicates_in_export: bool = False
    include_raw_text_in_export: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normalization"] = asdict(self.normalization)
        for key in (
            "allowed_image_extensions",
            "allowed_record_extensions",
            "allowed_text_extensions",
        ):
            payload[key] = sorted(payload[key])
        return payload

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PreparationConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise PreparationConfigError(
                f"Unknown preparation config fields: {sorted(unknown)}"
            )
        kwargs = dict(data)
        normalization_data = kwargs.pop("normalization", {})
        kwargs["normalization"] = NormalizationPolicy(**normalization_data)
        for key in (
            "allowed_image_extensions",
            "allowed_record_extensions",
            "allowed_text_extensions",
        ):
            if key in kwargs:
                kwargs[key] = frozenset(kwargs[key])
        if "split_ratios" in kwargs:
            kwargs["split_ratios"] = dict(kwargs["split_ratios"])
        config = cls(**kwargs)
        _validate(config)
        return config


def _validate(config: PreparationConfig) -> None:
    if config.text_without_image_policy not in {"exclude", "keep"}:
        raise PreparationConfigError("text_without_image_policy must be exclude|keep")
    if set(config.split_ratios) != {"train", "validation", "test", "holdout"}:
        raise PreparationConfigError(
            "split_ratios must define train, validation, test, holdout"
        )
    if abs(sum(config.split_ratios.values()) - 1.0) > 1e-9:
        raise PreparationConfigError("split_ratios must sum to 1.0")
    if config.split_seed < 0:
        raise PreparationConfigError("split_seed cannot be negative")
    if config.workers < 1:
        raise PreparationConfigError("workers must be at least 1")
    if config.min_width < 1 or config.min_height < 1:
        raise PreparationConfigError("min_width/min_height must be positive")
    if config.max_text_chars < 1:
        raise PreparationConfigError("max_text_chars must be positive")
    if config.exporter != "jsonl":
        raise PreparationConfigError("exporter must be 'jsonl' (only built-in)")


def load_preparation_config(path: str | Path | None = None) -> PreparationConfig:
    if path is None:
        return PreparationConfig()
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - PyYAML ships in data extra
            raise RuntimeError("PyYAML is required for YAML configs.") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise PreparationConfigError("Config file must contain a mapping.")
    return PreparationConfig.from_mapping(data)
