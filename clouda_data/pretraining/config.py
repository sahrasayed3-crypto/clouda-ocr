"""Preparation configuration: explicit defaults, safely committable."""

from __future__ import annotations

import json
import hashlib
import math
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

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PreparationConfig:
        data = dict(data)
        version = data.pop("_schema_version", PREPARATION_CONFIG_VERSION)
        if version != PREPARATION_CONFIG_VERSION:
            raise PreparationConfigError(
                f"Unsupported preparation config version: {version!r}"
            )
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
    extension_groups = (
        (config.allowed_image_extensions, IMAGE_EXTENSIONS, "image"),
        (config.allowed_record_extensions, RECORD_EXTENSIONS, "record"),
        (config.allowed_text_extensions, TEXT_EXTENSIONS, "text"),
    )
    for configured, supported, label in extension_groups:
        if not all(isinstance(value, str) for value in configured):
            raise PreparationConfigError(f"allowed_{label}_extensions must be strings")
        unsupported = set(configured) - supported
        if unsupported:
            raise PreparationConfigError(
                f"Unsupported {label} extensions: {sorted(unsupported)}"
            )
    if set(config.split_ratios) != {"train", "validation", "test", "holdout"}:
        raise PreparationConfigError(
            "split_ratios must define train, validation, test, holdout"
        )
    ratio_values = list(config.split_ratios.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
        for value in ratio_values
    ):
        raise PreparationConfigError(
            "split_ratios values must be finite numbers between 0 and 1"
        )
    if abs(sum(ratio_values) - 1.0) > 1e-9:
        raise PreparationConfigError("split_ratios must sum to 1.0")
    boolean_fields = (
        "holdout_enabled",
        "require_text",
        "require_image",
        "hash_cache_enabled",
        "include_holdout_in_export",
        "include_duplicates_in_export",
        "include_raw_text_in_export",
    )
    if any(not isinstance(getattr(config, name), bool) for name in boolean_fields):
        raise PreparationConfigError(
            "Boolean preparation options must be JSON/YAML booleans."
        )
    if not config.holdout_enabled and config.split_ratios["holdout"] != 0:
        raise PreparationConfigError(
            "holdout ratio must be zero when holdout_enabled is false"
        )
    integer_fields = (
        "split_seed",
        "workers",
        "min_width",
        "min_height",
        "max_pixels",
        "max_text_chars",
    )
    if any(
        isinstance(getattr(config, name), bool)
        or not isinstance(getattr(config, name), int)
        for name in integer_fields
    ):
        raise PreparationConfigError(
            "Seed, worker, dimension, pixel, and text limits must be integers."
        )
    if config.split_seed < 0:
        raise PreparationConfigError("split_seed cannot be negative")
    if config.workers < 1:
        raise PreparationConfigError("workers must be at least 1")
    if config.min_width < 1 or config.min_height < 1:
        raise PreparationConfigError("min_width/min_height must be positive")
    if config.max_text_chars < 1 or config.max_pixels < 1:
        raise PreparationConfigError("max_text_chars/max_pixels must be positive")
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
