from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates its strict schema."""


EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    parent_experiment: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ModelSection:
    model_id: str
    revision: str = "unresolved"
    model_family: str = "generic"
    adapter_type: str = "mock"
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    precision: str = "float32"


@dataclass(frozen=True)
class DatasetSection:
    dataset_id: str
    dataset_version: str
    manifest_path: Path
    split: str = "train"
    sample_limit: int | None = None
    preprocessing_version: str = "unversioned"


@dataclass(frozen=True)
class TrainingSection:
    seed: int = 20260723
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    scheduler: str = "linear"
    max_grad_norm: float = 1.0
    mixed_precision: bool = False
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class CheckpointSection:
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    resume_from: str | None = None
    keep_best: bool = True
    metric_for_best: str = "loss"
    greater_is_better: bool = False


@dataclass(frozen=True)
class EvaluationSection:
    enabled: bool = True
    eval_split: str = "validation"
    eval_steps: int = 100
    metrics: tuple[str, ...] = ("cer", "wer")
    max_samples: int | None = None


@dataclass(frozen=True)
class RuntimeSection:
    device: str = "cpu"
    num_workers: int = 0
    output_root: Path = Path("runs")
    dry_run: bool = True
    offline: bool = True
    deterministic: bool = True


@dataclass(frozen=True)
class TrackingSection:
    enabled: bool = True
    backend: str = "jsonl"
    log_steps: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection
    model: ModelSection
    dataset: DatasetSection
    training: TrainingSection = field(default_factory=TrainingSection)
    checkpoint: CheckpointSection = field(default_factory=CheckpointSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    tracking: TrackingSection = field(default_factory=TrackingSection)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


T = TypeVar("T")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _strict_section(cls: type[T], raw: object, name: str, *, path_base: Path) -> T:
    if not isinstance(raw, dict):
        raise ConfigError(f"{name} must be an object")
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{name} has unknown field(s): {sorted(unknown)}")
    values = dict(raw)
    tuple_fields = {"tags", "metrics"}
    for key in tuple_fields & values.keys():
        if not isinstance(values[key], list) or not all(
            isinstance(item, str) for item in values[key]
        ):
            raise ConfigError(f"{name}.{key} must be a list of strings")
        values[key] = tuple(values[key])
    if "manifest_path" in values:
        candidate = Path(str(values["manifest_path"])).expanduser()
        values["manifest_path"] = (
            candidate if candidate.is_absolute() else path_base / candidate
        ).resolve()
    if "output_root" in values:
        candidate = Path(str(values["output_root"])).expanduser()
        values["output_root"] = (
            candidate if candidate.is_absolute() else path_base / candidate
        ).resolve()
    try:
        result = cls(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid {name}: {exc}") from exc
    _validate_types(result, name)
    return result


def _validate_types(section: object, name: str) -> None:
    annotations = get_type_hints(section.__class__)
    for key, expected in annotations.items():
        value = getattr(section, key)
        origin = str(expected)
        if expected is bool and not isinstance(value, bool):
            raise ConfigError(f"{name}.{key} must be a boolean")
        if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(f"{name}.{key} must be an integer")
        if expected is float and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ConfigError(f"{name}.{key} must be a number")
        if expected is str and not isinstance(value, str):
            raise ConfigError(f"{name}.{key} must be a string")
        if (
            "int | None" in origin
            and value is not None
            and (isinstance(value, bool) or not isinstance(value, int))
        ):
            raise ConfigError(f"{name}.{key} must be an integer or null")


def _validate(config: ExperimentConfig) -> None:
    if config.schema_version != 1:
        raise ConfigError("Unsupported experiment schema_version")
    for label, value in (
        ("experiment.name", config.experiment.name),
        ("model.model_id", config.model.model_id),
        ("dataset.dataset_id", config.dataset.dataset_id),
        ("dataset.dataset_version", config.dataset.dataset_version),
    ):
        if not value.strip():
            raise ConfigError(f"{label} cannot be blank")
    if not EXPERIMENT_NAME_RE.fullmatch(config.experiment.name):
        raise ConfigError(
            "experiment.name must be a filesystem-safe identifier containing "
            "only letters, digits, dot, underscore, and hyphen"
        )
    positive: dict[str, int] = {
        "training.epochs": config.training.epochs,
        "training.batch_size": config.training.batch_size,
        "training.gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "checkpoint.save_steps": config.checkpoint.save_steps,
        "checkpoint.save_total_limit": config.checkpoint.save_total_limit,
        "evaluation.eval_steps": config.evaluation.eval_steps,
        "tracking.log_steps": config.tracking.log_steps,
    }
    if config.training.max_steps is not None:
        positive["training.max_steps"] = config.training.max_steps
    if config.dataset.sample_limit is not None:
        positive["dataset.sample_limit"] = config.dataset.sample_limit
    if config.evaluation.max_samples is not None:
        positive["evaluation.max_samples"] = config.evaluation.max_samples
    for numeric_label, numeric_value in positive.items():
        if numeric_value < 1:
            raise ConfigError(f"{numeric_label} must be positive")
    if config.training.learning_rate <= 0:
        raise ConfigError("training.learning_rate must be positive")
    if config.training.weight_decay < 0:
        raise ConfigError("training.weight_decay cannot be negative")
    if config.checkpoint.save_strategy not in {"steps", "epoch", "none"}:
        raise ConfigError("checkpoint.save_strategy must be steps, epoch, or none")
    if config.tracking.backend != "jsonl":
        raise ConfigError("Only the jsonl tracking backend is available offline")
    if config.evaluation.eval_split not in {"train", "validation", "test"}:
        raise ConfigError(
            "evaluation.eval_split must be train, validation, or test; "
            "protected holdout evaluation is forbidden"
        )


def _apply_override(payload: dict[str, Any], expression: str) -> None:
    dotted = expression.split("=", 1)[0]
    if "=" not in expression or dotted.count(".") != 1:
        raise ConfigError(f"Invalid override {expression!r}; use section.field=value")
    dotted, text = expression.split("=", 1)
    section, key = dotted.split(".", 1)
    if section not in payload or not isinstance(payload[section], dict):
        raise ConfigError(f"Override references unknown field: {dotted}")
    known_types = {
        name: set(cast(Any, cls).__dataclass_fields__)
        for name, cls in _SECTION_TYPES.items()
    }
    if section not in known_types or key not in known_types[section]:
        raise ConfigError(f"Override references unknown field: {dotted}")
    payload[section][key] = yaml.safe_load(text)


_SECTION_TYPES = {
    "experiment": ExperimentSection,
    "model": ModelSection,
    "dataset": DatasetSection,
    "training": TrainingSection,
    "checkpoint": CheckpointSection,
    "evaluation": EvaluationSection,
    "runtime": RuntimeSection,
    "tracking": TrackingSection,
}


def config_from_dict(payload: dict[str, Any], *, path_base: Path) -> ExperimentConfig:
    allowed = {"schema_version", *_SECTION_TYPES}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(f"Configuration has unknown field(s): {sorted(unknown)}")
    missing = {"experiment", "model", "dataset"} - set(payload)
    if missing:
        raise ConfigError(
            f"Configuration missing required section(s): {sorted(missing)}"
        )
    sections: dict[str, Any] = {
        name: _strict_section(cls, payload.get(name, {}), name, path_base=path_base)
        for name, cls in _SECTION_TYPES.items()
    }
    config = ExperimentConfig(
        **sections, schema_version=payload.get("schema_version", 1)  # type: ignore[arg-type]
    )
    _validate(config)
    return config


def load_experiment_config(
    path: str | Path, *, overrides: list[str] | None = None
) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to load configuration: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Experiment configuration must be an object")
    for expression in overrides or []:
        _apply_override(payload, expression)
    return config_from_dict(payload, path_base=config_path.parent)
