from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when project configuration is invalid."""


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    raw_documents: Path
    raw_pages: Path
    ground_truth: Path
    layout_annotations: Path
    outputs: Path
    previews: Path
    manifests: Path
    cache: Path
    temp: Path
    logs: Path


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int = 20260722
    render_dpi: int = 300
    workers: int = 1
    cpu_limit_percent: int = 90
    batch_size: int = 16
    resume_enabled: bool = True
    retry_limit: int = 3
    disk_free_min_gb: float = 5.0


@dataclass(frozen=True)
class GpuConfig:
    enabled: bool = False
    device: str = "auto"
    memory_limit_gb: float | None = None


@dataclass(frozen=True)
class GroundTruthConfig:
    preserve_original: bool = True
    normalized_copy: bool = True
    unicode_normalization: str = "NFC"
    preserve_line_breaks: bool = True
    preserve_reading_order: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    cer_normalization: str = "comparison_arabic"
    wer_normalization: str = "comparison_arabic"
    ignore_empty_reference: bool = False


@dataclass(frozen=True)
class AwsConfig:
    enabled: bool = False
    region: str = "us-east-2"
    spot_family: str = "P"
    expected_test_instance: str = "p3.2xlarge"
    s3_bucket: str = ""


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    aws: AwsConfig = field(default_factory=AwsConfig)


def _path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def config_from_mapping(
    data: dict[str, Any], project_root: Path | None = None
) -> ProjectConfig:
    root = Path(project_root or data.get("project_root") or ".").resolve()
    paths_data = data.get("paths", {})
    paths = PathConfig(
        project_root=root,
        raw_documents=_path(
            root, paths_data.get("raw_documents", "data/raw/documents")
        ),
        raw_pages=_path(root, paths_data.get("raw_pages", "data/raw/pages")),
        ground_truth=_path(
            root, paths_data.get("ground_truth", "data/raw/ground_truth")
        ),
        layout_annotations=_path(
            root, paths_data.get("layout_annotations", "data/raw/layout_annotations")
        ),
        outputs=_path(root, paths_data.get("outputs", "outputs")),
        previews=_path(root, paths_data.get("previews", "outputs/previews")),
        manifests=_path(root, paths_data.get("manifests", "data/manifests")),
        cache=_path(root, paths_data.get("cache", "cache")),
        temp=_path(root, paths_data.get("temp", "temp")),
        logs=_path(root, paths_data.get("logs", "logs")),
    )
    runtime = RuntimeConfig(**{**RuntimeConfig().__dict__, **data.get("runtime", {})})
    gpu = GpuConfig(**{**GpuConfig().__dict__, **data.get("gpu", {})})
    ground_truth = GroundTruthConfig(
        **{**GroundTruthConfig().__dict__, **data.get("ground_truth", {})}
    )
    evaluation = EvaluationConfig(
        **{**EvaluationConfig().__dict__, **data.get("evaluation", {})}
    )
    aws = AwsConfig(**{**AwsConfig().__dict__, **data.get("aws", {})})
    cfg = ProjectConfig(
        paths=paths,
        runtime=runtime,
        gpu=gpu,
        ground_truth=ground_truth,
        evaluation=evaluation,
        aws=aws,
    )
    validate_config(cfg)
    return cfg


def validate_config(config: ProjectConfig) -> None:
    errors: list[str] = []
    if config.runtime.render_dpi < 72 or config.runtime.render_dpi > 600:
        errors.append("runtime.render_dpi must be between 72 and 600.")
    if config.runtime.workers < 1:
        errors.append("runtime.workers must be at least 1.")
    if not 1 <= config.runtime.cpu_limit_percent <= 100:
        errors.append("runtime.cpu_limit_percent must be between 1 and 100.")
    if config.runtime.batch_size < 1:
        errors.append("runtime.batch_size must be at least 1.")
    if config.runtime.retry_limit < 0:
        errors.append("runtime.retry_limit cannot be negative.")
    if config.runtime.disk_free_min_gb < 0:
        errors.append("runtime.disk_free_min_gb cannot be negative.")
    if config.ground_truth.unicode_normalization not in {"NFC", "NFKC", "NFD", "NFKD"}:
        errors.append(
            "ground_truth.unicode_normalization must be NFC, NFKC, NFD, or NFKD."
        )
    if config.aws.enabled and not config.aws.region:
        errors.append("aws.region is required when aws.enabled is true.")
    root = config.paths.project_root.resolve()
    for field_name, value in config.paths.__dict__.items():
        if field_name == "project_root":
            continue
        try:
            value.resolve().relative_to(root)
        except ValueError:
            errors.append(f"paths.{field_name} must stay inside project_root.")
    if errors:
        raise ConfigError("\n".join(errors))
