from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clouda_data.distortion.base import DistortionSpec
from clouda_data.distortion.registry import default_registry
from clouda_data.locations import default_profile_dir
from jsonschema import Draft202012Validator
import yaml

REQUIRED_PROFILE_FIELDS = {
    "name",
    "recommended_use",
    "distortions",
    "ordering_constraints",
    "mutually_exclusive",
    "maximum_allowed_crop",
    "minimum_readable_text_threshold",
    "metadata_fields",
}
_SEVERITY_COST = {
    "none": 0.0,
    "minimal": 0.25,
    "light": 0.5,
    "medium": 1.0,
    "heavy": 2.0,
    "extreme": 4.0,
}


def load_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    text = profile_path.read_text(encoding="utf-8")
    profile = (
        yaml.safe_load(text)
        if profile_path.suffix.lower() in {".yaml", ".yml"}
        else json.loads(text)
    )
    validate_profile(profile)
    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    if "profile_id" in profile:
        repository_schema = (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "distortion-profile-v1.schema.json"
        )
        packaged_schema = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "schemas"
            / "distortion-profile-v1.schema.json"
        )
        schema_path = (
            repository_schema if repository_schema.is_file() else packaged_schema
        )
        if not schema_path.is_file():
            raise FileNotFoundError("Distortion profile schema is not installed")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = sorted(
            Draft202012Validator(schema).iter_errors(profile),
            key=lambda item: list(item.path),
        )
        if errors:
            raise ValueError("; ".join(error.message for error in errors))
        profile.setdefault("name", profile["profile_id"])
        profile.setdefault("recommended_use", profile["intended_use"])
        profile.setdefault("distortions", profile["transformations"])
        profile.setdefault("ordering_constraints", [])
        profile.setdefault("mutually_exclusive", profile.get("exclusions", []))
        profile.setdefault("maximum_allowed_crop", 0.05)
        profile.setdefault("minimum_readable_text_threshold", 0.7)
        profile.setdefault(
            "metadata_fields", ["operation_order", "parameters", "random_seed"]
        )
    missing = sorted(REQUIRED_PROFILE_FIELDS - set(profile))
    if missing:
        raise ValueError(f"Profile is missing fields: {', '.join(missing)}")
    if not 0 <= profile["maximum_allowed_crop"] <= 0.2:
        raise ValueError("maximum_allowed_crop must be between 0 and 0.2.")
    if not 0 <= profile["minimum_readable_text_threshold"] <= 1:
        raise ValueError("minimum_readable_text_threshold must be between 0 and 1.")
    if not isinstance(profile["distortions"], list):
        raise ValueError("distortions must be a list.")
    maximum_chain_length = int(profile.get("maximum_chain_length", 32))
    if len(profile["distortions"]) > maximum_chain_length:
        raise ValueError("Profile exceeds maximum_chain_length")
    known_operators = set(default_registry().names())
    severity_budget = 0.0
    for item in profile["distortions"]:
        if item["name"] not in known_operators:
            raise ValueError(f"Unknown distortion operator: {item['name']}")
        spec = DistortionSpec(
            name=item["name"],
            probability=float(item.get("probability", 1.0)),
            severity=item.get("severity", "medium"),
            parameters=item.get("parameters", {}),
        )
        spec.validate()
        severity_budget += _SEVERITY_COST[spec.severity] * spec.probability
    maximum_severity_budget = float(profile.get("maximum_severity_budget", 32))
    if severity_budget > maximum_severity_budget:
        raise ValueError(
            f"Profile severity budget {severity_budget:.2f} exceeds "
            f"{maximum_severity_budget:.2f}"
        )


def profile_to_specs(profile: dict[str, Any]) -> list[DistortionSpec]:
    return [
        DistortionSpec(
            name=item["name"],
            probability=float(item.get("probability", 1.0)),
            severity=item.get("severity", "medium"),
            parameters=item.get("parameters", {}),
        )
        for item in profile["distortions"]
    ]


def list_profile_paths(config_dir: str | Path | None = None) -> list[Path]:
    root = Path(config_dir) if config_dir is not None else default_profile_dir()
    return sorted([*root.glob("*.json"), *root.glob("*.yaml"), *root.glob("*.yml")])
