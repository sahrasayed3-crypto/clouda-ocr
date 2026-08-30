from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ProjectConfig, config_from_mapping


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML configuration files."
        ) from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the top level.")
    return data


def load_config(
    path: str | Path = "config.example.yaml", project_root: str | Path | None = None
) -> ProjectConfig:
    config_path = Path(path)
    data = _load_yaml_or_json(config_path)
    root = (
        Path(project_root).resolve() if project_root else config_path.parent.resolve()
    )
    return config_from_mapping(data, root)
