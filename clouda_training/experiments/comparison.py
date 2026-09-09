from __future__ import annotations

from typing import Any

import yaml

from .runs import RunHandle


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, dotted))
        else:
            result[dotted] = item
    return result


def compare_runs(left: RunHandle, right: RunHandle) -> dict[str, Any]:
    configs = [
        yaml.safe_load((run.path / "resolved_config.yaml").read_text(encoding="utf-8"))
        for run in (left, right)
    ]
    flat = [_flatten(config) for config in configs]
    config_differences = {
        key: [flat[0].get(key), flat[1].get(key)]
        for key in sorted(set(flat[0]) | set(flat[1]))
        if flat[0].get(key) != flat[1].get(key)
    }
    summaries = [run.summary() for run in (left, right)]
    metric_names = set(summaries[0].get("final_metrics", {})) | set(
        summaries[1].get("final_metrics", {})
    )
    metric_differences = {
        name: [
            summaries[0].get("final_metrics", {}).get(name),
            summaries[1].get("final_metrics", {}).get(name),
        ]
        for name in sorted(metric_names)
    }
    return {
        "schema_version": 1,
        "runs": [run.to_dict() for run in (left, right)],
        "config_differences": config_differences,
        "metric_differences": metric_differences,
    }
