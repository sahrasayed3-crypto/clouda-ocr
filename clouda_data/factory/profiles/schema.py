"""Unified profile schema and loaders (MERGE_PLAN.md phase 3).

A unified profile has:
  - optional `steps`: ordered atomic distortions with named severities
    (System A semantics, executed by distort.atomic with a per-stage rng);
  - optional `composite`: true to run System B's fixed-order scan simulation
    instead of/in addition to atomic steps;
  - presentation fields from System B: dpi_target, color, jpeg_quality,
    photocopy_generations, paper_border;
  - metadata: family, recoverability, qc thresholds.

Legacy sets load through lossless adapters:
  - load_ocr_benchmark(): System A configs/distortions.yaml verbatim
    (DistortionSpec.params(severity) contracts unchanged).
  - load_scan_factory(): System B's 12 profile dicts verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ProfileError(ValueError):
    """Raised when a profile or distortion config is invalid."""


@dataclass(frozen=True)
class ProfileStep:
    distortion: str
    severity: str


@dataclass(frozen=True)
class DistortionSpec:
    """One atomic distortion and its per-severity parameter sets.

    Interface-compatible with ocrbench.config.DistortionSpec (System A) so
    the vendored engine consumes it unchanged.
    """

    name: str
    family: str
    output_dir: str
    description: str
    severities: Mapping[str, Mapping[str, Any]]

    def params(self, severity: str) -> Mapping[str, Any]:
        if severity not in self.severities:
            raise ProfileError(
                f"distortion '{self.name}' has no severity '{severity}' "
                f"(available: {sorted(self.severities)})"
            )
        return self.severities[severity]


@dataclass
class UnifiedProfile:
    name: str
    schema: str  # benchmark_legacy | scan_family_v1 | unified_v1
    steps: tuple[ProfileStep, ...] = ()
    composite: bool = False
    dpi_target: int = 150
    color: str = "grayscale"
    jpeg_quality: int = 90
    photocopy_generations: int = 0
    paper_border: tuple[int, int, int] = (245, 241, 230)
    damage: float = 0.0  # composite backend scalar (System B)
    paper: str = "white"  # composite backend paper mode
    family: str = ""
    recoverability: str = "RECOVERABLE"
    qc: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "steps": [
                {"distortion": s.distortion, "severity": s.severity} for s in self.steps
            ],
            "composite": self.composite,
            "dpi_target": self.dpi_target,
            "color": self.color,
            "jpeg_quality": self.jpeg_quality,
            "photocopy_generations": self.photocopy_generations,
            "paper_border": list(self.paper_border),
            "damage": self.damage,
            "paper": self.paper,
            "family": self.family,
            "recoverability": self.recoverability,
            "qc": dict(self.qc),
            "description": self.description,
        }


def load_ocr_benchmark(
    yaml_path: Path,
) -> tuple[dict[str, DistortionSpec], dict[str, UnifiedProfile], dict]:
    """Load System A's distortions.yaml losslessly (distortion specs +
    combined profiles mapped into UnifiedProfile with steps; raw dict kept)."""
    yaml_path = Path(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "distortions" not in raw:
        raise ProfileError(f"{yaml_path}: expected a 'distortions' mapping")

    specs: dict[str, DistortionSpec] = {}
    for name, body in raw["distortions"].items():
        severities = body["severities"]
        specs[name] = DistortionSpec(
            name=name,
            family=str(body.get("family", name)),
            output_dir=str(body.get("output_dir", body.get("family", name))),
            description=str(body.get("description", "")),
            severities=severities,
        )

    aliases = raw.get("profile_aliases", {}) or {}
    order = raw.get("mapping_expansion_order", []) or []
    profiles: dict[str, UnifiedProfile] = {}
    for name, body in (raw.get("profiles") or {}).items():
        if "steps" in body:
            steps = tuple(
                ProfileStep(
                    distortion=str(s["distortion"]), severity=str(s["severity"])
                )
                for s in body["steps"]
            )
        elif "mapping" in body:
            resolved: dict[str, str] = {}
            for alias, severity in body["mapping"].items():
                target = aliases.get(alias, alias)
                if target not in specs:
                    raise ProfileError(
                        f"profile '{name}': unknown distortion '{alias}'"
                    )
                resolved[target] = str(severity)
            ordered = [d for d in order if d in resolved]
            ordered += sorted(d for d in resolved if d not in ordered)
            steps = tuple(
                ProfileStep(distortion=d, severity=resolved[d]) for d in ordered
            )
        else:
            raise ProfileError(f"profile '{name}': needs 'steps' or 'mapping'")
        for step in steps:
            specs[step.distortion].params(step.severity)
        profiles[name] = UnifiedProfile(
            name=name,
            schema="benchmark_legacy",
            steps=steps,
            dpi_target=150,
            color="grayscale",
            family=str(body.get("output_dir", "combined")),
            qc=dict(raw.get("readability", {}) or {}),
            description=str(body.get("description", "")),
        )
    return specs, profiles, raw


def load_scan_factory(module: Any) -> dict[str, UnifiedProfile]:
    """Convert the vendored System B PROFILES dict into UnifiedProfile entries
    (all fields preserved verbatim)."""
    profiles: dict[str, UnifiedProfile] = {}
    for name, params in module.PROFILES.items():
        profiles[name] = UnifiedProfile(
            name=name,
            schema="scan_family_v1",
            steps=(),
            composite=True,
            dpi_target=int(params["dpi"]),
            color=str(params["color"]),
            jpeg_quality=int(params["jpeg_quality"]),
            photocopy_generations=int(params["photocopy_generation"]),
            paper_border=(245, 241, 230),
            damage=float(params["damage"]),
            paper=str(params["paper"]),
            family=str(params["family"]),
            recoverability=str(params["recoverability"]),
            description="System B scan-family simulation (composite backend)",
        )
    return profiles
