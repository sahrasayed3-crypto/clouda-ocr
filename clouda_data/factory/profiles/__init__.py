"""Profile loading: unified YAML + lossless legacy adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    DistortionSpec,
    ProfileError,
    ProfileStep,
    UnifiedProfile,
    load_ocr_benchmark,
    load_scan_factory,
)
from . import scan_families

_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "data_factory"


@dataclass
class ProfileBook:
    """All distortion specs and profiles visible to a run."""

    distortions: dict[str, DistortionSpec] = field(default_factory=dict)
    profiles: dict[str, UnifiedProfile] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def profile(self, name: str) -> UnifiedProfile:
        if name not in self.profiles:
            raise ProfileError(
                f"unknown profile '{name}' (have: {sorted(self.profiles)})"
            )
        return self.profiles[name]

    def spec(self, name: str) -> DistortionSpec:
        if name not in self.distortions:
            raise ProfileError(
                f"unknown distortion '{name}' (have: {sorted(self.distortions)})"
            )
        return self.distortions[name]


def load_profile_book(
    ocr_benchmark_yaml: Path | None = None,
    include_scan_factory: bool = True,
) -> ProfileBook:
    """Load System A's distortion catalogue (verbatim) + System B's 12 scan
    families (first-class scan_families module). Names from both systems are
    preserved."""
    book = ProfileBook()
    a_yaml = (
        Path(ocr_benchmark_yaml)
        if ocr_benchmark_yaml
        else _CONFIGS / "ocr_benchmark.yaml"
    )
    specs, profiles, raw = load_ocr_benchmark(a_yaml)
    book.distortions.update(specs)
    book.profiles.update(profiles)
    book.raw = raw

    if include_scan_factory:
        book.profiles.update(load_scan_factory(scan_families))
    return book


def load_unified_yaml(path: Path) -> dict[str, UnifiedProfile]:
    """Load a unified-format profile YAML (configs/distortion_profiles/*.yaml
    with schema: unified_v1 entries)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    profiles: dict[str, UnifiedProfile] = {}
    for name, body in (raw.get("profiles") or {}).items():
        steps = tuple(
            ProfileStep(distortion=str(s["distortion"]), severity=str(s["severity"]))
            for s in body.get("steps", [])
        )
        profiles[name] = UnifiedProfile(
            name=name,
            schema=str(body.get("schema", "unified_v1")),
            steps=steps,
            composite=bool(body.get("composite", False)),
            dpi_target=int(body.get("dpi_target", 150)),
            color=str(body.get("color", "grayscale")),
            jpeg_quality=int(body.get("jpeg_quality", 90)),
            photocopy_generations=int(body.get("photocopy_generations", 0)),
            paper_border=tuple(body.get("paper_border", (245, 241, 230))),
            damage=float(body.get("damage", 0.0)),
            paper=str(body.get("paper", "white")),
            family=str(body.get("family", "")),
            recoverability=str(body.get("recoverability", "RECOVERABLE")),
            qc=dict(body.get("qc", {}) or {}),
            description=str(body.get("description", "")),
        )
    return profiles
