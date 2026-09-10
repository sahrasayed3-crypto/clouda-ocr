"""Distortion Experiment Engine.

Backend experiment layer **over** the integrated Data Factory
(``clouda_data.factory``) — it never reimplements distortion operations.

What it does:
- plans tiny, deterministic distortion experiments (source samples × profile
  or explicit distortion list × seed);
- derives per-variant seeds with the canonical
  ``clouda_data.factory.seed.derive.derive_seed`` scheme;
- executes through the canonical profile book + atomic distortion engine
  (the same code path ``factory._distort_page`` uses);
- attaches full provenance (source sample, seed, profile, distortions,
  factory version) to every generated variant;
- writes a machine-readable experiment manifest (JSON + JSONL rows).

Sensitivity analysis (Phase 12) combines OCR evaluation results with the
distortion metadata recorded here — only from actual available results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from clouda_data.factory.distort.atomic import apply_distortion, available
from clouda_data.factory.profiles import load_profile_book
from clouda_data.factory.seed.derive import derive_seed

from .io import export_json, export_jsonl

EXPERIMENT_SCHEMA_VERSION = "clouda.lab.distortion_experiment.v1"

# Factory severity vocabulary: light/medium/heavy (not low/high).
DEFAULT_SEVERITIES: tuple[str, ...] = ("light", "medium", "heavy")

_DEFAULT_BASE_SEED = 20260831  # factory DEFAULT_BASE_SEED


@dataclass(frozen=True)
class VariantResult:
    """One generated variant with full provenance."""

    variant_id: str
    experiment_id: str
    source_sample_id: str
    profile: str
    distortions: tuple[str, ...]
    seed: int
    output_path: str | None
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "experiment_id": self.experiment_id,
            "source_sample_id": self.source_sample_id,
            "profile": self.profile,
            "distortions": list(self.distortions),
            "seed": self.seed,
            "output_path": self.output_path,
            "provenance": dict(self.provenance),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _experiment_id(
    source_manifest: str, source_sample_ids: Sequence[str], profile: str, seed: int
) -> str:
    material = "\x1f".join(
        [source_manifest, ",".join(sorted(source_sample_ids)), profile, str(seed)]
    )
    return "dex_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def plan_distortion_experiment(
    *,
    source_manifest: str,
    source_sample_ids: Sequence[str],
    profile: str | None = None,
    distortions: Sequence[str] | None = None,
    severities: Sequence[str] = DEFAULT_SEVERITIES,
    variants_per_sample: int = 1,
    seed: int = _DEFAULT_BASE_SEED,
    notes: str = "",
) -> dict[str, Any]:
    """Plan an experiment deterministically; no images are generated here.

    Either ``profile`` (use a unified profile's step list) or ``distortions``
    (explicit atomic distortion names) must be given. Returns a plan dict;
    generation happens in :func:`run_distortion_experiment`.
    """
    if not profile and not distortions:
        raise ValueError("Provide a profile or an explicit distortion list")
    if profile and distortions:
        raise ValueError("Provide either profile or distortions, not both")
    book = load_profile_book(include_scan_factory=False)
    if profile and profile not in book.profiles:
        raise ValueError(f"Unknown profile: {profile} (have: {sorted(book.profiles)})")
    if distortions:
        unknown = [d for d in distortions if d not in available()]
        if unknown:
            raise ValueError(f"Unknown distortions: {unknown} (have: {available()})")

    experiment_id = _experiment_id(
        source_manifest, source_sample_ids, profile or ",".join(distortions or []), seed
    )
    variants: list[dict[str, Any]] = []
    for sample_id in source_sample_ids:
        for variant_index in range(max(1, variants_per_sample)):
            if profile:
                steps = [
                    (step.distortion, step.severity)
                    for step in book.profile(profile).steps
                ]
                variant_seed = derive_seed(
                    seed,
                    sample_id,
                    document_id=sample_id,
                    variant_index=variant_index,
                    profile=profile,
                )
                distortion_names = tuple(d for d, _s in steps)
            else:
                names = list(distortions or [])
                steps = [
                    (name, severities[index % len(severities)])
                    for index, name in enumerate(names)
                ]
                variant_seed = derive_seed(
                    seed,
                    sample_id,
                    document_id=sample_id,
                    variant_index=variant_index,
                    profile="explicit:" + ",".join(names),
                )
                distortion_names = tuple(names)
            variants.append(
                {
                    "variant_id": f"{experiment_id}:{sample_id}:v{variant_index}",
                    "source_sample_id": sample_id,
                    "profile": profile or "",
                    "steps": [{"distortion": d, "severity": s} for d, s in steps],
                    "distortions": list(distortion_names),
                    "seed": variant_seed,
                }
            )
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "source_manifest": source_manifest,
        "source_sample_ids": list(source_sample_ids),
        "base_seed": seed,
        "notes": notes,
        "created_utc": _utc_now(),
        "variants": variants,
    }


def run_distortion_experiment(
    plan: Mapping[str, Any],
    *,
    source_images: Mapping[str, str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Execute a plan through the canonical atomic distortion engine.

    ``source_images`` maps source_sample_id -> image path (PNG/JPEG). Each
    variant image is written under ``output_dir``; rows carry provenance.
    """
    from PIL import Image

    book = load_profile_book(include_scan_factory=False)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in plan.get("variants", []):
        sample_id = str(variant["source_sample_id"])
        image_path = source_images.get(sample_id)
        if not image_path:
            rows.append(
                {
                    "variant_id": variant["variant_id"],
                    "status": "error",
                    "error": f"no source image for {sample_id}",
                }
            )
            continue
        image = Image.open(image_path).convert("RGB")
        array = __import__("numpy").asarray(image)
        applied: list[dict[str, Any]] = []
        seed = int(variant["seed"])
        for index, step in enumerate(variant.get("steps", [])):
            name = str(step["distortion"])
            severity = str(step["severity"])
            spec = book.spec(name)
            stage_seed = derive_seed(
                seed,
                sample_id,
                distortion_stage=name,
                severity=severity,
                variant_index=index,
            )
            array = apply_distortion(
                name,
                array,
                spec.params(severity),
                __import__("numpy").random.default_rng(stage_seed),
            )
            applied.append(
                {"distortion": name, "severity": severity, "seed": stage_seed}
            )
        out_image = Image.fromarray(array)
        output_name = f"{variant['variant_id'].replace(':', '__')}.png"
        output_path = out_root / output_name
        out_image.save(output_path)
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "experiment_id": plan.get("experiment_id", ""),
                "source_sample_id": sample_id,
                "profile": variant.get("profile", ""),
                "distortions": variant.get("distortions", []),
                "seed": seed,
                "applied": applied,
                "output_path": str(output_path),
                "source_image": str(image_path),
                "status": "ok",
                "created_utc": _utc_now(),
            }
        )
    result = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": plan.get("experiment_id", ""),
        "plan": dict(plan),
        "rows": rows,
        "output_dir": str(out_root),
    }
    export_json(result, out_root / "experiment.json")
    export_jsonl(rows, out_root / "variants.jsonl")
    return result


# ---------------------------------------------------------------------------
# Phase 12 — distortion sensitivity analysis
# ---------------------------------------------------------------------------


def sensitivity_by_distortion(
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate actual evaluation results by distortion type / profile.

    ``evaluation_rows`` must carry per-sample ``cer`` (or ``wer``) plus
    distortion metadata (``distortion`` / ``profile`` / ``applied``).
    Only available results are aggregated — nothing is simulated.
    """
    by_distortion: dict[str, list[tuple[float, float]]] = {}
    by_profile: dict[str, list[tuple[float, float]]] = {}
    for row in evaluation_rows:
        cer = row.get("cer")
        wer = row.get("wer")
        if cer is None and wer is None:
            continue
        cer_value = float(cer) if cer is not None else 0.0
        wer_value = float(wer) if wer is not None else 0.0
        profile = str(row.get("profile", "") or "unspecified")
        by_profile.setdefault(profile, []).append((cer_value, wer_value))
        distortion = row.get("distortion")
        applied = row.get("applied")
        names: set[str] = set()
        if isinstance(distortion, str) and distortion:
            names.add(distortion)
        elif isinstance(distortion, (list, tuple)):
            names.update(str(d) for d in distortion if d)
        if isinstance(applied, (list, tuple)):
            for step in applied:
                if isinstance(step, Mapping) and step.get("distortion"):
                    names.add(str(step["distortion"]))
        for name in sorted(names) or ["clean"]:
            by_distortion.setdefault(name, []).append((cer_value, wer_value))

    def _summarize(
        groups: dict[str, list[tuple[float, float]]],
    ) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for key, pairs in sorted(groups.items()):
            cers = [p[0] for p in pairs]
            wers = [p[1] for p in pairs]
            summary[key] = {
                "count": len(pairs),
                "cer_mean": sum(cers) / len(cers) if cers else 0.0,
                "wer_mean": sum(wers) / len(wers) if wers else 0.0,
                "cer_max": max(cers) if cers else 0.0,
            }
        return summary

    return {
        "by_distortion": _summarize(by_distortion),
        "by_profile": _summarize(by_profile),
    }


def model_resilience_comparison(
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    model_field: str = "model_id",
) -> dict[str, dict[str, dict[str, float]]]:
    """CER by distortion per model — which model resists which distortion."""
    per_model: dict[str, list[Mapping[str, Any]]] = {}
    for row in evaluation_rows:
        model = str(row.get(model_field, "unspecified"))
        per_model.setdefault(model, []).append(row)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for model, rows in sorted(per_model.items()):
        result[model] = sensitivity_by_distortion(rows)["by_distortion"]
    return result


__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "VariantResult",
    "model_resilience_comparison",
    "plan_distortion_experiment",
    "run_distortion_experiment",
    "sensitivity_by_distortion",
]
