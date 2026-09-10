"""Tests: Distortion Experiments + Sensitivity (Phases 11-12).

Verifies the lab calls the canonical Data Factory engine (no duplication)
and that plans/seeds are deterministic with full provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from clouda_data.factory.distort.atomic import apply_distortion, available
from clouda_data.factory.profiles import load_profile_book
from clouda_data.factory.seed.derive import derive_seed
from clouda_lab.distortion_experiments import (
    plan_distortion_experiment,
    run_distortion_experiment,
    sensitivity_by_distortion,
    model_resilience_comparison,
)


def _write_manifest(path: Path, sample_ids: list[str]) -> Path:
    rows = [
        {
            "_schema_version": "clouda.pretraining.manifest.v1",
            "_row_count": len(sample_ids),
        }
    ]
    rows.extend(
        {
            "sample_id": sid,
            "target_split": "train",
            "source_id": "synthetic",
            "source_license": "Apache-2.0",
        }
        for sid in sample_ids
    )
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )
    return path


class TestPlanning:
    def test_plan_by_profile(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1", "s-2"])
        plan = plan_distortion_experiment(
            source_manifest=str(manifest),
            source_sample_ids=["s-1", "s-2"],
            profile="bad_scan_medium",
            seed=100,
        )
        assert plan["experiment_id"].startswith("dex_")
        assert len(plan["variants"]) == 2
        steps = plan["variants"][0]["steps"]
        book = load_profile_book(include_scan_factory=False)
        expected = [
            (s.distortion, s.severity) for s in book.profile("bad_scan_medium").steps
        ]
        assert [(st["distortion"], st["severity"]) for st in steps] == expected
        # provenance fields present
        assert plan["variants"][0]["seed"] > 0
        assert plan["source_manifest"].endswith("m.jsonl")

    def test_plan_by_explicit_distortions(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        plan = plan_distortion_experiment(
            source_manifest=str(manifest),
            source_sample_ids=["s-1"],
            distortions=["gaussian_blur", "skew"],
            severities=["light", "heavy"],
        )
        steps = plan["variants"][0]["steps"]
        assert [st["distortion"] for st in steps] == ["gaussian_blur", "skew"]
        assert [st["severity"] for st in steps] == ["light", "heavy"]

    def test_plan_requires_profile_or_distortions(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        with pytest.raises(ValueError, match="profile or an explicit distortion"):
            plan_distortion_experiment(
                source_manifest=str(manifest), source_sample_ids=["s-1"]
            )

    def test_plan_rejects_both(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        with pytest.raises(ValueError, match="either profile or distortions"):
            plan_distortion_experiment(
                source_manifest=str(manifest),
                source_sample_ids=["s-1"],
                profile="bad_scan_medium",
                distortions=["skew"],
            )

    def test_plan_unknown_profile_or_distortion(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        with pytest.raises(ValueError, match="Unknown profile"):
            plan_distortion_experiment(
                source_manifest=str(manifest), source_sample_ids=["s-1"], profile="nope"
            )
        with pytest.raises(ValueError, match="Unknown distortions"):
            plan_distortion_experiment(
                source_manifest=str(manifest),
                source_sample_ids=["s-1"],
                distortions=["nope"],
            )

    def test_plan_deterministic(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1", "s-2"])
        kwargs: dict[str, Any] = dict(
            source_manifest=str(manifest),
            source_sample_ids=["s-1", "s-2"],
            profile="bad_scan_medium",
            seed=55,
        )
        first = plan_distortion_experiment(**kwargs)
        second = plan_distortion_experiment(**kwargs)
        # Everything except the wall-clock stamp is identical.
        first.pop("created_utc"), second.pop("created_utc")
        assert first == second

    def test_seed_uses_canonical_derive(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        plan = plan_distortion_experiment(
            source_manifest=str(manifest),
            source_sample_ids=["s-1"],
            seed=7,
            distortions=["skew"],
        )
        expected = derive_seed(
            7,
            "s-1",
            document_id="s-1",
            variant_index=0,
            profile="explicit:skew",
        )
        assert plan["variants"][0]["seed"] == expected


class TestGeneration:
    def test_runs_through_canonical_engine(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        # Textured source so the blur actually changes pixels.
        image_path = tmp_path / "src.png"
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
            image_path
        )
        plan = plan_distortion_experiment(
            source_manifest=str(manifest),
            source_sample_ids=["s-1"],
            distortions=["gaussian_blur"],
            severities=["heavy"],
            seed=3,
        )
        result = run_distortion_experiment(
            plan, source_images={"s-1": str(image_path)}, output_dir=tmp_path / "out"
        )
        rows = result["rows"]
        assert len(rows) == 1 and rows[0]["status"] == "ok"
        output = Path(rows[0]["output_path"])
        assert output.is_file()
        # output differs from source and is a valid image
        generated = np.asarray(Image.open(output))
        source = np.asarray(Image.open(image_path))
        assert not np.array_equal(generated, source)
        # provenance recorded
        assert rows[0]["applied"][0]["distortion"] == "gaussian_blur"
        assert rows[0]["seed"] == plan["variants"][0]["seed"]
        assert (tmp_path / "out" / "experiment.json").is_file()
        assert (tmp_path / "out" / "variants.jsonl").is_file()

    def test_deterministic_seed_same_output(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        image_path = tmp_path / "src.png"
        Image.new("RGB", (80, 60), (200, 200, 210)).save(image_path)
        plan = plan_distortion_experiment(
            source_manifest=str(manifest),
            source_sample_ids=["s-1"],
            distortions=["gaussian_noise"],
            severities=["medium"],
            seed=11,
        )
        out_a = run_distortion_experiment(
            plan, source_images={"s-1": str(image_path)}, output_dir=tmp_path / "out_a"
        )
        out_b = run_distortion_experiment(
            plan, source_images={"s-1": str(image_path)}, output_dir=tmp_path / "out_b"
        )
        a = np.asarray(Image.open(out_a["rows"][0]["output_path"]))
        b = np.asarray(Image.open(out_b["rows"][0]["output_path"]))
        assert np.array_equal(a, b)

    def test_missing_source_image_row_errors(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path / "m.jsonl", ["s-1"])
        plan = plan_distortion_experiment(
            source_manifest=str(manifest),
            source_sample_ids=["s-1"],
            distortions=["skew"],
        )
        result = run_distortion_experiment(
            plan, source_images={}, output_dir=tmp_path / "out"
        )
        assert result["rows"][0]["status"] == "error"

    def test_uses_factory_apply_distortion(self):
        # The lab module must import the canonical engine (no parallel impl).
        import clouda_lab.distortion_experiments as de

        assert de.apply_distortion is apply_distortion
        assert de.available is available


class TestSensitivity:
    def test_cer_by_distortion(self):
        rows = [
            {"sample_id": "1", "cer": 0.10, "wer": 0.20, "distortion": "clean"},
            {"sample_id": "2", "cer": 0.30, "wer": 0.40, "distortion": "gaussian_blur"},
            {"sample_id": "3", "cer": 0.50, "wer": 0.60, "distortion": "gaussian_blur"},
            {
                "sample_id": "4",
                "cer": 0.25,
                "wer": 0.35,
                "applied": [{"distortion": "skew", "severity": "heavy"}],
            },
        ]
        summary = sensitivity_by_distortion(rows)
        assert summary["by_distortion"]["clean"]["count"] == 1
        assert summary["by_distortion"]["gaussian_blur"]["count"] == 2
        assert summary["by_distortion"]["gaussian_blur"]["cer_mean"] == pytest.approx(
            0.40
        )
        assert summary["by_distortion"]["skew"]["count"] == 1
        assert summary["by_profile"]["unspecified"]["count"] == 4
        # hardest distortion first by CER
        by_cer = sorted(
            summary["by_distortion"].items(), key=lambda kv: -kv[1]["cer_mean"]
        )
        assert by_cer[0][0] == "gaussian_blur"

    def test_only_available_results_used(self):
        assert sensitivity_by_distortion([]) == {"by_distortion": {}, "by_profile": {}}
        no_metrics = sensitivity_by_distortion([{"sample_id": "x", "profile": "p"}])
        assert no_metrics["by_profile"] == {}

    def test_model_resilience(self):
        rows = [
            {"sample_id": "1", "model_id": "A", "cer": 0.5, "distortion": "skew"},
            {"sample_id": "2", "model_id": "B", "cer": 0.2, "distortion": "skew"},
        ]
        result = model_resilience_comparison(rows)
        assert result["A"]["skew"]["cer_mean"] == 0.5
        assert result["B"]["skew"]["cer_mean"] == 0.2
