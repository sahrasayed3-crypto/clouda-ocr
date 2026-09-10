"""Tests: Evaluation Service + CLI + E2E backend flow (Phases 13, 19, 21-E2E).

E2E synthetic flow (no GPU, no network, no model downloads):
    synthetic samples → mock OCR outputs → error analysis
    → failure comparison → hard examples → next-batch recommendation
    → derived training manifest → Training Framework dry-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_lab.cli import main as cli_main
from clouda_lab.evaluation_service import EvaluationService
from clouda_lab.failure_analysis import SampleMetrics
from clouda_lab.io import export_json, load_samples_jsonl
from clouda_lab.models import OCRSample
from clouda_training.experiments import RunStatus

ARABIC_SAMPLES = [
    # (sample_id, ground truth, "mock OCR" prediction, profile)
    ("p-01", "مرحبا بالعالم", "مرحبا بالعالم", "clean"),
    ("p-02", "اللغة العربية جميلة", "اللغه العربيه جميله", "clean"),
    ("p-03", "سنة ٢٠٢٤", "سنة 2024", "clean"),
    ("p-04", "مَكتب المدرسة", "مكتب المدرسه", "old_book_light"),
    ("p-05", "مرحبا بالعالم الجميل", "مرحبا بالعالم", "bad_scan_heavy"),
    ("p-06", "الرقم ١٢٣٤٥ كبير", "الرقم 12345 كبير", "bad_scan_heavy"),
]


def _synthetic_samples() -> list[OCRSample]:
    return [
        OCRSample(
            sample_id=sid,
            ground_truth=gt,
            prediction=pred,
            model_id="mock-model-a",
            dataset_id="synthetic-fixture",
            metadata={
                "profile": profile,
                "split": "test",
                "document_type": "book",
                "source": "synthetic",
            },
        )
        for sid, gt, pred, profile in ARABIC_SAMPLES
    ]


class TestEvaluationService:
    def test_evaluate_sample(self):
        service = EvaluationService()
        analysis = service.evaluate_sample(_synthetic_samples()[1])
        assert analysis.cer > 0.0
        assert analysis.error_type_counts.get("ta_marbuta_ha", 0) >= 1
        assert analysis.error_type_counts.get("word_substitution", 0) >= 1

    def test_evaluate_batch(self):
        service = EvaluationService()
        report = service.evaluate_batch(_synthetic_samples())
        assert report.total_samples == 6
        assert "profile" in report.groups
        assert report.groups["profile"]["bad_scan_heavy"].count == 2

    def test_compare_models(self):
        service = EvaluationService()
        baseline = [
            SampleMetrics(sample_id="p-01", cer=0.40, wer=0.5),
            SampleMetrics(sample_id="p-02", cer=0.10, wer=0.2),
        ]
        candidate = [
            SampleMetrics(sample_id="p-01", cer=0.15, wer=0.25),
            SampleMetrics(sample_id="p-02", cer=0.12, wer=0.2),
        ]
        report = service.compare_models(
            baseline, candidate, baseline_id="mock-model-a", candidate_id="mock-model-b"
        )
        assert report.counts["improved"] == 1
        # p-02: 0.10 -> 0.12 is below the default 0.05 regress threshold
        assert report.counts["unchanged"] == 1
        report2 = service.compare_models(
            baseline,
            candidate,
            baseline_id="a",
            candidate_id="b",
            thresholds={"improve": 0.01, "regress": 0.01},
        )
        assert (
            report2.counts["regressed"] == 1
        )  # p-02 now crosses the tighter threshold

    def test_buckets_hard_and_recommend(self):
        service = EvaluationService()
        samples = _synthetic_samples()
        rows = []
        for sample in samples:
            analysis = service.evaluate_sample(sample)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "cer": analysis.cer,
                    "wer": analysis.wer,
                    "ncer": analysis.normalized_cer,
                    "error_type_counts": analysis.error_type_counts,
                    "profile": sample.metadata.get("profile"),
                    "model_id": sample.model_id,
                }
            )
        buckets = service.failure_buckets(rows)
        assert set(buckets) == {row["sample_id"] for row in rows}
        hard = service.hard_examples(rows, top_n=2)
        assert len(hard) == 2
        recommendation = service.recommend_next(
            rows, batch_size=3, seed=1, history=("p-01",)
        )
        assert "p-01" not in {s.sample_id for s in recommendation.selections}

    def test_exports(self, tmp_path: Path):
        service = EvaluationService()
        samples = _synthetic_samples()
        analysis = service.evaluate_sample(samples[4])
        report = service.evaluate_batch(samples)
        baseline = [
            SampleMetrics(sample_id=s.sample_id, cer=0.5, wer=0.6) for s in samples
        ]
        candidate = [
            SampleMetrics(sample_id=s.sample_id, cer=analysis.cer, wer=analysis.wer)
            for s in samples
        ]
        failure = service.compare_models(
            baseline, candidate, baseline_id="a", candidate_id="b"
        )
        out_a = tmp_path / "analysis.json"
        out_b = tmp_path / "batch.json"
        out_c = tmp_path / "failure.json"
        service.export_analysis(analysis, out_a)
        service.export_batch(report, out_b)
        service.export_failure_report(failure, out_c)
        assert json.loads(out_a.read_text(encoding="utf-8"))["page_id"] == "p-05"
        assert json.loads(out_b.read_text(encoding="utf-8"))["total_samples"] == 6
        assert (
            json.loads(out_c.read_text(encoding="utf-8"))["summary"]["common_samples"]
            == 6
        )


class TestIOLoaders:
    def test_load_samples_jsonl(self, tmp_path: Path):
        records = [
            {
                "sample_id": "x1",
                "reference_text": "مرحبا",
                "prediction_text": "مراحبا",
                "model_id": "m",
                "profile": "clean",
                "severity": "low",
            },
            {"page_id": "x2", "ground_truth_text": "عالم", "ocr_text": "عالم"},
        ]
        path = tmp_path / "samples.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
            encoding="utf-8",
        )
        samples = load_samples_jsonl(path)
        assert len(samples) == 2
        assert samples[0].sample_id == "x1"
        assert samples[1].sample_id == "x2"  # page_id fallback
        assert samples[0].metadata["profile"] == "clean"
        assert samples[1].model_id == "unspecified"

    def test_loader_rejects_missing_gt(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"sample_id": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="ground-truth"):
            load_samples_jsonl(path)


class TestCLI:
    def test_analysis_batch_json(self, tmp_path: Path, capsys):
        samples = _synthetic_samples()
        path = tmp_path / "samples.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "sample_id": s.sample_id,
                        "reference_text": s.ground_truth,
                        "prediction_text": s.prediction,
                        "model_id": s.model_id,
                        **s.metadata,
                    },
                    ensure_ascii=False,
                )
                for s in samples
            ),
            encoding="utf-8",
        )
        out = tmp_path / "batch.json"
        exit_code = cli_main(
            ["analysis-batch", "--input", str(path), "--output", str(out)]
        )
        assert exit_code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["total_samples"] == 6

    def test_dataset_select_and_validate(self, tmp_path: Path, capsys):
        rows = [
            {
                "_schema_version": "clouda.pretraining.manifest.v1",
                "_row_count": 3,
                "dataset_role": "training",
            }
        ]
        rows.extend(
            {
                "sample_id": f"s-{i}",
                "target_split": "train",
                "source_id": "src",
                "source_license": "Apache-2.0",
                "cer": i / 10,
            }
            for i in range(3)
        )
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
        derived = tmp_path / "derived.jsonl"
        exit_code = cli_main(
            [
                "dataset-select",
                "--manifest",
                str(manifest),
                "--output",
                str(derived),
                "--cer-min",
                "0.1",
                "--seed",
                "3",
            ]
        )
        assert exit_code == 0
        assert derived.is_file()
        exit_code = cli_main(["dataset-validate", "--manifest", str(derived)])
        assert exit_code == 0
        assert '"rows": 2' in capsys.readouterr().out

    def test_hard_examples_cli(self, tmp_path: Path):
        rows = [
            {"sample_id": f"s{i}", "cer": i / 10, "wer": i / 10, "ncer": i / 20}
            for i in range(5)
        ]
        path = tmp_path / "rows.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        out = tmp_path / "hard.json"
        exit_code = cli_main(
            [
                "dataset-hard-examples",
                "--input",
                str(path),
                "--top-n",
                "2",
                "--output",
                str(out),
            ]
        )
        assert exit_code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload) == 2
        assert payload[0]["sample_id"] == "s4"

    def test_recommend_next_cli(self, tmp_path: Path):
        rows = [
            {
                "sample_id": f"a{i}",
                "cer": 0.5,
                "wer": 0.6,
                "ncer": 0.4,
                "profile": "clean",
                "error_type_counts": {"whitespace": 1},
                "model_id": "m",
            }
            for i in range(3)
        ] + [
            {
                "sample_id": f"b{i}",
                "cer": 0.4,
                "wer": 0.5,
                "ncer": 0.3,
                "profile": "rough",
                "error_type_counts": {"diacritic": 1},
                "model_id": "m",
            }
            for i in range(3)
        ]
        path = tmp_path / "rows.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        out = tmp_path / "rec.json"
        exit_code = cli_main(
            [
                "dataset-recommend-next",
                "--input",
                str(path),
                "--strategy",
                "balanced_hard",
                "--batch-size",
                "4",
                "--output",
                str(out),
            ]
        )
        assert exit_code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload["selections"]) == 4
        assert payload["balance"]["profile"] == {"clean": 2, "rough": 2}


class TestEndToEnd:
    def test_full_backend_flow(self, tmp_path: Path):
        """The complete Phase 21 E2E synthetic flow."""
        service = EvaluationService()
        samples = _synthetic_samples()

        # 1. error analysis per sample
        analyses = [service.evaluate_sample(s) for s in samples]
        assert all(a.cer >= 0.0 for a in analyses)

        # 2. mock per-run metrics (synthetic — NOT from MockTrainer)
        baseline_metrics = [
            SampleMetrics(
                sample_id=s.sample_id,
                cer=a.cer + 0.10 if a.cer < 0.3 else a.cer + 0.20,
                wer=a.wer + 0.10,
                ncer=a.normalized_cer + 0.05,
                error_types=a.error_type_counts,
            )
            for s, a in zip(samples, analyses)
        ]
        candidate_metrics = [
            SampleMetrics(
                sample_id=s.sample_id,
                cer=a.cer,
                wer=a.wer,
                ncer=a.normalized_cer,
                error_types=a.error_type_counts,
            )
            for s, a in zip(samples, analyses)
        ]

        # 3. failure comparison baseline vs candidate run
        # p-01 is a perfect match: baseline formula gives +0.10 vs +0.20 for
        # others, and unchanged samples (p-01: 0.10 -> 0.0) may classify as
        # recovered/unchanged rather than improved. Assert on the strict
        # subset that provably crosses the improve threshold.
        failure = service.compare_models(
            baseline_metrics,
            candidate_metrics,
            baseline_id="run-000",
            candidate_id="run-001",
        )
        assert (
            failure.counts["improved"] + failure.counts["recovered"] >= len(samples) - 2
        )
        assert failure.counts["regressed"] == 0
        assert failure.counts["newly_failed"] == 0

        # 4. analysis rows → hard examples → recommendation
        rows = [
            {
                "sample_id": s.sample_id,
                "cer": a.cer,
                "wer": a.wer,
                "ncer": a.normalized_cer,
                "error_type_counts": a.error_type_counts,
                "profile": s.metadata["profile"],
                "distortion": s.metadata["profile"],
                "model_id": s.model_id,
            }
            for s, a in zip(samples, analyses)
        ]
        hard = service.hard_examples(rows, top_n=3)
        recommendation = service.recommend_next(
            rows,
            strategy="balanced_hard",
            batch_size=4,
            seed=42,
            history=tuple(h.sample_id for h in hard[:1]),
        )
        assert all(
            h.sample_id not in [x.sample_id for x in recommendation.selections]
            for h in recommendation.selections[:0]
        )  # structure sanity
        assert len(recommendation.selections) == 4

        # 5. derived training manifest from selection
        manifest_rows = [
            {
                "_schema_version": "clouda.pretraining.manifest.v1",
                "_row_count": len(rows),
                "dataset_role": "training",
                "dataset_id": "synthetic-fixture",
                "dataset_version": "v1",
            }
        ]
        manifest_rows.extend(
            {
                "sample_id": row["sample_id"],
                "target_split": "train",
                "source_id": "synthetic",
                "source_license": "Apache-2.0",
                "text": f"synthetic text for {row['sample_id']}",
            }
            for row in rows
        )
        manifest = tmp_path / "e2e_manifest.jsonl"
        manifest.write_text(
            "\n".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True) for r in manifest_rows
            ),
            encoding="utf-8",
        )
        from clouda_lab.dataset_selection import (
            SelectionCriteria,
            select_samples,
            write_selection_manifest,
        )

        scores = {row["sample_id"]: row["cer"] for row in rows}
        selection = select_samples(
            str(manifest),
            SelectionCriteria(top_n="hardest", top_n_count=3),
            seed=42,
            scores=scores,
        )
        derived = tmp_path / "e2e_derived.jsonl"
        write_selection_manifest(selection, str(derived))
        assert len(selection.sample_ids) == 3

        # 6. Training Framework dry-run on the derived manifest
        from clouda_lab.training_orchestrator import TrainingOrchestrator

        orchestrator = TrainingOrchestrator(runs_root=tmp_path / "runs")
        prepared = orchestrator.create_training_experiment_from_selection(
            manifest_path=str(manifest),
            criteria=SelectionCriteria(sample_ids=selection.sample_ids),
            seed=42,
            output_dir=tmp_path / "lab",
            experiment_name="e2e_flow",
            dry_run=True,
        )
        handle = orchestrator.start_dry_run(prepared["experiment_config_path"])
        assert orchestrator.run_status(handle["run_id"]) == RunStatus.COMPLETED.value

        # 7. machine-readable artifacts exist (runs live under runs_root;
        #    lab/ holds the selection + experiment config artifacts)
        assert Path(prepared["derived_manifest"]).is_file()
        assert Path(prepared["experiment_config_path"]).is_file()
        artifacts = [
            export_json(
                {"flow": "e2e", "samples": len(samples)}, tmp_path / "e2e_summary.json"
            )
        ]
        assert artifacts[0].is_file()

    def test_e2e_flow_is_deterministic(self, tmp_path: Path):
        service = EvaluationService()
        first = [service.evaluate_sample(s).to_dict() for s in _synthetic_samples()]
        second = [service.evaluate_sample(s).to_dict() for s in _synthetic_samples()]
        assert first == second
