"""Offline CPU-only proof of the integrated Clouda backend chain."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from clouda_data.factory.adapters import run_dir_to_dataset_manifest
from clouda_data.factory.factory import generate_run
from clouda_data.pretraining.manifest import read_manifest
from clouda_data.results.identity import ArtifactRef
from clouda_data.results.ingest import training_run_to_model_record
from clouda_data.results.models import (
    ModelRecord,
    PageRecord,
    ProtectionInfo,
    Provenance,
)
from clouda_data.results.service import NewPrediction, ResultsService
from clouda_lab import SelectionCriteria, StoredResultsAnalysisService
from clouda_lab.training_orchestrator import TrainingOrchestrator


def _tiny_factory_dataset(tmp_path: Path) -> tuple[Path, dict, dict, str]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    image_path = source_dir / "arabic.png"
    image = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(image).text((20, 40), "000000", fill="black")
    image.save(image_path)
    (source_dir / "arabic.txt").write_text("هذا نص عربي تجريبي\n", encoding="utf-8")
    metadata = generate_run(
        inputs=[image_path],
        runs_root=tmp_path / "factory-runs",
        profile_names=["05_old_book_medium"],
        variants=1,
        base_seed=41,
        seed_mode="v1",
        backend="raqm",
        workers=1,
        export_pdf=True,
        export_png=True,
        max_pages=1,
    )
    run_dir = tmp_path / "factory-runs" / metadata["run_id"]
    manifest_path, report, manifest_sha = run_dir_to_dataset_manifest(
        run_dir,
        tmp_path / "canonical.manifest.jsonl",
        source_id="clouda-backend-e2e",
    )
    assert report.samples == 1
    header, rows = read_manifest(manifest_path)
    return manifest_path, header, rows[0], manifest_sha


def test_tiny_offline_backend_flow_from_factory_to_training_lineage(
    tmp_path: Path,
) -> None:
    manifest_path, header, source_row, manifest_sha = _tiny_factory_dataset(tmp_path)
    assert source_row["target_split"] == "train"
    page_id = source_row["sample_id"]
    raw_text = source_row["raw_text"]

    results = ResultsService(tmp_path / "results")
    results.register_dataset(
        dataset_id=header["dataset_id"],
        version=header["dataset_version"],
        manifest_sha256=manifest_sha,
        splits=("train",),
    )
    page = PageRecord(
        page_id=page_id,
        document_id=source_row["document_id"],
        dataset_id=header["dataset_id"],
        dataset_version=header["dataset_version"],
        split="train",
        page_number=1,
        image_artifact=ArtifactRef(
            artifact_id=f"factory-{page_id}",
            kind="page_image",
            uri=f"dataset://{source_row['image_path']}",
            sha256=source_row["file_sha256"],
            role="page_image",
        ),
        ground_truth_sha256=source_row["normalized_text_sha256"],
        profile=source_row["provenance"]["profile"],
        distortions=tuple(source_row["provenance"]["transform_steps"]),
        distortion_seed=source_row["provenance"]["seed"],
        source_identity={"source_record_id": source_row["source_record_id"]},
        protection=ProtectionInfo(split="train"),
        provenance=Provenance(
            source_format="clouda.factory.adapter.v1",
            source_uri="factory://canonical.manifest.jsonl",
            source_sha256=source_row["provenance"]["source_sha256"],
        ),
        metadata={"canonical_manifest_row": source_row},
    )

    run_ids: list[str] = []
    for model_id, revision, prediction_text, created_at in (
        ("fake/base", "v1", raw_text, "2026-09-10T01:00:00Z"),
        ("fake/candidate", "v2", "هذا نص خاطئ\n", "2026-09-10T01:01:00Z"),
    ):
        results.register_model(ModelRecord(model_id=model_id, revision=revision))
        run = results.create_run(
            model_id=model_id,
            model_revision=revision,
            dataset_id=header["dataset_id"],
            dataset_version=header["dataset_version"],
            split="train",
            manifest_sha256=manifest_sha,
            created_at=created_at,
        )
        results.add_page(run.run_id, page)
        results.add_ground_truth(run.run_id, page, raw_text)
        prediction = results.add_prediction(
            run.run_id,
            NewPrediction(
                page_id=page_id,
                text=prediction_text,
                model_id=model_id,
                model_revision=revision,
            ),
        )
        results.evaluate_prediction(run.run_id, prediction)
        results.finalize_run(run.run_id)
        assert results.verify(run.run_id)["ok"] is True
        run_ids.append(run.run_id)

    bridge = StoredResultsAnalysisService(results)
    assert bridge.sample(run_ids[1], page_id).ground_truth == raw_text
    assert bridge.analyze_page(run_ids[1], page_id).cer > 0.0
    comparison = bridge.compare_runs(run_ids[0], run_ids[1])
    assert comparison.comparisons[0].classification in {"regressed", "newly_failed"}
    recommendations = bridge.recommend_next(
        run_ids[1], strategy="hardest_only", batch_size=1, seed=41
    )
    assert recommendations[0]["sample_id"] == page_id
    selection = bridge.select_pages(
        run_ids[1], SelectionCriteria(split="train"), seed=41
    )
    assert selection.sample_ids == (page_id,)
    assert selection.rows[0]["training_eligible"] is True

    orchestrator = TrainingOrchestrator(tmp_path / "training-runs")
    prepared = orchestrator.create_training_experiment_from_result(
        selection=selection,
        seed=41,
        output_dir=tmp_path / "training-config",
        experiment_name="clouda-backend-e2e",
    )
    training_run = orchestrator.start_dry_run(prepared["experiment_config_path"])
    assert training_run["status"] == "COMPLETED"
    assert prepared["experiment_config"]["runtime"]["dry_run"] is True
    assert prepared["experiment_config"]["model"]["adapter_type"] == "mock"

    trained_model = training_run_to_model_record(Path(training_run["path"]))
    results.register_model(trained_model)
    linked_run = results.create_run(
        model_id=trained_model.model_id,
        model_revision=trained_model.revision,
        dataset_id=header["dataset_id"],
        dataset_version=header["dataset_version"],
        manifest_sha256=manifest_sha,
        training_lineage=trained_model.training_lineage,
        created_at="2026-09-10T01:02:00Z",
    )
    stored_linkage = results.get_run(linked_run.run_id)["training_lineage"]
    assert stored_linkage["training_run_id"] == training_run["run_id"]
    assert stored_linkage["manifest_sha256"] == prepared["derived_manifest_sha256"]
