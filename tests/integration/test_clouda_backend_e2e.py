"""Offline CPU-only proof of the integrated Clouda backend chain."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from clouda_contracts.checksums import sha256_file
from clouda_data.factory.adapters import run_dir_to_dataset_manifest
from clouda_data.factory.factory import generate_run
from clouda_data.pretraining.manifest import read_manifest
from clouda_data.quality.derived import write_clean_manifest
from clouda_data.quality.gate import run_quality_gate
from clouda_data.quality.models import GateVerdict
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
from clouda_training.adapters.data_adapter import get_default_data_adapter_registry
from clouda_training.adapters.registry import get_default_registry
from clouda_training.experiments import load_experiment_config, run_experiment
from clouda_training.hunyuan.data_adapter import HUNYUAN_DATA_ADAPTER_TYPE
from clouda_training.hunyuan.descriptor import HUNYUAN_DESCRIPTOR
from clouda_training.hunyuan.models import (
    ARABIC_DOCUMENT_OCR_PROMPT,
    HunyuanExportConfig,
)
from clouda_training.hunyuan.registration import register_hunyuan_adapters
from clouda_training.planner.models import (
    ExperimentProfile,
    TrainingMode,
    TrainingScale,
)
from clouda_training.planner.planner import (
    build_experiment_plan,
    generate_training_config,
)
from clouda_training.preflight.models import PreflightFinalStatus
from clouda_training.preflight.orchestrator import run_preflight


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
        run_dir / "canonical.manifest.jsonl",
        source_id="clouda-backend-e2e",
    )
    assert report.samples == 1
    header, rows = read_manifest(manifest_path)
    return manifest_path, header, rows[0], manifest_sha


def test_tiny_offline_backend_flow_from_factory_to_training_lineage(
    tmp_path: Path,
) -> None:
    source_manifest, _, _, _ = _tiny_factory_dataset(tmp_path)
    quality_scan = run_quality_gate(str(source_manifest), no_near_duplicates=True)
    assert quality_scan.result.verdict is not GateVerdict.FAIL
    assert not quality_scan.exclusions
    quality_manifest = source_manifest.with_name("quality-clean.manifest.jsonl")
    quality_result = write_clean_manifest(
        source_manifest,
        quality_scan.samples,
        quality_scan.exclusions,
        quality_scan.quarantine_ids,
        quality_scan.run,
        quality_manifest,
    )
    header, quality_rows = read_manifest(quality_manifest)
    source_row = quality_rows[0]
    manifest_sha = sha256_file(quality_manifest)
    assert Path(quality_result["clean_manifest_path"]) == quality_manifest
    assert header["source_manifest_sha256"] == quality_scan.manifest_sha256
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
    loader = orchestrator.create_training_data_loader(prepared["training_data"])
    assert list(loader.iter_sample_ids(epoch=0)) == [page_id]

    # The selected dataset feeds the canonical planner and preflight before
    # the exact generated config is handed to the runtime.
    base_config = load_experiment_config(Path(prepared["experiment_config_path"]))
    profile = ExperimentProfile(
        scale=TrainingScale.SMOKE,
        sample_count=1,
        epochs=1,
        max_steps=4,
        micro_batch=1,
        gradient_accumulation=1,
        world_size=1,
        precision="float32",
        training_mode=TrainingMode.FULL_FINETUNE,
        checkpoint_interval=2,
    )
    plan = build_experiment_plan(base_config, profile, dataset_row_count=1)
    assert (
        plan.plan_id
        == build_experiment_plan(base_config, profile, dataset_row_count=1).plan_id
    )
    config = generate_training_config(plan, base_config=base_config)
    preflight = run_preflight(config, dataset_row_count=1, write_probe=False)
    assert preflight.final_status() in {
        PreflightFinalStatus.READY,
        PreflightFinalStatus.READY_WITH_WARNINGS,
    }, [blocker.reason for blocker in preflight.blockers]

    # Registry selection and the Hunyuan data bridge operate on the same Lab
    # derived rows without importing/downloading a real model stack.
    register_hunyuan_adapters()
    assert (
        get_default_registry().get(HUNYUAN_DESCRIPTOR.adapter_type)
        == HUNYUAN_DESCRIPTOR
    )
    hunyuan_data_adapter = get_default_data_adapter_registry().create(
        HUNYUAN_DATA_ADAPTER_TYPE
    )
    _, selected_rows = read_manifest(Path(prepared["derived_manifest"]))
    hunyuan_config = HunyuanExportConfig(
        dataset_id=prepared["training_data"]["dataset_id"],
        dataset_version=prepared["training_data"]["dataset_version"],
        manifest_path=prepared["derived_manifest"],
        manifest_hash=prepared["derived_manifest_sha256"],
        split="train",
        image_root=str(quality_manifest.parent.resolve()),
        prompt_profile=ARABIC_DOCUMENT_OCR_PROMPT,
    )
    hunyuan_records = hunyuan_data_adapter.export(selected_rows, hunyuan_config)
    assert hunyuan_data_adapter.validate(hunyuan_records)["valid"] is True
    assert Path(hunyuan_records[0]["image_path"][0]).is_file()
    assert (
        hunyuan_data_adapter.last_lineage_report["manifest_sha256"]
        == prepared["derived_manifest_sha256"]
    )

    interrupted_loader = orchestrator.create_training_data_loader(
        prepared["training_data"]
    )
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config, interrupt_at_step=3, data_loader=interrupted_loader)
    interrupted_path = next(
        (tmp_path / "training-runs" / config.experiment.name).iterdir()
    )
    interrupted_state = json.loads(
        (interrupted_path / "checkpoints" / "step-00000002" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    assert interrupted_state["data_cursor"]["yielded_count"] > 0

    training_run = orchestrator.resume(
        interrupted_path.name, training_data=prepared["training_data"]
    )
    assert training_run["status"] == "COMPLETED"
    assert config.runtime.dry_run is True
    assert config.model.adapter_type == "mock"
    run_path = Path(training_run["path"])
    run_metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
    assert (
        run_metadata["training_data"]["manifest_sha256"]
        == prepared["training_data"]["manifest_sha256"]
    )
    checkpoint_path = Path(
        orchestrator.get_checkpoints(training_run["run_id"])[-1]["path"]
    )
    checkpoint_state = json.loads(
        (checkpoint_path / "state.json").read_text(encoding="utf-8")
    )
    assert checkpoint_state["data_cursor"]["yielded_count"] > 0

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
