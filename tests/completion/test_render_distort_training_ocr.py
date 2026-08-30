from __future__ import annotations

import json
import shutil
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

from clouda_data.distortion.base import DistortionSpec
from clouda_data.distortion.checkpoints import RunCheckpointStore
from clouda_data.distortion.registry import default_registry
from clouda_data.distortion.workflow import (
    classify_visual_difficulty,
    generate_preview,
    read_jsonl,
    run_distortion_batch,
    validate_distortion_manifest,
)
from clouda_data.evaluation.execution import evaluate_records
from clouda_data.pipeline.profiles import (
    list_profile_paths,
    load_profile,
    validate_profile,
)
from clouda_data.rendering import (
    RenderConfig,
    render_document,
    validate_render_manifest,
)
from clouda_training.exporter import export_training_data
from pdfword.local_ocr_adapters import (
    CommandLineOCRProvider,
    LocalHTTPProvider,
    LocalOCRConfig,
    MockOCRProvider,
    TransformersVisionLanguageProvider,
    model_revision,
    provider_from_config,
)
from pdfword.ocr_pipeline import process_pdf


@pytest.fixture()
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "state"
    monkeypatch.setenv("CLOUDA_STATE_HOME", str(root))
    monkeypatch.setenv("CLOUDA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    for name in ("datasets", "artifacts", "cache", "models", "runtime"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _synthetic_page(path: Path) -> None:
    image = Image.new("RGB", (640, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 600, 760), outline="black", width=3)
    draw.text((80, 100), "Arabic synthetic test 123", fill="black")
    draw.text((80, 700), "footnote 1", fill="black")
    image.save(path)


def test_all_registered_real_operators_execute_and_are_deterministic() -> None:
    image = Image.new("RGB", (96, 96), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 75, 75), fill="black")
    registry = default_registry()
    names = [name for name in registry.names() if name != "metadata_only"]
    assert len(names) >= 20
    changed = 0
    for name in names:
        operator = registry.create(DistortionSpec(name, severity="light"))
        first, first_meta = operator.apply(image, 12345)
        second, second_meta = operator.apply(image, 12345)
        assert first.tobytes() == second.tobytes(), name
        assert first_meta.random_seed == second_meta.random_seed == 12345
        assert first.width > 0 and first.height > 0
        if first.size != image.size or ImageChops.difference(first, image).getbbox():
            changed += 1
    assert changed >= 20


def test_required_yaml_profiles_validate() -> None:
    paths = [path for path in list_profile_paths() if path.suffix == ".yaml"]
    assert len(paths) == 18
    for path in paths:
        profile = load_profile(path)
        assert profile["schema_version"] == 1
        assert profile["random_seed_policy"] == "required_deterministic"
        assert len(profile["transformations"]) <= profile["maximum_chain_length"]
        assert profile["maximum_severity_budget"] > 0


def test_required_yaml_profiles_and_schema_are_packaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("clouda_data.locations.repository_root", lambda: None)
    packaged = [path for path in list_profile_paths() if path.suffix == ".yaml"]
    assert len(packaged) == 18
    for path in packaged:
        assert load_profile(path)["profile_id"]


def test_profile_rejects_unknown_operator_and_resource_exhaustion() -> None:
    base = {
        "name": "unsafe",
        "recommended_use": "test",
        "ordering_constraints": [],
        "mutually_exclusive": [],
        "maximum_allowed_crop": 0.05,
        "minimum_readable_text_threshold": 0.7,
        "metadata_fields": ["random_seed"],
        "maximum_chain_length": 1,
        "maximum_severity_budget": 0.5,
    }
    with pytest.raises(ValueError, match="Unknown distortion operator"):
        validate_profile(
            {
                **base,
                "distortions": [{"name": "not_registered", "severity": "light"}],
            }
        )
    with pytest.raises(ValueError, match="maximum_chain_length"):
        validate_profile(
            {
                **base,
                "distortions": [
                    {"name": "rotation", "severity": "light"},
                    {"name": "gaussian_blur", "severity": "light"},
                ],
            }
        )
    with pytest.raises(ValueError, match="severity budget"):
        validate_profile(
            {
                **base,
                "distortions": [{"name": "rotation", "severity": "extreme"}],
            }
        )


def test_yaml_profile_loader_rejects_python_object_tags(tmp_path: Path) -> None:
    payload = tmp_path / "unsafe.yaml"
    payload.write_text(
        "!!python/object/apply:os.system ['echo unsafe']\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_profile(payload)


def test_checkpoint_state_machine_recovers_stale_and_enforces_retry_limit(
    tmp_path: Path,
) -> None:
    store = RunCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store.start_run(
        "run-1",
        input_checksum="input",
        profile_hash="profile",
        metadata={"test": True},
        resume=False,
    )
    store.queue_page(
        run_id="run-1",
        generated_page_id="page-1",
        source_page_id="source-1",
        variant=0,
        max_retries=1,
    )
    assert store.claim_page("page-1")
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with store.connect() as connection, connection:
        connection.execute(
            "UPDATE pages SET heartbeat_at = ? WHERE generated_page_id = ?",
            (old, "page-1"),
        )
    assert store.recover_stale(stale_after_seconds=60) == {
        "requeued": 1,
        "failed": 0,
    }
    assert store.claim_page("page-1")
    store.finish_page("page-1", status="failed", error="controlled failure")
    assert not store.claim_page("page-1")
    page = store.page("page-1")
    assert page is not None
    assert page["retry_count"] == 1


def test_visual_difficulty_emits_extended_signals() -> None:
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).text((12, 48), "OCR 123", fill="black")
    metrics = classify_visual_difficulty(
        image,
        "medium",
        {
            "transformation_chain": ["rotation"],
            "transformation_parameters": [{"degrees": 3.5}],
            "regions": [{"bbox": [0, 0, 100, 20]}],
        },
    )
    assert metrics["estimated_visual_difficulty"] in {
        "easy",
        "medium",
        "difficult",
        "extreme",
        "invalid",
    }
    assert metrics["estimated_skew_degrees"] == 3.5
    assert metrics["estimated_text_size"] == 20
    for key in (
        "blur_score",
        "noise_level",
        "edge_density",
        "compression_artifact_score",
        "foreground_ratio",
    ):
        assert isinstance(metrics[key], float)


def test_render_image_is_copy_on_write_and_valid(state: Path) -> None:
    source = state / "datasets" / "source.png"
    _synthetic_page(source)
    before = source.read_bytes()
    manifest = render_document(
        source,
        config=RenderConfig(dpi=144, color_mode="grayscale"),
    )
    assert validate_render_manifest(manifest)["passed"]
    assert source.read_bytes() == before
    record = read_jsonl(manifest)[0]
    assert record["schema_version"] == 1
    assert record["source_checksum"]
    assert record["output_checksum"]


def test_render_resume_recovers_orphan_output_and_rejects_corruption(
    state: Path,
) -> None:
    source = state / "datasets" / "resume-source.png"
    _synthetic_page(source)
    config = RenderConfig(dpi=144, color_mode="grayscale")
    manifest = render_document(source, config=config)
    record = read_jsonl(manifest)[0]
    output = state / "datasets" / str(record["output_uri"]).removeprefix("dataset://")

    manifest.unlink()
    recovered = render_document(
        source,
        config=RenderConfig(dpi=144, color_mode="grayscale", resume=True),
        run_id=str(record["run_id"]),
    )
    assert read_jsonl(recovered)[0]["output_checksum"] == record["output_checksum"]

    output.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        render_document(
            source,
            config=RenderConfig(dpi=144, color_mode="grayscale", resume=True),
            run_id=str(record["run_id"]),
        )


def test_render_pdf_page(state: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "digital_text.pdf"
    source = state / "datasets" / "digital_text.pdf"
    shutil.copy2(fixture, source)
    manifest = render_document(
        source,
        config=RenderConfig(start_page=1, end_page=1, dpi=100),
    )
    assert validate_render_manifest(manifest)["passed"]
    assert len(read_jsonl(manifest)) == 1


def test_distortion_batch_resume_validate_preview_and_export(state: Path) -> None:
    source = state / "datasets" / "clean.png"
    _synthetic_page(source)
    source_checksum = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    input_manifest = state / "datasets" / "synthetic_manifest.jsonl"
    input_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "output_uri": "dataset://clean.png",
                "output_checksum": source_checksum,
                "source_document_id": "synthetic-doc-1",
                "source_page_number": 1,
                "page_id": "synthetic-doc-1:1",
                "dataset_id": "synthetic_evaluation",
                "license_status": "evaluation_only",
                "commercial_training_allowed": False,
                "ground_truth_reference": "synthetic://text/1",
                "ground_truth_text": "نص عربي اصطناعي",
                "attribution": "Generated synthetic fixture",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    profile = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "data_foundation"
        / "distortions"
        / "modern_scan_medium.yaml"
    )
    manifest = run_distortion_batch(
        input_manifest,
        profile,
        seed=77,
        variants=3,
        max_pages=1,
    )
    first = read_jsonl(manifest)
    assert len(first) == 3
    assert all(item["ground_truth_text"] == "نص عربي اصطناعي" for item in first)
    assert all(item["output_checksum"] != source_checksum for item in first)
    assert all(item["run_id"] and item["variant"] is not None for item in first)
    assert all(
        item["parameters"] == item["transformation_parameters"] for item in first
    )
    checkpoint = RunCheckpointStore(manifest.parent / "checkpoints.sqlite3")
    summary = checkpoint.summary(first[0]["run_id"])
    assert summary["state"] == "complete"
    assert sum(summary["statuses"].values()) == 3
    resumed = run_distortion_batch(
        input_manifest,
        profile,
        seed=77,
        variants=3,
        max_pages=1,
        resume=True,
    )
    assert resumed == manifest
    assert len(read_jsonl(resumed)) == 3
    assert validate_distortion_manifest(manifest)["passed"]
    preview = generate_preview(manifest, limit=2, layout_overlay=True)
    assert preview.is_file()
    assert (preview.parent / "contact-sheet.jpg").is_file()
    assert "Seeds" in preview.read_text(encoding="utf-8")
    export = state / "artifacts" / "training" / "synthetic.jsonl"
    summary = export_training_data(
        manifest,
        export,
        purpose="evaluation",
        export_format="conversation_multimodal",
    )
    assert summary["records"] == 3
    assert not summary["document_leakage"]
    assert summary["training_started"] is False
    capped = export_training_data(
        manifest,
        state / "artifacts" / "training" / "capped.jsonl",
        purpose="evaluation",
        max_contribution_per_source=1,
    )
    assert capped["records"] == 1
    assert capped["balanced"] is True
    perceptual = export_training_data(
        manifest,
        state / "artifacts" / "training" / "perceptual.jsonl",
        purpose="evaluation",
        perceptual_duplicate=lambda candidate, existing: True,
    )
    assert perceptual["records"] == 1
    assert perceptual["duplicates_rejected"] == 2

    invalid_root = state / "datasets" / "invalid-run"
    invalid_root.mkdir()
    invalid_manifest = invalid_root / "distortion_manifest.v1.jsonl"
    invalid_record = {**first[0], "output_checksum": "0" * 64}
    invalid_manifest.write_text(
        json.dumps(invalid_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validation = validate_distortion_manifest(invalid_manifest, quarantine=True)
    assert not validation["passed"]
    assert validation["failures"][0]["quarantine_uri"].startswith(
        "dataset://quarantine/"
    )
    quarantine_path = (
        state
        / "datasets"
        / validation["failures"][0]["quarantine_uri"].replace("dataset://", "")
    )
    assert quarantine_path.is_file()
    assert (
        state / "artifacts" / "reports" / "validation" / "invalid-run.jsonl"
    ).is_file()


def test_distortion_resume_recovers_missing_manifest_and_reconciles_checkpoint(
    state: Path,
) -> None:
    source = state / "datasets" / "resume-clean.png"
    _synthetic_page(source)
    source_checksum = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    input_manifest = state / "datasets" / "resume-input.jsonl"
    input_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "output_uri": "dataset://resume-clean.png",
                "output_checksum": source_checksum,
                "source_document_id": "resume-document",
                "source_page_number": 1,
                "page_id": "resume-document:1",
                "dataset_id": "synthetic_evaluation",
                "license_status": "evaluation_only",
                "commercial_training_allowed": False,
                "ground_truth_reference": "synthetic://resume/1",
                "ground_truth_text": "نص استئناف اصطناعي",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    profile = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "data_foundation"
        / "distortions"
        / "modern_scan_light.yaml"
    )
    manifest = run_distortion_batch(
        input_manifest,
        profile,
        seed=91,
        variants=1,
        max_pages=1,
    )
    first = read_jsonl(manifest)[0]
    checkpoint = RunCheckpointStore(manifest.parent / "checkpoints.sqlite3")

    with checkpoint.connect() as connection, connection:
        connection.execute(
            "UPDATE pages SET status = 'processing' WHERE generated_page_id = ?",
            (first["generated_page_id"],),
        )
    run_distortion_batch(
        input_manifest,
        profile,
        seed=91,
        variants=1,
        max_pages=1,
        resume=True,
    )
    checkpoint_page = checkpoint.page(first["generated_page_id"])
    assert checkpoint_page is not None
    assert checkpoint_page["status"] == first["status"]

    manifest.unlink()
    recovered = run_distortion_batch(
        input_manifest,
        profile,
        seed=91,
        variants=1,
        max_pages=1,
        resume=True,
    )
    recovered_record = read_jsonl(recovered)[0]
    assert recovered_record["generated_page_id"] == first["generated_page_id"]
    assert recovered_record["output_checksum"] == first["output_checksum"]


def test_commercial_export_fails_closed_for_synthetic(state: Path) -> None:
    manifest = state / "datasets" / "records.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "ground_truth_text": "x",
                "license_status": "evaluation_only",
                "commercial_training_allowed": False,
                "dataset_id": "synthetic_evaluation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        export_training_data(
            manifest,
            state / "artifacts" / "blocked.jsonl",
        )


def test_commercial_export_fails_closed_for_unknown_dataset(state: Path) -> None:
    manifest = state / "datasets" / "unknown-records.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "ground_truth_text": "x",
                "license_status": "unknown",
                "commercial_training_allowed": False,
                "dataset_id": "unknown_external_dataset",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="approved license catalog"):
        export_training_data(
            manifest,
            state / "artifacts" / "blocked-unknown.jsonl",
        )


def test_evaluation_metrics_are_executed() -> None:
    report = evaluate_records(
        [
            {
                "page_id": "1",
                "reference_text": "مرحبا بالعالم",
                "prediction_text": "مرحبا بالعالم",
                "processing_time": 0.5,
            },
            {
                "page_id": "2",
                "reference_text": "نص",
                "prediction_text": "",
                "processing_time": 0.5,
            },
        ]
    )
    assert report["summary"]["pages"] == 2
    assert report["summary"]["cer"] > 0
    assert report["summary"]["missing_text_rate"] == 0.5


def test_mock_local_ocr_runtime_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDA_LOCAL_OCR_ENABLED", "true")
    monkeypatch.setenv("CLOUDA_LOCAL_OCR_ENGINE", "mock")
    monkeypatch.setenv("CLOUDA_ALLOW_MOCK_OCR", "true")
    monkeypatch.setenv("CLOUDA_MOCK_OCR_TEXT", "نص ممسوح")
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "scanned.pdf"
    results, _ = process_pdf(
        fixture.read_bytes(),
        1,
        1,
        enabled_engines=["direct_pdf_text", "local_model_ocr"],
    )
    assert results[0].markdown == "نص ممسوح"
    assert results[0].metadata is not None
    assert results[0].metadata["page_state"] == "local_model_ocr"
    assert results[0].accepted


def test_mock_provider_requires_explicit_test_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDA_ALLOW_MOCK_OCR", raising=False)
    assert not MockOCRProvider().available()


def test_remote_http_endpoint_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDA_LOCAL_OCR_ALLOW_REMOTE_ENDPOINT", raising=False)
    config = LocalOCRConfig(
        enabled=True, engine="local_http", endpoint="http://example.com"
    )
    with pytest.raises(ValueError):
        LocalHTTPProvider(config)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(output, "PNG")
    return output.getvalue()


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode()
        self.length = len(self.payload)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.payload


def test_local_http_and_openai_compatible_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LocalOCRConfig(
        enabled=True, engine="local_http", endpoint="http://127.0.0.1:8001/ocr"
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response({"text": "نص", "confidence": 0.8}),
    )
    result = LocalHTTPProvider(config).extract_page(image_bytes=_png_bytes(), page_no=2)
    assert result.success and result.text == "نص" and result.confidence == 0.8
    openai = LocalOCRConfig(
        enabled=True,
        engine="openai_compatible",
        endpoint="http://localhost:8001/v1",
        model_path="local-test",
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(
            {"choices": [{"message": {"content": "markdown"}}], "confidence": 0.7}
        ),
    )
    result = LocalHTTPProvider(openai, openai_compatible=True).extract_page(
        image_bytes=_png_bytes(), page_no=1
    )
    assert result.success and result.text == "markdown"


def test_local_http_adapter_returns_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    config = LocalOCRConfig(enabled=True, engine="local_http")

    def fail(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    result = LocalHTTPProvider(config).extract_page(image_bytes=_png_bytes(), page_no=1)
    assert not result.success
    assert "URLError" in (result.error_message or "")


def test_command_line_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "ocr.py"
    script.write_text("print('synthetic cli text')\n", encoding="utf-8")
    monkeypatch.setenv(
        "CLOUDA_LOCAL_OCR_COMMAND",
        json.dumps([str(Path(sys.executable).resolve()), str(script)]),
    )
    monkeypatch.setenv(
        "CLOUDA_LOCAL_OCR_ALLOWED_EXECUTABLES", str(Path(sys.executable).resolve())
    )
    provider = CommandLineOCRProvider(
        LocalOCRConfig(enabled=True, engine="command_line")
    )
    assert provider.available()
    result = provider.extract_page(image_bytes=_png_bytes(), page_no=3)
    assert result.success and result.text == "synthetic cli text"


def test_transformers_adapter_fails_closed_without_local_model() -> None:
    provider = TransformersVisionLanguageProvider(
        LocalOCRConfig(enabled=True, engine="transformers")
    )
    assert not provider.available()
    result = provider.extract_page(image_bytes=_png_bytes(), page_no=1)
    assert not result.success
    assert "local model directory" in (result.error_message or "")


def test_local_ocr_config_factory_and_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUDA_LOCAL_OCR_ENABLED", "true")
    monkeypatch.setenv("CLOUDA_LOCAL_OCR_ENGINE", "mock")
    config = LocalOCRConfig.from_env()
    config.validate()
    assert isinstance(provider_from_config(config), MockOCRProvider)
    assert model_revision(config) == "test-fixture-v1"
    model = tmp_path / "model"
    model.mkdir()
    configured = LocalOCRConfig(
        enabled=True, engine="transformers", model_path=str(model)
    )
    assert model_revision(configured) != "unresolved"
    with pytest.raises(ValueError):
        LocalOCRConfig(enabled=True, engine="unknown").validate()
    with pytest.raises(ValueError):
        LocalOCRConfig(enabled=True, engine="mock", batch_size=99).validate()
