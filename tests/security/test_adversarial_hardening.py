from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings, strategies as st
from PIL import Image

from clouda_contracts.archive_security import validate_zip_archive
from clouda_contracts.storage import StorageRoots, StorageSecurityError
from clouda_data.datasets.downloader import download_http
from clouda_data.distortion.workflow import generate_preview
from clouda_data.evaluation.aggregations import PageMetric
from clouda_data.evaluation.reports import write_csv_report
from clouda_data.ingestion.workflow import validate_source_manifest_file
from clouda_data.lifecycle import archive_run
from clouda_data.rendering.pipeline import RenderConfig
from clouda_models.local_models import resolve_local_checkpoint
from clouda_models.metadata import ModelMetadata
from clouda_training.exporter import export_training_data
from pdfword.backup import restore_backup, validate_backup
from pdfword.database import Database
from pdfword.docx_export import markdown_to_docx
from pdfword.local_ocr_adapters import (
    CommandLineOCRProvider,
    LocalOCRConfig,
    TransformersVisionLanguageProvider,
)
from pdfword.models import PageResult
from pdfword.operations import redact


def _roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> StorageRoots:
    monkeypatch.setenv("CLOUDA_STATE_HOME", str(tmp_path / "state"))
    roots = StorageRoots.from_env(read_only=False, create=True)
    return roots


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(path)


@pytest.mark.parametrize(
    "uri",
    [
        "dataset://folder/file.txt:secret",
        "dataset://CON",
        "dataset://folder/trailing.",
        "dataset://folder/trailing%20",
    ],
)
def test_storage_uri_rejects_windows_ambiguous_components(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, uri: str
) -> None:
    roots = _roots(monkeypatch, tmp_path)
    with pytest.raises(StorageSecurityError):
        roots.resolve_uri(uri)


def test_preview_rejects_manifest_uri_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = _roots(monkeypatch, tmp_path)
    outside = roots.dataset_root.parent / "outside.png"
    _png(outside)
    manifest = roots.dataset_root / "malicious" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "source_uri": "dataset://../outside.png",
                "output_uri": "dataset://../outside.png",
                "source_page_id": "page",
                "generated_page_id": "generated",
                "profile_id": "test",
                "overall_severity": "light",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((StorageSecurityError, PermissionError)):
        generate_preview(manifest)


def test_manifest_cannot_forge_commercial_training_license(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = _roots(monkeypatch, tmp_path)
    manifest = roots.dataset_root / "forged.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "attacker_controlled_dataset",
                "source_document_id": "doc",
                "source_page_id": "page",
                "generated_page_id": "generated",
                "output_uri": "dataset://pages/page.png",
                "output_checksum": "a" * 64,
                "ground_truth_text": "نص",
                "profile_id": "test",
                "license_status": "approved",
                "commercial_training_allowed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((PermissionError, KeyError)):
        export_training_data(
            manifest,
            roots.artifact_root / "forged.jsonl",
            purpose="commercial_training",
        )


def test_dataset_uri_statistics_cannot_escape_model_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = _roots(monkeypatch, tmp_path)
    outside = roots.model_root.parent / "outside-model"
    outside.mkdir()
    metadata = ModelMetadata(
        model_id="test",
        base_model_name="test",
        model_revision="sha256:" + "a" * 64,
        license="test",
        model_type="vision",
        parameter_count=1,
        architecture="dense",
        active_parameter_count=1,
        supported_image_inputs=("png",),
        tokenizer_revision="fixed",
        processor_revision="fixed",
        training_method="none",
        dataset_manifest_ids=(),
        checkpoint_uri="model://../outside-model",
        deployment_status="approved",
        commercial_use_status="approved",
        attribution_requirements="",
    )
    with pytest.raises(StorageSecurityError):
        resolve_local_checkpoint(metadata, roots=roots)


def test_downloader_blocks_private_destination_before_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    monkeypatch.delenv("CLOUDA_ALLOW_PRIVATE_DOWNLOADS", raising=False)
    monkeypatch.setattr("clouda_data.datasets.downloader._urlopen", forbidden)
    with pytest.raises(PermissionError):
        download_http(
            "http://127.0.0.1:9/forbidden",
            tmp_path / "download.bin",
            max_bytes=1024,
        )
    assert called is False


def test_command_ocr_uses_sanitized_environment_and_bounded_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = Path(sys.executable).resolve()
    monkeypatch.setenv("CLOUDA_LOCAL_OCR_COMMAND", json.dumps([str(executable)]))
    monkeypatch.setenv("CLOUDA_LOCAL_OCR_ALLOWED_EXECUTABLES", str(executable))
    monkeypatch.setenv("CLOUDA-SECURITY-SENSITIVE-MARKER", "must-not-leak")
    observed: dict = {}

    class Completed:
        returncode = 0
        stdout = "ok"

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        kwargs["stdout"].write(b"ok")
        kwargs["stdout"].flush()
        return Completed()

    monkeypatch.setattr("pdfword.local_ocr_adapters.subprocess.run", fake_run)
    provider = CommandLineOCRProvider(
        LocalOCRConfig(enabled=True, engine="command_line")
    )
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(image, "PNG")
    result = provider.extract_page(image_bytes=image.getvalue(), page_no=1)
    assert result.success
    assert "CLOUDA-SECURITY-SENSITIVE-MARKER" not in observed["env"]
    assert hasattr(observed["stdout"], "write")
    assert hasattr(observed["stderr"], "write")


def test_transformers_model_must_stay_in_model_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _roots(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises((PermissionError, StorageSecurityError)):
        TransformersVisionLanguageProvider(
            LocalOCRConfig(enabled=True, engine="transformers", model_path=str(outside))
        )


def test_docx_strips_xml_controls_and_has_no_active_relationships() -> None:
    payload = markdown_to_docx(
        [PageResult(1, "synthetic", "safe\x00text\x0b<script>alert(1)</script>")]
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert "word/vbaProject.bin" not in archive.namelist()
        document = archive.read("word/document.xml")
        assert b"\x00" not in document and b"\x0b" not in document
        relationships = b"".join(
            archive.read(name) for name in archive.namelist() if name.endswith(".rels")
        )
        assert b'TargetMode="External"' not in relationships


def test_csv_cells_cannot_execute_formulas(tmp_path: Path) -> None:
    output = tmp_path / "report.csv"
    write_csv_report(
        [
            PageMetric(
                page_id='=HYPERLINK("https://invalid")',
                profile="+cmd",
                severity="-1+1",
                cer=0,
                wer=0,
                dataset_id="@SUM(1,1)",
                document_type="body",
                quality_class="easy",
                language="ar",
            )
        ],
        output,
    )
    row = output.read_text(encoding="utf-8").splitlines()[1]
    assert "'=HYPERLINK" in row
    assert "'+cmd" in row
    assert "'-1+1" in row
    assert "'@SUM" in row


def test_zip_duplicate_and_case_collision_are_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicates.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/file.txt", "one")
        archive.writestr("DATA/FILE.TXT", "two")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError):
            validate_zip_archive(archive)


def test_backup_rejects_duplicate_database_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/clouda.sqlite3", b"one")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("data/clouda.sqlite3", b"two")
    with pytest.raises(ValueError):
        validate_backup(archive_path)


def test_backup_restore_refuses_symlinked_destination(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "source.sqlite3"
    Database(database_path)
    archive_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(database_path, "data/clouda.sqlite3")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "restore"
    try:
        target.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable for this local account")
    with pytest.raises((PermissionError, FileExistsError)):
        restore_backup(archive_path, target)


def test_ingestion_manifest_cannot_read_outside_its_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    manifest_dir = project / "incoming"
    manifest_dir.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "test",
                "documents": [
                    {
                        "document_id": "doc",
                        "source_path": str(outside),
                        "source_type": "text",
                        "language": "ar",
                        "source_license": "test",
                    }
                ],
                "pages": [
                    {
                        "document_id": "doc",
                        "page_id": "page",
                        "source_path": str(outside),
                        "source_type": "text",
                        "language": "ar",
                        "page_number": 1,
                        "clean_text": "safe",
                        "source_license": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = validate_source_manifest_file(manifest, project)
    assert not plan.ok
    assert any(issue.code == "unsafe_source_path" for issue in plan.issues)


def test_dynamic_database_fields_are_allowlisted(tmp_path: Path) -> None:
    database = Database(tmp_path / "database.sqlite3")
    with pytest.raises(ValueError):
        database.create_conversion({"job_id) VALUES ('injected') --": "x"})
    with pytest.raises(ValueError):
        database.record_attempt({"conversion_id": 1, "evil_column": "x"})


def test_nested_and_variant_secret_names_are_redacted() -> None:
    protected = redact(
        {
            "nested": {
                "provider-api-token": "secret",  # pragma: allowlist secret
                "db_password_value": "secret",  # pragma: allowlist secret
            }
        }
    )
    assert protected["nested"]["provider-api-token"] == "[REDACTED]"
    assert protected["nested"]["db_password_value"] == "[REDACTED]"


def test_render_configuration_limits_total_pages() -> None:
    with pytest.raises(ValueError):
        RenderConfig(start_page=1, end_page=10_001).validate()


def test_archive_run_rejects_symlinked_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = _roots(monkeypatch, tmp_path)
    run = roots.dataset_root / "distorted" / "run"
    run.mkdir(parents=True)
    outside = roots.dataset_root.parent / "private.txt"
    outside.write_text("private", encoding="utf-8")
    link = run / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable for this local account")
    with pytest.raises(PermissionError):
        archive_run(run)


def test_worker_api_rejects_declared_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDA_MAX_REQUEST_BYTES", "1024")
    from pdfword.worker_api import app

    response = TestClient(app).post(
        "/internal/workers/status",
        headers={"Content-Length": "2048"},
        content=b"{}",
    )
    assert response.status_code == 413


def test_deployment_defaults_to_loopback_and_ci_actions_are_pinned() -> None:
    run = Path("deploy/linux/run.sh").read_text(encoding="utf-8")
    run_api = Path("deploy/linux/run_api.sh").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'BIND_ADDRESS="${CLOUDA_BIND_ADDRESS:-127.0.0.1}"' in run
    assert '--server.address="${BIND_ADDRESS}"' in run
    assert 'BIND_ADDRESS="${CLOUDA_API_BIND_ADDRESS:-127.0.0.1}"' in run_api
    assert '--host="${BIND_ADDRESS}"' in run_api
    assert "actions/checkout@v4" not in workflow
    assert "actions/setup-python@v5" not in workflow


@settings(
    max_examples=100,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.text(max_size=200))
def test_storage_uri_property_never_resolves_outside_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, component: str
) -> None:
    roots = _roots(monkeypatch, tmp_path)
    try:
        resolved = roots.resolve_uri("dataset://" + component)
    except (StorageSecurityError, ValueError, OSError):
        return
    assert resolved.is_relative_to(roots.dataset_root)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(st.text(max_size=500))
def test_document_sanitizer_property_emits_valid_xml_text(value: str) -> None:
    from clouda_contracts.security import sanitize_document_text

    sanitized = sanitize_document_text(value)
    assert all(ord(character) >= 32 or character in "\t\n\r" for character in sanitized)
    assert not any(
        "\u202a" <= character <= "\u202e" or "\u2066" <= character <= "\u2069"
        for character in sanitized
    )
