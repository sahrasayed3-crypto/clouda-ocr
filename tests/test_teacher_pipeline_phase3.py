from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from pdfword.database import Database, SCHEMA_VERSION
from pdfword.key_router.audit import sanitize_metadata
from pdfword.key_router.capabilities import ModelEndpointCapability
from pdfword.key_router.enums import Modality, PrivacyClassification
from pdfword.key_router.exceptions import UnsupportedCapability
from pdfword.key_router.models import RequestContext
from pdfword.teacher_pipeline.config import (
    TeacherPipelineConfig,
    TeacherPipelineConfigurationError,
)
from pdfword.teacher_pipeline.contracts import (
    CONTRACT_VERSION,
    GROUNDING_RULES,
    PROMPT_VERSION,
    build_task,
    contract_catalog,
)
from pdfword.teacher_pipeline.enums import (
    DatasetDecision,
    DispatchMode,
    TeacherPipelineMode,
    TeacherRole,
)
from pdfword.teacher_pipeline.manifests import (
    DATASET_LAYOUT,
    ensure_dataset_layout,
    hamming_distance,
    perceptual_dhash,
    validate_manifest_items,
    write_manifest,
)
from pdfword.teacher_pipeline.models import (
    PageAnalysis,
    Region,
    SourceRightsRecord,
    TeacherRoleAssignment,
)
from pdfword.teacher_pipeline.preparation import prepare_regions, sha256_file
from pdfword.teacher_pipeline.repository import TeacherRepository
from pdfword.teacher_pipeline.scoring import decide_dataset, score_pair
from pdfword.teacher_pipeline.service import TeacherPipeline, safe_status
from pdfword.worker_api import app
from tools.validation.repository_scan import _secret_findings


def image_page(path: Path, text: str = "ABC") -> Path:
    image = Image.new("RGB", (160, 100), "white")
    ImageDraw.Draw(image).text((20, 30), text, fill="black")
    image.save(path)
    return path


def page_analysis() -> PageAnalysis:
    region = Region("main-1", "main_text", (10, 10, 150, 80), 0, 1.0)
    return PageAnalysis(
        page_id="page-1",
        source_document_id="doc-1",
        page_type="text",
        primary_language="ar",
        secondary_languages=(),
        is_born_digital=False,
        is_scanned=True,
        quality_score=95.0,
        orientation=0,
        deskew_needed=False,
        layout_type="single_column",
        column_count=1,
        reading_order=("main-1",),
        main_text_regions=(region,),
        recommended_teacher_roles=(TeacherRole.ARABIC_MAIN_OCR,),
        analysis_confidence=1.0,
    )


def rights(path: Path, **overrides) -> SourceRightsRecord:
    values: dict[str, Any] = {
        "source_document_id": "doc-1",
        "source_page_id": "page-1",
        "source_hash": sha256_file(path),
        "rights_record_id": "rights-1",
        "rights_status": "verified",
        "allowed_for_processing": True,
        "allowed_for_training": True,
        "allowed_for_commercial_training": False,
        "allowed_for_weight_release": False,
        "allowed_for_data_redistribution": False,
        "permission_source": "authorized-test-fixture",
        "permission_version": "v1",
        "permission_date": "2026-07-27",
        "data_classification": "internal",
        "retention_policy": "test-only",
    }
    values.update(overrides)
    return SourceRightsRecord(**values)


def assignment(
    *,
    assignment_id: str = "assignment-1",
    account_id: str = "account-1",
    owner_id: str = "owner-1",
    quota_domain_id: str = "quota-1",
    priority: int = 10,
) -> TeacherRoleAssignment:
    return TeacherRoleAssignment(
        assignment_id=assignment_id,
        account_id=account_id,
        owner_id=owner_id,
        provider_id="google_gemini",
        quota_domain_id=quota_domain_id,
        allowed_teacher_roles=(TeacherRole.ARABIC_MAIN_OCR,),
        allowed_task_types=(TeacherRole.ARABIC_MAIN_OCR.value,),
        allowed_data_classifications=("internal",),
        enabled=True,
        dispatch_mode=DispatchMode.MOCK,
        secret_ref=f"env:{account_id.upper().replace('-', '_')}_API_KEY",
        priority=priority,
        daily_request_budget=10,
        daily_token_budget=1000,
        daily_cost_budget=0.0,
        terms_metadata={"consent": "recorded"},
        provenance_metadata={"source": "test"},
    )


def mock_pipeline(tmp_path: Path) -> TeacherPipeline:
    pipeline = TeacherPipeline(
        Database(tmp_path / "teacher.sqlite3"),
        TeacherPipelineConfig(mode=TeacherPipelineMode.MOCK),
    )
    pipeline.repository.upsert_assignment(assignment())
    return pipeline


def test_disabled_mode_preserves_legacy_and_never_dispatches(tmp_path):
    pipeline = TeacherPipeline(
        Database(tmp_path / "disabled.sqlite3"),
        TeacherPipelineConfig(),
    )
    result = pipeline.run_mock(
        source_page=tmp_path / "missing.png",
        rights=SourceRightsRecord(
            "", "", "", "", "", False, False, False, False, False, "", "", "", "", ""
        ),
        analysis=page_analysis(),
        mock_transcriptions=("unused", "unused"),
        work_dir=tmp_path / "work",
        idempotency_key="disabled",
    )
    assert result == {
        "status": "disabled",
        "reason_code": "teacher_pipeline_disabled",
        "legacy_path_unchanged": True,
    }
    assert pipeline.repository.status_snapshot()["teacher_outputs"] == 0


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("TEACHER_PIPELINE_MODE", "live", "live_teacher_mode_not_implemented"),
        (
            "TEACHER_LIVE_DISPATCH_ALLOWED",
            "true",
            "live_dispatch_locked_for_phase3",
        ),
        ("TRAINING_ENABLED", "true", "training_locked_for_phase3"),
        ("CLOUD_UPLOAD_ENABLED", "true", "cloud_upload_locked_for_phase3"),
        ("TEACHER_LIVE_DISPATCH_ALLOWED", "perhaps", "invalid_teacher"),
        ("TEACHER_CHARACTER_AGREEMENT_MIN", "2.0", "invalid_teacher"),
    ],
)
def test_unsafe_configuration_fails_closed(monkeypatch, name, value, reason):
    monkeypatch.setenv(name, value)
    with pytest.raises(TeacherPipelineConfigurationError, match=reason):
        TeacherPipelineConfig.from_env()


def test_page_analysis_and_coordinate_validation():
    page_analysis().validate(160, 100)
    broken = Region("bad", "main_text", (0, 0, 161, 20))
    with pytest.raises(ValueError, match="invalid_original_pixel_coordinates"):
        broken.validate(160, 100)
    unknown = PageAnalysis(
        **{
            **page_analysis().__dict__,
            "reading_order": ("missing",),
        }
    )
    with pytest.raises(ValueError, match="unknown_reading_order_region"):
        unknown.validate(160, 100)


def test_non_destructive_preparation_and_original_hash(tmp_path):
    source = image_page(tmp_path / "page.png")
    before = sha256_file(source)
    assets = prepare_regions(
        source,
        page_analysis().all_regions(),
        tmp_path / "derived",
        enhance_contrast=True,
    )
    assert sha256_file(source) == before
    assert assets[0].source_page_hash == before
    assert assets[0].parent_asset_hash == before
    assert assets[0].derived_asset_hash == sha256_file(assets[0].path)
    assert assets[0].original_coordinates == (10, 10, 150, 80)
    assert assets[0].path != source


def test_source_rights_fail_closed(tmp_path):
    page = image_page(tmp_path / "page.png")
    with pytest.raises(PermissionError, match="processing_not_authorized"):
        rights(page, allowed_for_processing=False).validate()
    with pytest.raises(PermissionError, match="source_rights_not_verified"):
        rights(page, rights_status="pending").validate()
    with pytest.raises(PermissionError, match="source_rights_incomplete"):
        rights(page, permission_source="").validate()


def test_contracts_are_versioned_and_grounded():
    assert len(contract_catalog()) == 10
    task = build_task(TeacherRole.PAGE_ANALYSIS, source_hash="a" * 64)
    assert task.contract_version == CONTRACT_VERSION
    assert task.prompt_version == PROMPT_VERSION
    for prohibited in (
        "Do not add",
        "paraphrase",
        "summarize",
        "modernize",
        "guess spelling",
        "drop unreadable",
        "fabricate",
        "complete sentences",
        "structured spans",
    ):
        assert prohibited in GROUNDING_RULES


def test_assignment_eligibility_owner_and_quota_isolation():
    selected = assignment()
    assert selected.eligible(
        TeacherRole.ARABIC_MAIN_OCR, "ARABIC_MAIN_OCR", "internal"
    ) == (True, "eligible")
    assert (
        assignment(owner_id="").eligible(
            TeacherRole.ARABIC_MAIN_OCR, "ARABIC_MAIN_OCR", "internal"
        )[1]
        == "owner_missing"
    )
    assert (
        assignment(quota_domain_id="").eligible(
            TeacherRole.ARABIC_MAIN_OCR, "ARABIC_MAIN_OCR", "internal"
        )[1]
        == "quota_domain_missing"
    )


def test_static_selection_does_not_cycle_accounts(tmp_path):
    pipeline = TeacherPipeline(
        Database(tmp_path / "select.sqlite3"),
        TeacherPipelineConfig(mode=TeacherPipelineMode.MOCK),
    )
    pipeline.repository.upsert_assignment(
        assignment(assignment_id="low", account_id="low", priority=1)
    )
    pipeline.repository.upsert_assignment(
        assignment(assignment_id="high", account_id="high", priority=100)
    )
    selected, rejected = pipeline.select_assignment(
        TeacherRole.ARABIC_MAIN_OCR,
        task_type="ARABIC_MAIN_OCR",
        classification="internal",
    )
    assert selected and selected.account_id == "high"
    assert not rejected
    assert (
        pipeline.select_assignment(
            TeacherRole.PAGE_ANALYSIS,
            task_type="PAGE_ANALYSIS",
            classification="internal",
        )[0]
        is None
    )


def test_teacher_endpoint_capability_requirements():
    endpoint = ModelEndpointCapability(
        endpoint_id="teacher",
        canonical_model_id="vision",
        provider="google_gemini",
        provider_model_id="mock",
        supports_vision_declared=True,
        supports_text_output=True,
        teacher_role_support=("PAGE_ANALYSIS",),
        region_support=("page",),
        privacy_compatibility=("internal",),
        training_output_policy_status="approved",
    )
    context = RequestContext(
        request_id="r",
        operation_id="o",
        task_type="PAGE_ANALYSIS",
        modality=Modality.IMAGE,
        requested_model="mock",
        requires_vision=True,
        privacy_classification=PrivacyClassification.INTERNAL,
        teacher_role="PAGE_ANALYSIS",
        region_type="page",
        required_training_output_policy_status="approved",
    )
    endpoint.validate_request(
        context, require_verified_vision=False, allow_declared_only=True
    )
    with pytest.raises(UnsupportedCapability, match="teacher_role_not_supported"):
        endpoint.validate_request(
            RequestContext(**{**context.__dict__, "teacher_role": "QUALITY_JUDGE"}),
            require_verified_vision=False,
            allow_declared_only=True,
        )


def test_scoring_and_dataset_policy():
    metrics = score_pair("ABC\nDEF", "ABC\nDEF")
    assert metrics.normalized_character_agreement == 1.0
    assert metrics.normalized_word_agreement == 1.0
    config = TeacherPipelineConfig()
    silver = decide_dataset(metrics, config)
    assert silver.decision is DatasetDecision.SILVER
    assert (
        decide_dataset(metrics, config, human_verified=True).decision
        is DatasetDecision.GOLD
    )
    assert (
        decide_dataset(metrics, config, trusted_reference=True).decision
        is DatasetDecision.GOLD
    )
    assert (
        decide_dataset(metrics, config, holdout=True).decision
        is DatasetDecision.HOLDOUT
    )
    review = decide_dataset(score_pair("ABC", "XYZ", margin_coverage=0.0), config)
    assert review.decision is DatasetDecision.REVIEW
    rejected = decide_dataset(metrics, config, rights_verified=False)
    assert rejected.decision is DatasetDecision.REJECTED


def test_mock_orchestration_persists_provenance_and_reservations(tmp_path):
    page = image_page(tmp_path / "page.png")
    pipeline = mock_pipeline(tmp_path)
    result = pipeline.run_mock(
        source_page=page,
        rights=rights(page),
        analysis=page_analysis(),
        mock_transcriptions=("ABC", "ABC"),
        work_dir=tmp_path / "work",
        idempotency_key="mock-page-1",
    )
    assert result["decision"] == "SILVER"
    assert result["candidate_type"] == "pseudo_ground_truth"
    assert result["network_requests"] == 0
    with pipeline.database.connect() as connection:
        output = connection.execute("""
            SELECT owner_id, quota_domain_id, reservation_id,
                   raw_output_retention_status, mock_or_live
            FROM teacher_outputs LIMIT 1
            """).fetchone()
        candidate = connection.execute(
            "SELECT * FROM candidate_ground_truth"
        ).fetchone()
        reservations = connection.execute(
            "SELECT state, fencing_token FROM request_reservations"
        ).fetchall()
        output_regions = connection.execute(
            "SELECT COUNT(*) FROM teacher_output_regions"
        ).fetchone()[0]
        manifests = connection.execute(
            "SELECT COUNT(*) FROM dataset_manifests"
        ).fetchone()[0]
    assert tuple(output) == (
        "owner-1",
        "quota-1",
        output["reservation_id"],
        "not_retained",
        "mock",
    )
    assert output["reservation_id"]
    assert candidate["dataset_decision"] == "SILVER"
    assert {row["state"] for row in reservations} == {"SETTLED_SUCCESS"}
    assert {row["fencing_token"] for row in reservations} == {2}
    assert output_regions == 2
    assert manifests == 1


def test_mock_orchestration_is_idempotent(tmp_path):
    page = image_page(tmp_path / "page.png")
    pipeline = mock_pipeline(tmp_path)
    kwargs = dict(
        source_page=page,
        rights=rights(page),
        analysis=page_analysis(),
        mock_transcriptions=("ABC", "ABC"),
        work_dir=tmp_path / "work",
        idempotency_key="same",
    )
    assert (
        pipeline.run_mock(**kwargs)["job_id"] == pipeline.run_mock(**kwargs)["job_id"]
    )
    with pipeline.database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM teacher_jobs").fetchone()[0] == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM request_reservations").fetchone()[
                0
            ]
            == 2
        )


def test_manifest_leakage_and_holdout_protection(tmp_path):
    first = image_page(tmp_path / "a.png", "A")
    second = image_page(tmp_path / "b.png", "A")
    first_hash = perceptual_dhash(first)
    second_hash = perceptual_dhash(second)
    assert hamming_distance(first_hash, second_hash) <= 5
    with pytest.raises(ValueError, match="near_duplicate_split_leakage"):
        validate_manifest_items(
            [
                {
                    "source_page_id": "p1",
                    "source_document_id": "d1",
                    "split": "train",
                    "perceptual_hash": first_hash,
                },
                {
                    "source_page_id": "p2",
                    "source_document_id": "d2",
                    "split": "holdout",
                    "perceptual_hash": second_hash,
                },
            ]
        )
    with pytest.raises(ValueError, match="document_split_leakage"):
        validate_manifest_items(
            [
                {
                    "source_page_id": "p1",
                    "source_document_id": "d",
                    "split": "train",
                    "perceptual_hash": "0" * 16,
                },
                {
                    "source_page_id": "p2",
                    "source_document_id": "d",
                    "split": "holdout",
                    "perceptual_hash": "f" * 16,
                },
            ]
        )
    target = tmp_path / "holdout" / "manifest.json"
    write_manifest([], target, frozen=True)
    with pytest.raises(PermissionError, match="frozen_holdout"):
        write_manifest([], target, frozen=False)
    paths = ensure_dataset_layout(tmp_path / "training_data")
    assert tuple(path.name for path in paths) == DATASET_LAYOUT


def test_secret_reference_only_and_sanitized_logging(tmp_path):
    repository = TeacherRepository(Database(tmp_path / "secret.sqlite3"))
    with pytest.raises(ValueError, match="Invalid secret reference"):
        repository.upsert_assignment(
            TeacherRoleAssignment(
                **{
                    **assignment().__dict__,
                    "secret_ref": "raw-secret-value",  # pragma: allowlist secret
                }
            )
        )
    repository.upsert_assignment(assignment())
    with repository.database.connect() as connection:
        stored = connection.execute(
            "SELECT secret_ref FROM teacher_role_assignments"
        ).fetchone()[0]
    assert stored == "env:ACCOUNT_1_API_KEY"
    sanitized = sanitize_metadata(
        {
            "authorization": "Bearer forbidden",
            "image_base64": "forbidden",
            "full_text": "private",
            "safe_hash": "a" * 64,
        }
    )
    assert sanitized["authorization"] == "[redacted]"
    assert sanitized["image_base64"] == "[redacted]"
    assert sanitized["full_text"] == "[redacted]"
    assert not _secret_findings(
        Path(".env.example"), "ACCOUNT_SECRET_REF=env:ACCOUNT_API_KEY"
    )
    assert _secret_findings(
        Path(".env.example"), "ACCOUNT_SECRET=actual-secret-value-123"
    )


def test_schema_v5_upgrade_preserves_existing_data(tmp_path):
    path = tmp_path / "upgrade.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE schema_meta(version INTEGER NOT NULL);
        INSERT INTO schema_meta(version) VALUES (5);
        CREATE TABLE conversions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL
        );
        INSERT INTO conversions(username, created_at, status)
        VALUES ('preserved', 'before', 'completed');
        """)
    connection.commit()
    connection.close()
    database = Database(path)
    with database.connect() as upgraded:
        assert (
            upgraded.execute("SELECT version FROM schema_meta").fetchone()[0]
            == SCHEMA_VERSION
        )
        assert (
            upgraded.execute("SELECT username FROM conversions").fetchone()[0]
            == "preserved"
        )
        tables = {
            row[0]
            for row in upgraded.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert SCHEMA_VERSION >= 6
    assert {
        "teacher_roles",
        "teacher_role_assignments",
        "teacher_task_contracts",
        "teacher_jobs",
        "teacher_outputs",
        "teacher_output_regions",
        "teacher_agreement_decisions",
        "candidate_ground_truth",
        "source_rights_records",
        "dataset_manifests",
        "dataset_manifest_items",
        "human_review_queue",
        "pipeline_activation_readiness",
    } <= tables


def test_internal_api_is_authenticated_sanitized_and_locked(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "phase3-worker")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setenv("TEACHER_PIPELINE_MODE", "disabled")
    client = TestClient(app)
    assert client.get("/internal/teacher-pipeline/status").status_code == 401
    headers = {"X-Worker-API-Key": "phase3-worker"}
    response = client.get("/internal/teacher-pipeline/status", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["live_dispatch_lock"]
    assert payload["training_lock"]
    assert payload["network_permitted"] is False
    assert payload["live_dispatch_status"] == "NOT_READY_FOR_LIVE_DISPATCH"
    for path in (
        "assignments",
        "contracts",
        "capabilities",
        "jobs",
        "outputs",
        "decisions",
        "review-queue",
        "rights-failures",
        "leakage-alerts",
        "dataset-manifests",
        "activation-readiness",
    ):
        result = client.get(f"/internal/teacher-pipeline/{path}", headers=headers)
        assert result.status_code == 200, (path, result.text)
        lowered = result.text.lower()
        assert "authorization" not in lowered
        assert "raw-secret-value" not in lowered


def test_human_review_can_create_gold_and_is_audited(tmp_path):
    page = image_page(tmp_path / "page.png")
    pipeline = mock_pipeline(tmp_path)
    result = pipeline.run_mock(
        source_page=page,
        rights=rights(page),
        analysis=page_analysis(),
        mock_transcriptions=("ABC", "XYZ"),
        work_dir=tmp_path / "work",
        idempotency_key="review",
    )
    assert result["decision"] == "REVIEW"
    changed = pipeline.record_human_review(
        job_id=result["job_id"],
        reviewer_id="reviewer-1",
        resolution=DatasetDecision.GOLD,
    )
    assert changed["decision"] == "GOLD"
    with pipeline.database.connect() as connection:
        decision = connection.execute(
            "SELECT decision, human_verified FROM teacher_agreement_decisions"
        ).fetchone()
        audits = connection.execute(
            "SELECT event_type FROM key_router_audit"
        ).fetchall()
    assert tuple(decision) == ("GOLD", 1)
    assert "human_review_recorded" in {row[0] for row in audits}


def test_status_never_reports_activation_ready(tmp_path):
    status = safe_status(Database(tmp_path / "status.sqlite3"))
    assert (
        status["release_gate"] == "READY_FOR_DORMANT_CONFIGURATION_AND_MOCK_VALIDATION"
    )
    assert status["live_dispatch_status"] == "NOT_READY_FOR_LIVE_DISPATCH"
    assert status["training_status"] == "NOT_READY_FOR_TRAINING"
    assert status["production_status"] == "NOT_READY_FOR_PRODUCTION"
    assert status["activation_blockers"]
    assert status["training_blockers"]


def test_examples_and_ui_contain_no_activation_controls():
    root = Path(__file__).resolve().parents[1]
    google = json.loads(
        (root / "configs/teacher_pipeline/google_gemini.example.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(google["assignments"]) == 10
    assert all(
        not item.get("enabled", False) for item in [google["assignment_defaults"]]
    )
    assert all(item["secret_ref"].startswith("env:") for item in google["assignments"])
    app_source = (root / "app.py").read_text(encoding="utf-8")
    assert "مسار المعلّمين الخامل" in app_source
    assert "start_teacher_training" not in app_source
    assert "activate_teacher_dispatch" not in app_source
