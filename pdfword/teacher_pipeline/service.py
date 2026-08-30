from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from ..database import Database, utc_now
from ..key_router.audit import sanitize_metadata
from ..key_router.enums import ReservationState
from .config import TeacherPipelineConfig, TeacherPipelineConfigurationError
from .contracts import build_task
from .enums import DatasetDecision, DispatchMode, TeacherPipelineMode, TeacherRole
from .models import (
    PageAnalysis,
    SourceRightsRecord,
    TeacherOutput,
    TeacherRoleAssignment,
)
from .manifests import perceptual_dhash
from .preparation import prepare_regions, sha256_file
from .repository import TeacherRepository
from .scoring import decide_dataset, normalize_for_agreement, score_pair

LIVE_BLOCKERS = (
    "authorized_credentials_missing",
    "credential_ownership_and_consent_missing",
    "provider_terms_not_reviewed",
    "capabilities_not_verified_live",
    "mime_and_payload_limits_not_verified_live",
    "privacy_and_retention_not_approved",
    "per_account_budgets_not_configured",
    "retry_and_circuit_settings_not_calibrated",
    "authorized_staging_smoke_not_passed",
    "audit_not_reviewed",
    "explicit_activation_approval_missing",
)
TRAINING_BLOCKERS = (
    "authorized_source_registry_incomplete",
    "quality_thresholds_not_validated",
    "human_review_workflow_not_operational",
    "holdout_not_frozen",
    "leakage_checks_not_approved",
    "dataset_manifest_not_versioned",
    "training_configuration_not_approved",
)


class TeacherPipeline:
    def __init__(
        self,
        database: Database | None = None,
        config: TeacherPipelineConfig | None = None,
    ) -> None:
        self.database = database or Database()
        self.config = config or TeacherPipelineConfig.from_env()
        self.repository = TeacherRepository(self.database)

    def status(self) -> dict[str, Any]:
        snapshot = self.repository.status_snapshot()
        payload = {
            "pipeline_mode": self.config.mode.value,
            "live_dispatch_lock": True,
            "training_lock": True,
            "cloud_upload_lock": True,
            "network_permitted": False,
            "mock_only": self.config.mode is TeacherPipelineMode.MOCK,
            "release_gate": "READY_FOR_DORMANT_CONFIGURATION_AND_MOCK_VALIDATION",
            "live_dispatch_status": "NOT_READY_FOR_LIVE_DISPATCH",
            "training_status": "NOT_READY_FOR_TRAINING",
            "production_status": "NOT_READY_FOR_PRODUCTION",
            "live_provider_testing_status": "NOT_READY_FOR_LIVE_PROVIDER_TESTING",
            "activation_blockers": list(LIVE_BLOCKERS),
            "training_blockers": list(TRAINING_BLOCKERS),
            **snapshot,
        }
        result = sanitize_metadata(payload)
        return result if isinstance(result, dict) else {}

    def select_assignment(
        self,
        role: TeacherRole,
        *,
        task_type: str,
        classification: str,
    ) -> tuple[TeacherRoleAssignment | None, tuple[dict[str, str], ...]]:
        rejected: list[dict[str, str]] = []
        eligible: list[TeacherRoleAssignment] = []
        for assignment in self.repository.list_assignments():
            allowed, reason = assignment.eligible(role, task_type, classification)
            if not allowed:
                rejected.append(
                    {"assignment_id": assignment.assignment_id, "reason": reason}
                )
            else:
                eligible.append(assignment)
        eligible.sort(
            key=lambda item: (
                -item.priority,
                item.provider_id,
                item.account_id,
                item.assignment_id,
            )
        )
        return (eligible[0] if eligible else None), tuple(rejected)

    def run_mock(
        self,
        *,
        source_page: str | Path,
        rights: SourceRightsRecord,
        analysis: PageAnalysis,
        mock_transcriptions: tuple[str, str],
        work_dir: str | Path,
        idempotency_key: str,
        human_verified: bool = False,
        trusted_reference: bool = False,
        holdout: bool = False,
    ) -> dict[str, Any]:
        if self.config.mode is TeacherPipelineMode.DISABLED:
            return {
                "status": "disabled",
                "reason_code": "teacher_pipeline_disabled",
                "legacy_path_unchanged": True,
            }
        if self.config.mode is not TeacherPipelineMode.MOCK:
            raise TeacherPipelineConfigurationError("mock_mode_required")
        rights.validate()
        source_path = Path(source_page).resolve()
        source_hash = sha256_file(source_path)
        if source_hash != rights.source_hash:
            raise PermissionError("source_hash_mismatch")
        with Image.open(source_path) as image:
            analysis.validate(image.width, image.height)
        if analysis.page_id != rights.source_page_id:
            raise PermissionError("rights_page_mismatch")
        self.repository.seed_catalog()
        self.repository.upsert_rights(rights)
        task = build_task(
            TeacherRole.ARABIC_MAIN_OCR,
            source_hash=source_hash,
            metadata={"page_id": analysis.page_id},
        )
        assignment, rejected = self.select_assignment(
            TeacherRole.ARABIC_MAIN_OCR,
            task_type=task.task_type,
            classification=rights.data_classification,
        )
        if assignment is None:
            return {
                "status": "blocked",
                "reason_code": "no_eligible_teacher_assignment",
                "rejected": list(rejected),
            }
        if assignment.dispatch_mode is not DispatchMode.MOCK:
            raise TeacherPipelineConfigurationError("mock_assignment_required")
        assets = prepare_regions(
            source_path, analysis.all_regions(), work_dir, enhance_contrast=True
        )
        job_id = uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key).hex
        self.repository.create_job(
            job_id=job_id,
            idempotency_key=idempotency_key,
            rights=rights,
            mode=self.config.mode.value,
            input_manifest_hash=task.input_manifest_hash,
            policy_version=self.config.acceptance_policy_version,
        )
        outputs: list[TeacherOutput] = []
        for index, transcription in enumerate(mock_transcriptions, 1):
            normalized = normalize_for_agreement(transcription)
            output_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            attempt_id = f"{job_id}:attempt:{index}"
            reservation = self.repository.key_router_repository.reserve_quota(
                idempotency_key=f"{idempotency_key}:teacher:{index}",
                provider=assignment.provider_id,
                pool_id=f"teacher:{assignment.quota_domain_id}",
                account_id=assignment.account_id,
                model="deterministic-mock-v1",
                request_limit=assignment.daily_request_budget,
                token_limit=assignment.daily_token_budget,
                cost_limit=assignment.daily_cost_budget,
                estimated_input_tokens=0,
                estimated_output_tokens=len(transcription.split()),
                estimated_cost=0.0,
                ttl_seconds=300,
                request_id=f"{job_id}:request:{index}",
                parent_request_id=job_id,
                attempt_id=attempt_id,
                provider_model_id="deterministic-mock-v1",
                policy_id=task.task_type,
            )
            if reservation.state is ReservationState.RESERVED:
                dispatching = self.repository.key_router_repository.mark_dispatching(
                    reservation.reservation_id,
                    reservation.reservation_token,
                )
                sent = self.repository.key_router_repository.mark_sent(
                    dispatching.reservation_id,
                    dispatching.reservation_token,
                )
                reservation = self.repository.key_router_repository.settle_reservation(
                    reservation_id=sent.reservation_id,
                    reservation_token=sent.reservation_token,
                    state=ReservationState.SETTLED_SUCCESS,
                    actual_input_tokens=0,
                    actual_output_tokens=len(transcription.split()),
                    actual_cost=0.0,
                )
            output = TeacherOutput(
                output_id=f"{job_id}:mock:{index}",
                job_id=job_id,
                provider=assignment.provider_id,
                model="deterministic-mock-v1",
                endpoint="local://teacher-mock",
                provider_account_id=assignment.account_id,
                owner_id=assignment.owner_id,
                quota_domain_id=assignment.quota_domain_id,
                teacher_role=TeacherRole.ARABIC_MAIN_OCR,
                task_contract_version=task.contract_version,
                prompt_version=task.prompt_version,
                request_id=f"{job_id}:request:{index}",
                attempt_id=attempt_id,
                reservation_id=reservation.reservation_id,
                source_hash=source_hash,
                region_hash=assets[0].derived_asset_hash if assets else "",
                input_manifest_hash=task.input_manifest_hash,
                normalized_output_hash=output_hash,
                raw_output_retention_status="not_retained",
                latency_ms=0,
                token_estimate=len(transcription.split()),
                cost_estimate=0.0,
                mock_or_live="mock",
                created_at=utc_now(),
                decision_lineage=(task.contract_version, task.prompt_version),
                transcription=transcription,
                confidence=1.0,
            )
            self.repository.record_output(output)
            self.repository.record_output_regions(output.output_id, assets)
            outputs.append(output)
        metrics = score_pair(outputs[0].transcription, outputs[1].transcription)
        decision = decide_dataset(
            metrics,
            self.config,
            human_verified=human_verified,
            trusted_reference=trusted_reference,
            holdout=holdout,
        )
        candidate_text = outputs[0].transcription
        candidate_hash = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
        candidate_id = f"{job_id}:candidate"
        self.repository.record_decision(
            job_id=job_id,
            candidate_id=candidate_id,
            candidate_text=candidate_text,
            candidate_hash=candidate_hash,
            decision=decision,
            provenance_complete=True,
            source_document_id=rights.source_document_id,
            source_page_id=rights.source_page_id,
            source_hash=source_hash,
            perceptual_hash=perceptual_dhash(source_path),
        )
        return {
            "status": "completed",
            "job_id": job_id,
            "candidate_id": candidate_id,
            "candidate_type": "pseudo_ground_truth",
            "decision": decision.decision.value,
            "reason_codes": list(decision.reason_codes),
            "source_hash": source_hash,
            "assets": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(asset).items()
                }
                for asset in assets
            ],
            "network_requests": 0,
        }

    def record_human_review(
        self,
        *,
        job_id: str,
        reviewer_id: str,
        resolution: DatasetDecision,
    ) -> dict[str, str]:
        if resolution not in {
            DatasetDecision.GOLD,
            DatasetDecision.REVIEW,
            DatasetDecision.REJECTED,
        }:
            raise ValueError("invalid_human_review_resolution")
        if not reviewer_id.strip():
            raise ValueError("reviewer_id_required")
        now = utc_now()
        with self.database.transaction() as connection:
            candidate = connection.execute(
                "SELECT candidate_id FROM candidate_ground_truth WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if candidate is None:
                raise KeyError("candidate_not_found")
            connection.execute(
                """
                UPDATE candidate_ground_truth
                SET dataset_decision=?, updated_at=? WHERE job_id=?
                """,
                (resolution.value, now, job_id),
            )
            connection.execute(
                """
                UPDATE dataset_manifest_items
                SET decision=?, split=? WHERE candidate_id=?
                """,
                (
                    resolution.value,
                    resolution.value.lower(),
                    candidate["candidate_id"],
                ),
            )
            connection.execute(
                """
                UPDATE human_review_queue
                SET status='RESOLVED', assigned_reviewer=?, resolution=?,
                    updated_at=? WHERE candidate_id=?
                """,
                (
                    reviewer_id,
                    resolution.value,
                    now,
                    candidate["candidate_id"],
                ),
            )
            if resolution is DatasetDecision.GOLD:
                connection.execute(
                    """
                    UPDATE teacher_agreement_decisions
                    SET decision='GOLD', human_verified=1, updated_at=?
                    WHERE job_id=?
                    """,
                    (now, job_id),
                )
        self.repository._audit("human_review_recorded", request_id=job_id)
        return {"job_id": job_id, "decision": resolution.value}


def safe_status(database: Database | None = None) -> dict[str, Any]:
    try:
        return TeacherPipeline(database).status()
    except TeacherPipelineConfigurationError as exc:
        return {
            "pipeline_mode": "invalid",
            "live_dispatch_lock": True,
            "training_lock": True,
            "cloud_upload_lock": True,
            "network_permitted": False,
            "release_gate": "NOT_READY_FOR_DORMANT_CONFIGURATION_AND_MOCK_VALIDATION",
            "live_dispatch_status": "NOT_READY_FOR_LIVE_DISPATCH",
            "training_status": "NOT_READY_FOR_TRAINING",
            "configuration_error": str(exc),
            "activation_blockers": list(LIVE_BLOCKERS),
            "training_blockers": list(TRAINING_BLOCKERS),
        }
