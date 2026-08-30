from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from typing import Any

from ..database import Database, utc_now
from ..key_router.audit import audit_event
from ..key_router.credential_refs import validate_secret_ref
from ..key_router.repository import KeyRouterRepository
from .contracts import contract_catalog
from .enums import DatasetDecision, DispatchMode, TeacherRole
from .models import (
    DecisionRecord,
    SourceRightsRecord,
    TeacherOutput,
    TeacherRoleAssignment,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class TeacherRepository:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.key_router_repository = KeyRouterRepository(database)

    def seed_catalog(self) -> None:
        now = utc_now()
        with self.database.transaction() as connection:
            for item in contract_catalog():
                role = item["teacher_role"]
                connection.execute(
                    """
                    INSERT INTO teacher_roles(
                        role_id, role_name, contract_version, prompt_version,
                        enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    ON CONFLICT(role_id) DO UPDATE SET
                        contract_version=excluded.contract_version,
                        prompt_version=excluded.prompt_version,
                        updated_at=excluded.updated_at
                    """,
                    (
                        role,
                        role,
                        item["task_contract_version"],
                        item["prompt_version"],
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO teacher_task_contracts(
                        contract_id, teacher_role, task_contract_version,
                        prompt_version, instruction_hash, enabled, created_at
                    ) VALUES (?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(contract_id) DO NOTHING
                    """,
                    (
                        f"{role}:{item['task_contract_version']}",
                        role,
                        item["task_contract_version"],
                        item["prompt_version"],
                        item["instruction_hash"],
                        now,
                    ),
                )

    def upsert_rights(self, record: SourceRightsRecord) -> None:
        record.validate()
        now = utc_now()
        values = asdict(record)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO source_rights_records(
                    rights_record_id, source_document_id, source_page_id,
                    source_hash, rights_status, allowed_for_processing,
                    allowed_for_training, allowed_for_commercial_training,
                    allowed_for_weight_release, allowed_for_data_redistribution,
                    permission_source, permission_version, permission_date,
                    data_classification, retention_policy, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rights_record_id) DO UPDATE SET
                    rights_status=excluded.rights_status,
                    allowed_for_processing=excluded.allowed_for_processing,
                    allowed_for_training=excluded.allowed_for_training,
                    allowed_for_commercial_training=
                        excluded.allowed_for_commercial_training,
                    allowed_for_weight_release=excluded.allowed_for_weight_release,
                    allowed_for_data_redistribution=
                        excluded.allowed_for_data_redistribution,
                    permission_version=excluded.permission_version,
                    updated_at=excluded.updated_at
                """,
                (
                    values["rights_record_id"],
                    values["source_document_id"],
                    values["source_page_id"],
                    values["source_hash"],
                    values["rights_status"],
                    int(values["allowed_for_processing"]),
                    int(values["allowed_for_training"]),
                    int(values["allowed_for_commercial_training"]),
                    int(values["allowed_for_weight_release"]),
                    int(values["allowed_for_data_redistribution"]),
                    values["permission_source"],
                    values["permission_version"],
                    values["permission_date"],
                    values["data_classification"],
                    values["retention_policy"],
                    now,
                    now,
                ),
            )
        self._audit("source_rights_recorded", request_id=record.source_page_id)

    def upsert_assignment(self, assignment: TeacherRoleAssignment) -> None:
        now = utc_now()
        secret_ref = validate_secret_ref(assignment.secret_ref)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO teacher_role_assignments(
                    assignment_id, account_id, owner_id, provider_id,
                    quota_domain_id, allowed_teacher_roles_json,
                    allowed_task_types_json, allowed_data_classifications_json,
                    enabled, dispatch_mode, secret_ref, priority,
                    daily_request_budget, daily_token_budget, daily_cost_budget,
                    cooldown_until, terms_metadata_json,
                    provenance_metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assignment_id) DO UPDATE SET
                    enabled=excluded.enabled,
                    dispatch_mode=excluded.dispatch_mode,
                    cooldown_until=excluded.cooldown_until,
                    updated_at=excluded.updated_at,
                    version=teacher_role_assignments.version + 1
                """,
                (
                    assignment.assignment_id,
                    assignment.account_id,
                    assignment.owner_id,
                    assignment.provider_id,
                    assignment.quota_domain_id,
                    _json([item.value for item in assignment.allowed_teacher_roles]),
                    _json(list(assignment.allowed_task_types)),
                    _json(list(assignment.allowed_data_classifications)),
                    int(assignment.enabled),
                    assignment.dispatch_mode.value,
                    secret_ref,
                    assignment.priority,
                    assignment.daily_request_budget,
                    assignment.daily_token_budget,
                    assignment.daily_cost_budget,
                    assignment.cooldown_until,
                    _json(assignment.terms_metadata),
                    _json(assignment.provenance_metadata),
                    now,
                    now,
                ),
            )
        self._audit(
            "teacher_assignment_upserted",
            provider=assignment.provider_id,
            account_id=assignment.account_id,
        )

    def list_assignments(self) -> list[TeacherRoleAssignment]:
        with self.database.connect() as connection:
            rows = connection.execute("""
                SELECT * FROM teacher_role_assignments
                ORDER BY priority DESC, provider_id, account_id, assignment_id
                """).fetchall()
        return [
            TeacherRoleAssignment(
                assignment_id=row["assignment_id"],
                account_id=row["account_id"],
                owner_id=row["owner_id"],
                provider_id=row["provider_id"],
                quota_domain_id=row["quota_domain_id"],
                allowed_teacher_roles=tuple(
                    TeacherRole(item)
                    for item in json.loads(row["allowed_teacher_roles_json"])
                ),
                allowed_task_types=tuple(json.loads(row["allowed_task_types_json"])),
                allowed_data_classifications=tuple(
                    json.loads(row["allowed_data_classifications_json"])
                ),
                enabled=bool(row["enabled"]),
                dispatch_mode=DispatchMode(row["dispatch_mode"]),
                secret_ref=row["secret_ref"],
                priority=int(row["priority"]),
                daily_request_budget=int(row["daily_request_budget"]),
                daily_token_budget=int(row["daily_token_budget"]),
                daily_cost_budget=float(row["daily_cost_budget"]),
                cooldown_until=row["cooldown_until"],
                terms_metadata=json.loads(row["terms_metadata_json"]),
                provenance_metadata=json.loads(row["provenance_metadata_json"]),
            )
            for row in rows
        ]

    def create_job(
        self,
        *,
        job_id: str,
        idempotency_key: str,
        rights: SourceRightsRecord,
        mode: str,
        input_manifest_hash: str,
        policy_version: str,
    ) -> None:
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO teacher_jobs(
                    job_id, idempotency_key, source_document_id, source_page_id,
                    source_hash, rights_record_id, status, pipeline_mode,
                    input_manifest_hash, policy_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    idempotency_key,
                    rights.source_document_id,
                    rights.source_page_id,
                    rights.source_hash,
                    rights.rights_record_id,
                    mode,
                    input_manifest_hash,
                    policy_version,
                    now,
                    now,
                ),
            )

    def record_output(self, output: TeacherOutput) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO teacher_outputs(
                    output_id, job_id, provider, model, endpoint,
                    provider_account_id, owner_id, quota_domain_id, teacher_role,
                    task_contract_version, prompt_version, request_id, attempt_id,
                    reservation_id, source_hash, region_hash, input_manifest_hash,
                    normalized_output_hash, raw_output_retention_status, latency_ms,
                    token_estimate, cost_estimate, mock_or_live, confidence,
                    normalized_output_json, decision_lineage_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    output.output_id,
                    output.job_id,
                    output.provider,
                    output.model,
                    output.endpoint,
                    output.provider_account_id,
                    output.owner_id,
                    output.quota_domain_id,
                    output.teacher_role.value,
                    output.task_contract_version,
                    output.prompt_version,
                    output.request_id,
                    output.attempt_id,
                    output.reservation_id,
                    output.source_hash,
                    output.region_hash,
                    output.input_manifest_hash,
                    output.normalized_output_hash,
                    output.raw_output_retention_status,
                    output.latency_ms,
                    output.token_estimate,
                    output.cost_estimate,
                    output.mock_or_live,
                    output.confidence,
                    _json(
                        {
                            "transcription": output.transcription,
                            "uncertainty_spans": [
                                asdict(item) for item in output.uncertainty_spans
                            ],
                        }
                    ),
                    _json(list(output.decision_lineage)),
                    output.created_at,
                ),
            )

    def record_output_regions(self, output_id: str, assets: tuple[Any, ...]) -> None:
        now = utc_now()
        with self.database.transaction() as connection:
            for asset in assets:
                identity = hashlib.sha256(
                    f"{output_id}:{asset.region_id}:{asset.derived_asset_hash}".encode()
                ).hexdigest()
                connection.execute(
                    """
                    INSERT OR IGNORE INTO teacher_output_regions(
                        output_region_id, output_id, region_id, region_type,
                        source_hash, region_hash, coordinates_json,
                        transform_chain_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity,
                        output_id,
                        asset.region_id,
                        asset.region_type,
                        asset.source_page_hash,
                        asset.derived_asset_hash,
                        _json(list(asset.original_coordinates)),
                        _json(list(asset.transform_chain)),
                        now,
                    ),
                )

    def record_decision(
        self,
        *,
        job_id: str,
        candidate_id: str,
        candidate_text: str,
        candidate_hash: str,
        decision: DecisionRecord,
        provenance_complete: bool,
        source_document_id: str,
        source_page_id: str,
        source_hash: str,
        perceptual_hash: str,
    ) -> None:
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO teacher_agreement_decisions(
                    decision_id, job_id, decision, policy_version, metrics_json,
                    reason_codes_json, human_verified, trusted_reference,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    decision=excluded.decision,
                    metrics_json=excluded.metrics_json,
                    reason_codes_json=excluded.reason_codes_json,
                    human_verified=excluded.human_verified,
                    trusted_reference=excluded.trusted_reference,
                    updated_at=excluded.updated_at
                """,
                (
                    uuid.uuid4().hex,
                    job_id,
                    decision.decision.value,
                    decision.policy_version,
                    _json(asdict(decision.metrics)),
                    _json(list(decision.reason_codes)),
                    int(decision.human_verified),
                    int(decision.trusted_reference),
                    now,
                    now,
                ),
            )
            manifest_id = f"{job_id}:manifest"
            dataset_version = f"teacher-mock-{job_id}"
            manifest_hash = hashlib.sha256(
                _json(
                    {
                        "candidate_id": candidate_id,
                        "decision": decision.decision.value,
                        "policy_version": decision.policy_version,
                        "source_hash": source_hash,
                    }
                ).encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                INSERT OR IGNORE INTO dataset_manifests(
                    manifest_id, dataset_version, policy_version, state,
                    manifest_hash, created_at, frozen_at
                ) VALUES (?, ?, ?, 'draft', ?, ?, '')
                """,
                (
                    manifest_id,
                    dataset_version,
                    decision.policy_version,
                    manifest_hash,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO dataset_manifest_items(
                    item_id, manifest_id, source_document_id, source_page_id,
                    source_hash, perceptual_hash, split, decision, candidate_id,
                    checksum, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{manifest_id}:{source_page_id}",
                    manifest_id,
                    source_document_id,
                    source_page_id,
                    source_hash,
                    perceptual_hash,
                    decision.decision.value.lower(),
                    decision.decision.value,
                    candidate_id,
                    candidate_hash,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO candidate_ground_truth(
                    candidate_id, job_id, candidate_type, candidate_text,
                    candidate_hash, dataset_decision, provenance_complete,
                    created_at, updated_at
                ) VALUES (?, ?, 'pseudo_ground_truth', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    candidate_text=excluded.candidate_text,
                    candidate_hash=excluded.candidate_hash,
                    dataset_decision=excluded.dataset_decision,
                    provenance_complete=excluded.provenance_complete,
                    updated_at=excluded.updated_at
                """,
                (
                    candidate_id,
                    job_id,
                    candidate_text,
                    candidate_hash,
                    decision.decision.value,
                    int(provenance_complete),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE teacher_jobs SET status=?, reason_code=?, updated_at=?
                WHERE job_id=?
                """,
                (
                    (
                        "COMPLETED"
                        if decision.decision
                        in {
                            DatasetDecision.GOLD,
                            DatasetDecision.SILVER,
                            DatasetDecision.HOLDOUT,
                        }
                        else decision.decision.value
                    ),
                    ",".join(decision.reason_codes),
                    now,
                    job_id,
                ),
            )
            if decision.decision is DatasetDecision.REVIEW:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO human_review_queue(
                        review_id, candidate_id, status, reason_codes_json,
                        created_at, updated_at
                    ) VALUES (?, ?, 'PENDING', ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        candidate_id,
                        _json(list(decision.reason_codes)),
                        now,
                        now,
                    ),
                )

    def status_snapshot(self) -> dict[str, Any]:
        with self.database.connect() as connection:

            def count(table: str, where: str = "") -> int:
                row = connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table} {where}"
                ).fetchone()
                return int(row["count"])

            return {
                "teacher_roles": count("teacher_roles"),
                "assignments": count("teacher_role_assignments"),
                "enabled_assignments": count(
                    "teacher_role_assignments", "WHERE enabled=1"
                ),
                "pending_jobs": count(
                    "teacher_jobs", "WHERE status IN ('PENDING','RUNNING')"
                ),
                "teacher_outputs": count("teacher_outputs"),
                "review_queue": count("human_review_queue", "WHERE status='PENDING'"),
                "dataset_manifests": count("dataset_manifests"),
                "rights_failures": count(
                    "teacher_jobs",
                    "WHERE reason_code LIKE '%rights%' OR reason_code LIKE '%processing%'",
                ),
            }

    def sanitized_output_summaries(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT output_id, job_id, provider, model, endpoint,
                       provider_account_id, owner_id, quota_domain_id,
                       teacher_role, task_contract_version, prompt_version,
                       normalized_output_hash, raw_output_retention_status,
                       latency_ms, token_estimate, cost_estimate, mock_or_live,
                       confidence, created_at
                FROM teacher_outputs ORDER BY created_at DESC LIMIT ?
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def _audit(
        self,
        event_type: str,
        *,
        provider: str = "",
        account_id: str = "",
        request_id: str = "",
    ) -> None:
        self.key_router_repository.record_audit(
            audit_event(
                event_type=event_type,
                provider=provider,
                account_id=account_id,
                request_id=request_id,
            )
        )
