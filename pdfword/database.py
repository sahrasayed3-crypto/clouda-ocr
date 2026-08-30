import hashlib
import os
import secrets
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from clouda_contracts.security import sanitize_spreadsheet_cell
from typing import Any, Iterator

from clouda_contracts.storage import StorageRoots

DEFAULT_DB_PATH = StorageRoots.from_env().database_path
SCHEMA_VERSION = 8
LEGACY_EMPTY_CLAIM_TOKEN = ""
FINAL_STATUSES = {"completed", "failed", "manual_review", "cancelled"}
ALLOWED_STATUS_TRANSITIONS = {
    "pending": {"processing", "failed", "cancelled"},
    "processing": {
        "pending",
        "finalizing",
        "completed",
        "failed",
        "manual_review",
        "cancelled",
    },
    "finalizing": {"processing", "completed", "manual_review", "failed"},
    "failed": {"pending"},
    "cancelled": {"pending"},
    "manual_review": set(),
    "completed": set(),
}
CONVERSION_INSERT_FIELDS = {
    "job_id",
    "username",
    "original_pdf_name",
    "stored_pdf_path",
    "output_docx_name",
    "stored_docx_path",
    "page_from",
    "page_to",
    "page_numbers",
    "file_type",
    "text_quality_score",
    "layout_quality_score",
    "final_quality_score",
    "winning_engine",
    "winning_model",
    "total_cost",
    "processing_time",
    "status",
    "hidden",
    "created_at",
    "updated_at",
    "error_message",
    "corrected_docx_path",
    "actual_char_accuracy",
    "actual_word_accuracy",
    "attempt_count",
    "started_at",
    "completed_at",
    "worker_name",
    "last_heartbeat",
    "rq_job_id",
    "claim_token",
    "intended_final_status",
    "owner_user_id",
    "guest_scope_id",
    "visibility",
    "lifecycle_state",
    "expires_at",
}
ATTEMPT_INSERT_FIELDS = {
    "conversion_id",
    "engine_name",
    "model_name",
    "engine_type",
    "attempt_number",
    "quality_score",
    "cost",
    "cost_is_estimated",
    "prompt_tokens",
    "completion_tokens",
    "processing_time",
    "success",
    "failure_reason",
    "created_at",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = (
            path
            if path is not None
            else os.getenv("CLOUDA_DATABASE_PATH") or os.getenv("DATABASE_PATH")
        )
        self.path = Path(configured_path or DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def active_conversion_scope_ids(self) -> set[str]:
        """Return scope IDs for conversions with status pending or processing."""
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT owner_user_id, guest_scope_id FROM conversions WHERE status IN ('pending', 'processing')"
            ).fetchall()
        result = set()
        for row in rows:
            if row["owner_user_id"]:
                result.add(row["owner_user_id"])
            if row["guest_scope_id"]:
                result.add(row["guest_scope_id"])
        return result

    def initialize(self) -> None:
        with self.transaction() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    created_at TEXT NOT NULL,
                    last_login TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    username TEXT NOT NULL,
                    original_pdf_name TEXT NOT NULL,
                    stored_pdf_path TEXT NOT NULL,
                    output_docx_name TEXT,
                    stored_docx_path TEXT,
                    page_from INTEGER,
                    page_to INTEGER,
                    file_type TEXT,
                    text_quality_score REAL,
                    layout_quality_score REAL,
                    final_quality_score REAL,
                    winning_engine TEXT,
                    winning_model TEXT,
                    total_cost REAL NOT NULL DEFAULT 0,
                    processing_time REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    hidden INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT,
                    intended_final_status TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_conversions_user_created
                    ON conversions(username, created_at DESC);
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversion_id INTEGER NOT NULL REFERENCES conversions(id) ON DELETE CASCADE,
                    engine_name TEXT,
                    model_name TEXT,
                    engine_type TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    quality_score REAL,
                    cost REAL NOT NULL DEFAULT 0,
                    cost_is_estimated INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    processing_time REAL NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 1,
                    failure_reason TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    type TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS correction_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    replacement TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    approved INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(pattern, replacement, rule_type)
                );
                CREATE TABLE IF NOT EXISTS correction_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    stored_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    uploaded_by TEXT NOT NULL,
                    comparison_status TEXT NOT NULL DEFAULT 'pending',
                    learning_status TEXT NOT NULL DEFAULT 'review_only',
                    change_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(job_id, version),
                    UNIQUE(job_id, file_hash)
                );
                CREATE TABLE IF NOT EXISTS correction_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision_id INTEGER NOT NULL REFERENCES correction_revisions(id) ON DELETE CASCADE,
                    job_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    wrong_text TEXT NOT NULL,
                    correct_text TEXT NOT NULL,
                    context_before TEXT NOT NULL DEFAULT '',
                    context_after TEXT NOT NULL DEFAULT '',
                    paragraph_index INTEGER,
                    sentence_index INTEGER,
                    change_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    is_sensitive INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'proposed'
                        CHECK(status IN ('proposed','approved','rejected','ignored')),
                    approved_by TEXT,
                    approved_at TEXT,
                    rejected_at TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS correction_memory_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wrong_text TEXT NOT NULL,
                    correct_text TEXT NOT NULL,
                    context_pattern TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL DEFAULT 'global',
                    scope_value TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT 'unknown',
                    document_type TEXT NOT NULL DEFAULT '',
                    source_count INTEGER NOT NULL DEFAULT 0,
                    occurrence_count INTEGER NOT NULL DEFAULT 0,
                    accepted_count INTEGER NOT NULL DEFAULT 0,
                    rejected_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0,
                    is_sensitive INTEGER NOT NULL DEFAULT 0,
                    approved INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    approved_by TEXT,
                    approved_at TEXT,
                    last_applied_at TEXT,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(wrong_text, correct_text, context_pattern, scope, scope_value)
                );
                CREATE TABLE IF NOT EXISTS correction_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    rule_id INTEGER NOT NULL,
                    before_text TEXT NOT NULL,
                    after_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    context_match INTEGER NOT NULL,
                    applied_at TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'unknown'
                );
                CREATE TABLE IF NOT EXISTS correction_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision_id INTEGER NOT NULL,
                    job_id TEXT NOT NULL,
                    dataset_split TEXT NOT NULL CHECK(dataset_split IN ('training','evaluation')),
                    pre_word_accuracy REAL,
                    pre_char_accuracy REAL,
                    post_word_accuracy REAL,
                    post_char_accuracy REAL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backup_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backup_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_accounts (
                    account_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    secret_ref TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    environment TEXT NOT NULL DEFAULT 'local',
                    region TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    tenant_id TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 0,
                    weight INTEGER NOT NULL DEFAULT 1,
                    allows_free INTEGER NOT NULL DEFAULT 1,
                    allows_paid INTEGER NOT NULL DEFAULT 0,
                    allows_vision INTEGER NOT NULL DEFAULT 0,
                    allows_text INTEGER NOT NULL DEFAULT 1,
                    monthly_cost_limit REAL NOT NULL DEFAULT 0,
                    daily_cost_limit REAL NOT NULL DEFAULT 0,
                    daily_request_limit INTEGER NOT NULL DEFAULT 0,
                    daily_token_limit INTEGER NOT NULL DEFAULT 0,
                    max_concurrency INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_provider_accounts_provider_pool
                    ON provider_accounts(provider, pool_id, enabled);
                CREATE TABLE IF NOT EXISTS account_runtime (
                    router_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    active_account_id TEXT NOT NULL DEFAULT '',
                    pending_account_id TEXT NOT NULL DEFAULT '',
                    account_epoch INTEGER NOT NULL DEFAULT 0,
                    switch_id TEXT NOT NULL DEFAULT '',
                    drain_deadline TEXT NOT NULL DEFAULT '',
                    closed_at TEXT NOT NULL DEFAULT '',
                    switch_started_at TEXT NOT NULL DEFAULT '',
                    cooldown_until TEXT NOT NULL DEFAULT '',
                    last_error_code TEXT NOT NULL DEFAULT '',
                    last_error_at TEXT NOT NULL DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(router_id, provider, pool_id)
                );
                CREATE TABLE IF NOT EXISTS quota_runtime (
                    provider TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    window_type TEXT NOT NULL,
                    window_started_at TEXT NOT NULL,
                    request_limit INTEGER NOT NULL DEFAULT 0,
                    token_limit INTEGER NOT NULL DEFAULT 0,
                    cost_limit REAL NOT NULL DEFAULT 0,
                    reserved_requests INTEGER NOT NULL DEFAULT 0,
                    reserved_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_output_tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_cost REAL NOT NULL DEFAULT 0,
                    consumed_requests INTEGER NOT NULL DEFAULT 0,
                    consumed_input_tokens INTEGER NOT NULL DEFAULT 0,
                    consumed_output_tokens INTEGER NOT NULL DEFAULT 0,
                    consumed_cost REAL NOT NULL DEFAULT 0,
                    cooldown_until TEXT NOT NULL DEFAULT '',
                    circuit_state TEXT NOT NULL DEFAULT 'CLOSED',
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_success_at TEXT NOT NULL DEFAULT '',
                    last_failure_at TEXT NOT NULL DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(
                        provider, pool_id, account_id, model,
                        window_type, window_started_at
                    )
                );
                CREATE TABLE IF NOT EXISTS request_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    reserved_requests INTEGER NOT NULL DEFAULT 1,
                    estimated_input_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_output_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost REAL NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    reservation_token TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    settled_at TEXT NOT NULL DEFAULT '',
                    actual_input_tokens INTEGER NOT NULL DEFAULT 0,
                    actual_output_tokens INTEGER NOT NULL DEFAULT 0,
                    actual_cost REAL NOT NULL DEFAULT 0,
                    error_code TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_request_reservations_runtime
                    ON request_reservations(provider, pool_id, account_id, model, state);
                CREATE INDEX IF NOT EXISTS idx_request_reservations_expiry
                    ON request_reservations(state, expires_at);
                CREATE TABLE IF NOT EXISTS key_router_audit (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    pool_id TEXT NOT NULL DEFAULT '',
                    account_id TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    decision_reason TEXT NOT NULL DEFAULT '',
                    old_state TEXT NOT NULL DEFAULT '',
                    new_state TEXT NOT NULL DEFAULT '',
                    request_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_key_router_audit_created
                    ON key_router_audit(created_at DESC);
                CREATE TABLE IF NOT EXISTS capability_endpoints (
                    endpoint_id TEXT PRIMARY KEY,
                    canonical_model_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    provider_model_id TEXT NOT NULL,
                    endpoint_type TEXT NOT NULL DEFAULT 'chat',
                    supports_text INTEGER NOT NULL DEFAULT 1,
                    supports_vision_declared INTEGER NOT NULL DEFAULT 0,
                    supports_vision_verified INTEGER NOT NULL DEFAULT 0,
                    supports_base64 INTEGER NOT NULL DEFAULT 0,
                    supports_image_url INTEGER NOT NULL DEFAULT 0,
                    supports_native_pdf INTEGER NOT NULL DEFAULT 0,
                    supports_multiple_images INTEGER NOT NULL DEFAULT 0,
                    max_images_per_request INTEGER NOT NULL DEFAULT 1,
                    supported_mime_types_json TEXT NOT NULL DEFAULT '[]',
                    max_image_width INTEGER NOT NULL DEFAULT 0,
                    max_image_height INTEGER NOT NULL DEFAULT 0,
                    max_image_pixels INTEGER NOT NULL DEFAULT 0,
                    max_payload_bytes INTEGER NOT NULL DEFAULT 0,
                    max_context_tokens INTEGER NOT NULL DEFAULT 0,
                    max_output_tokens INTEGER NOT NULL DEFAULT 0,
                    supports_structured_json INTEGER NOT NULL DEFAULT 0,
                    supports_json_schema INTEGER NOT NULL DEFAULT 0,
                    supports_streaming INTEGER NOT NULL DEFAULT 0,
                    supports_system_prompt INTEGER NOT NULL DEFAULT 1,
                    supports_temperature INTEGER NOT NULL DEFAULT 1,
                    supports_seed INTEGER NOT NULL DEFAULT 0,
                    supports_tools INTEGER NOT NULL DEFAULT 0,
                    supports_parallel_tools INTEGER NOT NULL DEFAULT 0,
                    supports_reasoning INTEGER NOT NULL DEFAULT 0,
                    pricing_input REAL NOT NULL DEFAULT 0,
                    pricing_output REAL NOT NULL DEFAULT 0,
                    pricing_image REAL NOT NULL DEFAULT 0,
                    currency TEXT NOT NULL DEFAULT 'USD',
                    region_availability_json TEXT NOT NULL DEFAULT '[]',
                    declared_status TEXT NOT NULL DEFAULT 'declared',
                    verified_status TEXT NOT NULL DEFAULT 'unknown',
                    declared_at TEXT NOT NULL DEFAULT '',
                    last_verified_at TEXT NOT NULL DEFAULT '',
                    capability_source TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 0,
                    expected_latency_ms INTEGER NOT NULL DEFAULT 0,
                    quality_score REAL NOT NULL DEFAULT 0,
                    supports_text_output INTEGER NOT NULL DEFAULT 1,
                    privacy_compatibility_json TEXT NOT NULL DEFAULT '[]',
                    region_support_json TEXT NOT NULL DEFAULT '[]',
                    teacher_role_support_json TEXT NOT NULL DEFAULT '[]',
                    training_output_policy_status TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(provider, provider_model_id, endpoint_type)
                );
                CREATE INDEX IF NOT EXISTS idx_capability_endpoint_model
                    ON capability_endpoints(canonical_model_id, provider, enabled);
                CREATE TABLE IF NOT EXISTS key_router_circuits (
                    circuit_id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    provider_model_id TEXT NOT NULL DEFAULT '',
                    account_id TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'CLOSED',
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    window_started_at TEXT NOT NULL DEFAULT '',
                    opened_at TEXT NOT NULL DEFAULT '',
                    cooldown_until TEXT NOT NULL DEFAULT '',
                    half_open_in_flight INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(level, provider, provider_model_id, account_id)
                );
                CREATE INDEX IF NOT EXISTS idx_key_router_circuits_state
                    ON key_router_circuits(state, provider);
                CREATE TABLE IF NOT EXISTS key_router_errors (
                    error_id TEXT PRIMARY KEY,
                    error_type TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    provider_account_id TEXT NOT NULL DEFAULT '',
                    provider_model_id TEXT NOT NULL DEFAULT '',
                    http_status INTEGER,
                    provider_error_code TEXT NOT NULL DEFAULT '',
                    sanitized_message TEXT NOT NULL DEFAULT '',
                    retryable INTEGER NOT NULL DEFAULT 0,
                    safe_to_failover INTEGER NOT NULL DEFAULT 0,
                    retry_after_seconds INTEGER,
                    scope TEXT NOT NULL DEFAULT 'unknown',
                    request_may_have_been_processed INTEGER NOT NULL DEFAULT 0,
                    raw_error_fingerprint TEXT NOT NULL DEFAULT '',
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_key_router_errors_type
                    ON key_router_errors(error_type, provider, occurred_at DESC);
                CREATE TABLE IF NOT EXISTS key_router_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    parent_request_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    provider_model_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    reservation_id TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT 'pending',
                    error_type TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_key_router_attempts_parent
                    ON key_router_attempts(parent_request_id, attempt_number);
                CREATE TABLE IF NOT EXISTS teacher_roles (
                    role_id TEXT PRIMARY KEY,
                    role_name TEXT NOT NULL UNIQUE,
                    contract_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS teacher_role_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    quota_domain_id TEXT NOT NULL,
                    allowed_teacher_roles_json TEXT NOT NULL,
                    allowed_task_types_json TEXT NOT NULL,
                    allowed_data_classifications_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    dispatch_mode TEXT NOT NULL DEFAULT 'disabled',
                    secret_ref TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 0,
                    daily_request_budget INTEGER NOT NULL DEFAULT 0,
                    daily_token_budget INTEGER NOT NULL DEFAULT 0,
                    daily_cost_budget REAL NOT NULL DEFAULT 0,
                    cooldown_until TEXT NOT NULL DEFAULT '',
                    terms_metadata_json TEXT NOT NULL DEFAULT '{}',
                    provenance_metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_teacher_assignments_selection
                    ON teacher_role_assignments(
                        provider_id, enabled, dispatch_mode, priority
                    );
                CREATE TABLE IF NOT EXISTS teacher_task_contracts (
                    contract_id TEXT PRIMARY KEY,
                    teacher_role TEXT NOT NULL,
                    task_contract_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    instruction_hash TEXT NOT NULL,
                    schema_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(teacher_role, task_contract_version, prompt_version)
                );
                CREATE TABLE IF NOT EXISTS source_rights_records (
                    rights_record_id TEXT PRIMARY KEY,
                    source_document_id TEXT NOT NULL,
                    source_page_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    rights_status TEXT NOT NULL,
                    allowed_for_processing INTEGER NOT NULL DEFAULT 0,
                    allowed_for_training INTEGER NOT NULL DEFAULT 0,
                    allowed_for_commercial_training INTEGER NOT NULL DEFAULT 0,
                    allowed_for_weight_release INTEGER NOT NULL DEFAULT 0,
                    allowed_for_data_redistribution INTEGER NOT NULL DEFAULT 0,
                    permission_source TEXT NOT NULL,
                    permission_version TEXT NOT NULL,
                    permission_date TEXT NOT NULL,
                    data_classification TEXT NOT NULL,
                    retention_policy TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_page_id, source_hash)
                );
                CREATE TABLE IF NOT EXISTS teacher_jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    source_document_id TEXT NOT NULL,
                    source_page_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    rights_record_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pipeline_mode TEXT NOT NULL,
                    input_manifest_hash TEXT NOT NULL DEFAULT '',
                    policy_version TEXT NOT NULL,
                    reason_code TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_teacher_jobs_status
                    ON teacher_jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS teacher_outputs (
                    output_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    provider_account_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    quota_domain_id TEXT NOT NULL,
                    teacher_role TEXT NOT NULL,
                    task_contract_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    reservation_id TEXT NOT NULL DEFAULT '',
                    source_hash TEXT NOT NULL,
                    region_hash TEXT NOT NULL DEFAULT '',
                    input_manifest_hash TEXT NOT NULL,
                    normalized_output_hash TEXT NOT NULL,
                    raw_output_retention_status TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    token_estimate INTEGER NOT NULL DEFAULT 0,
                    cost_estimate REAL NOT NULL DEFAULT 0,
                    mock_or_live TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    normalized_output_json TEXT NOT NULL,
                    decision_lineage_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_teacher_outputs_job
                    ON teacher_outputs(job_id, teacher_role);
                CREATE TABLE IF NOT EXISTS teacher_output_regions (
                    output_region_id TEXT PRIMARY KEY,
                    output_id TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    region_type TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    region_hash TEXT NOT NULL,
                    coordinates_json TEXT NOT NULL,
                    transform_chain_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS teacher_agreement_decisions (
                    decision_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE,
                    decision TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    human_verified INTEGER NOT NULL DEFAULT 0,
                    trusted_reference INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_ground_truth (
                    candidate_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE,
                    candidate_type TEXT NOT NULL,
                    candidate_text TEXT NOT NULL DEFAULT '',
                    candidate_hash TEXT NOT NULL,
                    dataset_decision TEXT NOT NULL,
                    provenance_complete INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_manifests (
                    manifest_id TEXT PRIMARY KEY,
                    dataset_version TEXT NOT NULL UNIQUE,
                    policy_version TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'draft',
                    manifest_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    frozen_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS dataset_manifest_items (
                    item_id TEXT PRIMARY KEY,
                    manifest_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_page_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    perceptual_hash TEXT NOT NULL,
                    split TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    candidate_id TEXT NOT NULL DEFAULT '',
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(manifest_id, source_page_id)
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_items_leakage
                    ON dataset_manifest_items(source_document_id, split, source_hash);
                CREATE TABLE IF NOT EXISTS human_review_queue (
                    review_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    assigned_reviewer TEXT NOT NULL DEFAULT '',
                    resolution TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pipeline_activation_readiness (
                    readiness_id TEXT PRIMARY KEY,
                    live_dispatch_status TEXT NOT NULL,
                    training_status TEXT NOT NULL,
                    blocker_codes_json TEXT NOT NULL,
                    checked_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    normalized_email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    display_name TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT 'user'
                        CHECK(role IN ('user','admin')),
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK(status IN ('active','disabled','deletion_pending','deleted')),
                    accepted_terms_version TEXT NOT NULL DEFAULT '',
                    accepted_privacy_version TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_identities (
                    identity_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES auth_users(user_id) ON DELETE CASCADE,
                    provider TEXT NOT NULL,
                    provider_user_id TEXT NOT NULL,
                    email_at_link TEXT NOT NULL DEFAULT '',
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    display_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(provider, provider_user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_auth_identities_user
                    ON auth_identities(user_id);
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES auth_users(user_id) ON DELETE CASCADE,
                    csrf_token_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    absolute_expires_at TEXT NOT NULL,
                    revoked_at TEXT NOT NULL DEFAULT '',
                    user_agent_hash TEXT NOT NULL DEFAULT '',
                    ip_hash TEXT NOT NULL DEFAULT '',
                    idle_timeout_seconds INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                    ON auth_sessions(user_id, revoked_at, expires_at);
                CREATE TABLE IF NOT EXISTS auth_audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    actor_user_id TEXT NOT NULL DEFAULT '',
                    target_user_id TEXT NOT NULL DEFAULT '',
                    target_resource_type TEXT NOT NULL DEFAULT '',
                    target_resource_id TEXT NOT NULL DEFAULT '',
                    result TEXT NOT NULL,
                    correlation_id TEXT NOT NULL DEFAULT '',
                    ip_hash TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_auth_audit_created
                    ON auth_audit_events(created_at DESC);
                CREATE TABLE IF NOT EXISTS guest_sessions (
                    guest_scope_id TEXT PRIMARY KEY,
                    session_id_hash TEXT NOT NULL UNIQUE,
                    claim_token_hash TEXT NOT NULL DEFAULT '',
                    csrf_token_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    ip_hash TEXT NOT NULL DEFAULT '',
                    claimed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS guest_jobs (
                    job_id TEXT PRIMARY KEY,
                    guest_scope_id TEXT NOT NULL REFERENCES guest_sessions(guest_scope_id) ON DELETE CASCADE,
                    owner_user_id TEXT NOT NULL DEFAULT '',
                    original_pdf_name TEXT NOT NULL,
                    stored_pdf_path TEXT NOT NULL,
                    stored_docx_path TEXT NOT NULL DEFAULT '',
                    result_token_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    provider_policy TEXT NOT NULL DEFAULT 'local_free_only',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    claimed_at TEXT NOT NULL DEFAULT '',
                    deleted_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_guest_jobs_scope_status
                    ON guest_jobs(guest_scope_id, status, expires_at);
                CREATE TABLE IF NOT EXISTS auth_usage_events (
                    usage_id TEXT PRIMARY KEY,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    amount INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                """)
            auth_session_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(auth_sessions)"
                ).fetchall()
            }
            if "idle_timeout_seconds" not in auth_session_columns:
                connection.execute(
                    "ALTER TABLE auth_sessions ADD COLUMN idle_timeout_seconds INTEGER NOT NULL DEFAULT 0"
                )
            guest_job_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(guest_jobs)"
                ).fetchall()
            }
            if "downloaded_at" not in guest_job_columns:
                connection.execute(
                    "ALTER TABLE guest_jobs ADD COLUMN downloaded_at TEXT NOT NULL DEFAULT ''"
                )
            guest_session_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(guest_sessions)"
                ).fetchall()
            }
            if "csrf_token_hash" not in guest_session_columns:
                connection.execute(
                    "ALTER TABLE guest_sessions ADD COLUMN csrf_token_hash TEXT NOT NULL DEFAULT ''"
                )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(conversions)"
                ).fetchall()
            }
            if "corrected_docx_path" not in columns:
                connection.execute(
                    "ALTER TABLE conversions ADD COLUMN corrected_docx_path TEXT"
                )
            if "page_numbers" not in columns:
                connection.execute(
                    "ALTER TABLE conversions ADD COLUMN page_numbers TEXT"
                )
            if "actual_char_accuracy" not in columns:
                connection.execute(
                    "ALTER TABLE conversions ADD COLUMN actual_char_accuracy REAL"
                )
            if "actual_word_accuracy" not in columns:
                connection.execute(
                    "ALTER TABLE conversions ADD COLUMN actual_word_accuracy REAL"
                )
            additions = {
                "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "started_at": "TEXT",
                "completed_at": "TEXT",
                "worker_name": "TEXT",
                "last_heartbeat": "TEXT",
                "rq_job_id": "TEXT",
                "claim_token": "TEXT NOT NULL DEFAULT ''",
                "intended_final_status": "TEXT NOT NULL DEFAULT ''",
                "owner_user_id": "TEXT NOT NULL DEFAULT ''",
                "guest_scope_id": "TEXT NOT NULL DEFAULT ''",
                "visibility": "TEXT NOT NULL DEFAULT 'private'",
                "lifecycle_state": "TEXT NOT NULL DEFAULT 'active'",
                "expires_at": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE conversions ADD COLUMN {name} {definition}"
                    )
            version_row = connection.execute(
                "SELECT version FROM schema_meta LIMIT 1"
            ).fetchone()
            previous_schema_version = int(version_row["version"]) if version_row else 0
            if previous_schema_version < 7:
                legacy_rows = connection.execute("""
                    SELECT DISTINCT username FROM conversions
                    WHERE owner_user_id='' AND username LIKE '%@%'
                    """).fetchall()
                for legacy in legacy_rows:
                    email = str(legacy["username"]).strip().casefold()
                    if (
                        not email
                        or "@" not in email
                        or "." not in email.rsplit("@", 1)[-1]
                    ):
                        continue
                    user_id = str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"clouda-legacy:{email}")
                    )
                    now = utc_now()
                    connection.execute(
                        """
                        INSERT INTO auth_users(
                            user_id, normalized_email, email_verified, display_name,
                            role, status, created_at, updated_at, last_login_at
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(user_id) DO NOTHING
                        """,
                        (user_id, email, 0, email, "user", "active", now, now, ""),
                    )
                    connection.execute(
                        """
                        INSERT INTO auth_identities(
                            identity_id, user_id, provider, provider_user_id,
                            email_at_link, email_verified, display_name, created_at, updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(provider, provider_user_id) DO NOTHING
                        """,
                        (
                            str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_URL, f"legacy-identity:{email}"
                                )
                            ),
                            user_id,
                            "legacy",
                            email,
                            email,
                            0,
                            email,
                            now,
                            now,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE conversions
                        SET owner_user_id=?, visibility='private', lifecycle_state='active'
                        WHERE owner_user_id='' AND lower(username)=?
                        """,
                        (user_id, email),
                    )
                connection.execute("""
                    UPDATE conversions
                    SET visibility='legacy', lifecycle_state='legacy_unclaimed'
                    WHERE owner_user_id='' AND guest_scope_id=''
                      AND visibility='private'
                    """)
            account_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(provider_accounts)"
                ).fetchall()
            }
            account_additions = {
                "account_state": "TEXT NOT NULL DEFAULT 'ACTIVE'",
                "organization_id": "TEXT NOT NULL DEFAULT ''",
                "allowed_privacy_json": (
                    'TEXT NOT NULL DEFAULT \'["public","internal"]\''
                ),
                "cooldown_until": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in account_additions.items():
                if name not in account_columns:
                    connection.execute(
                        f"ALTER TABLE provider_accounts ADD COLUMN {name} {definition}"
                    )
            reservation_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(request_reservations)"
                ).fetchall()
            }
            reservation_additions = {
                "request_id": "TEXT NOT NULL DEFAULT ''",
                "parent_request_id": "TEXT NOT NULL DEFAULT ''",
                "attempt_id": "TEXT NOT NULL DEFAULT ''",
                "provider_model_id": "TEXT NOT NULL DEFAULT ''",
                "policy_id": "TEXT NOT NULL DEFAULT ''",
                "reserved_input_tokens": "INTEGER NOT NULL DEFAULT 0",
                "reserved_output_tokens": "INTEGER NOT NULL DEFAULT 0",
                "reserved_cost": "REAL NOT NULL DEFAULT 0",
                "request_count_consumed": "INTEGER NOT NULL DEFAULT 0",
                "token_cost_consumed": "INTEGER NOT NULL DEFAULT 0",
                "monetary_cost_consumed": "INTEGER NOT NULL DEFAULT 0",
                "dispatched_at": "TEXT NOT NULL DEFAULT ''",
                "sent_at": "TEXT NOT NULL DEFAULT ''",
                "fencing_token": "INTEGER NOT NULL DEFAULT 1",
                "uncertainty_reason": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in reservation_additions.items():
                if name not in reservation_columns:
                    connection.execute(
                        f"ALTER TABLE request_reservations ADD COLUMN {name} {definition}"
                    )
            capability_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(capability_endpoints)"
                ).fetchall()
            }
            capability_additions = {
                "supports_text_output": "INTEGER NOT NULL DEFAULT 1",
                "privacy_compatibility_json": "TEXT NOT NULL DEFAULT '[]'",
                "region_support_json": "TEXT NOT NULL DEFAULT '[]'",
                "teacher_role_support_json": "TEXT NOT NULL DEFAULT '[]'",
                "training_output_policy_status": ("TEXT NOT NULL DEFAULT 'unknown'"),
            }
            for name, definition in capability_additions.items():
                if name not in capability_columns:
                    connection.execute(
                        f"ALTER TABLE capability_endpoints ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                "UPDATE conversions SET status = 'pending' WHERE status = 'queued'"
            )
            row = connection.execute(
                "SELECT version FROM schema_meta LIMIT 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,)
                )
            else:
                connection.execute(
                    "UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,)
                )

    @staticmethod
    def hash_secret(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def upsert_auth_user(self, identity) -> dict:
        email = (identity.email or "").strip().casefold()
        if not email:
            raise ValueError("Verified email is required")
        now = utc_now()
        with self.transaction() as connection:
            existing_identity = connection.execute(
                """
                SELECT u.* FROM auth_identities i
                JOIN auth_users u ON u.user_id = i.user_id
                WHERE i.provider = ? AND i.provider_user_id = ?
                """,
                (identity.provider, identity.provider_user_id),
            ).fetchone()
            if existing_identity is not None:
                user_id = existing_identity["user_id"]
                connection.execute(
                    """
                    UPDATE auth_users
                    SET normalized_email=?, email_verified=?, display_name=?,
                        updated_at=?, last_login_at=?
                    WHERE user_id=?
                    """,
                    (
                        email,
                        int(bool(identity.email_verified)),
                        identity.display_name or "",
                        now,
                        now,
                        user_id,
                    ),
                )
            else:
                user_row = connection.execute(
                    "SELECT * FROM auth_users WHERE normalized_email = ?", (email,)
                ).fetchone()
                if user_row is None:
                    user_id = str(uuid.uuid4())
                    connection.execute(
                        """
                        INSERT INTO auth_users(
                            user_id, normalized_email, email_verified, display_name,
                            role, status, created_at, updated_at, last_login_at
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            user_id,
                            email,
                            int(bool(identity.email_verified)),
                            identity.display_name or "",
                            "user",
                            "active",
                            now,
                            now,
                            now,
                        ),
                    )
                else:
                    user_id = user_row["user_id"]
                    if user_row["status"] in {"deleted", "deletion_pending"}:
                        raise PermissionError("Account is not active")
                    connection.execute(
                        """
                        UPDATE auth_users
                        SET email_verified=MAX(email_verified, ?),
                            display_name=CASE WHEN display_name='' THEN ? ELSE display_name END,
                            updated_at=?, last_login_at=?
                        WHERE user_id=?
                        """,
                        (
                            int(bool(identity.email_verified)),
                            identity.display_name or "",
                            now,
                            now,
                            user_id,
                        ),
                    )
                connection.execute(
                    """
                    INSERT INTO auth_identities(
                        identity_id, user_id, provider, provider_user_id,
                        email_at_link, email_verified, display_name, created_at, updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        identity.provider,
                        identity.provider_user_id,
                        email,
                        int(bool(identity.email_verified)),
                        identity.display_name or "",
                        now,
                        now,
                    ),
                )
                self._record_auth_audit_on_connection(
                    connection,
                    "identity_linked",
                    actor_user_id=user_id,
                    target_user_id=user_id,
                    result="success",
                    metadata={"provider": identity.provider},
                )
            row = connection.execute(
                "SELECT * FROM auth_users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row)

    def get_auth_user(self, user_id: str) -> dict | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM auth_users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_auth_users(self, query: str = "") -> list[dict]:
        pattern = f"%{query.strip().casefold()}%"
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT user_id, normalized_email, email_verified, display_name, role,
                       status, created_at, updated_at, last_login_at
                FROM auth_users
                WHERE ? = '%%' OR normalized_email LIKE ?
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (pattern, pattern),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_auth_user_status(
        self, user_id: str, status: str, *, actor_user_id: str
    ) -> dict:
        if status not in {"active", "disabled", "deletion_pending", "deleted"}:
            raise ValueError("Invalid user status")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE auth_users SET status=?, updated_at=? WHERE user_id=?",
                (status, now, user_id),
            )
            if status != "active":
                connection.execute(
                    """
                    UPDATE auth_sessions SET revoked_at=?
                    WHERE user_id=? AND revoked_at=''
                    """,
                    (now, user_id),
                )
            self._record_auth_audit_on_connection(
                connection,
                "user_status_changed",
                actor_user_id=actor_user_id,
                target_user_id=user_id,
                result="success",
                metadata={"status": status},
            )
            row = connection.execute(
                "SELECT * FROM auth_users WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            raise KeyError(user_id)
        return dict(row)

    def set_auth_user_role(
        self, user_id: str, role: str, *, actor_user_id: str
    ) -> dict:
        if role not in {"user", "admin"}:
            raise ValueError("Invalid role")
        if actor_user_id == user_id:
            raise PermissionError("Cannot change own role")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE auth_users SET role=?, updated_at=? WHERE user_id=?",
                (role, now, user_id),
            )
            connection.execute(
                "UPDATE auth_sessions SET revoked_at=? WHERE user_id=? AND revoked_at=''",
                (now, user_id),
            )
            self._record_auth_audit_on_connection(
                connection,
                "role_changed",
                actor_user_id=actor_user_id,
                target_user_id=user_id,
                result="success",
                metadata={"role": role},
            )
            row = connection.execute(
                "SELECT * FROM auth_users WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            raise KeyError(user_id)
        return dict(row)

    def admin_count(self) -> int:
        with closing(self.connect()) as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM auth_users WHERE role='admin' AND status='active'"
                ).fetchone()[0]
            )

    def create_session(
        self,
        user_id: str,
        *,
        lifetime_seconds: int,
        idle_seconds: int,
        ip_hash: str = "",
        user_agent_hash: str = "",
    ) -> dict:
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=lifetime_seconds)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions(
                    session_id_hash, user_id, csrf_token_hash, created_at,
                    last_seen_at, expires_at, absolute_expires_at, ip_hash,
                    user_agent_hash, idle_timeout_seconds
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.hash_secret(session_id),
                    user_id,
                    self.hash_secret(csrf_token),
                    now.isoformat(),
                    now.isoformat(),
                    expires.isoformat(),
                    expires.isoformat(),
                    ip_hash,
                    user_agent_hash,
                    int(idle_seconds),
                ),
            )
        return {
            "session_id": session_id,
            "csrf_token": csrf_token,
            "expires_at": expires.isoformat(),
        }

    def get_session_user(self, session_id: str) -> tuple[dict, dict] | None:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT s.*, u.user_id, u.normalized_email, u.email_verified,
                       u.display_name, u.role, u.status
                FROM auth_sessions s
                JOIN auth_users u ON u.user_id=s.user_id
                WHERE s.session_id_hash=?
                  AND s.revoked_at=''
                  AND s.expires_at > ?
                  AND s.absolute_expires_at > ?
                """,
                (self.hash_secret(session_id), now, now),
            ).fetchone()
            if row is None:
                return None
            data = dict(row)
            last_seen_raw = data.get("last_seen_at") or ""
            idle_timeout = int(data.get("idle_timeout_seconds") or 0)
            if not last_seen_raw or idle_timeout <= 0:
                connection.execute(
                    "UPDATE auth_sessions SET revoked_at=? WHERE session_id_hash=?",
                    (now, self.hash_secret(session_id)),
                )
                return None
            try:
                last_seen = datetime.fromisoformat(last_seen_raw)
            except ValueError:
                connection.execute(
                    "UPDATE auth_sessions SET revoked_at=? WHERE session_id_hash=?",
                    (now, self.hash_secret(session_id)),
                )
                return None
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            if now_dt - last_seen > timedelta(seconds=idle_timeout):
                connection.execute(
                    "UPDATE auth_sessions SET revoked_at=? WHERE session_id_hash=?",
                    (now, self.hash_secret(session_id)),
                )
                return None
            connection.execute(
                """
                UPDATE auth_sessions SET last_seen_at=?
                WHERE session_id_hash=? AND revoked_at='' AND absolute_expires_at > ?
                """,
                (now, self.hash_secret(session_id), now),
            )
        user = {
            key: data[key]
            for key in (
                "user_id",
                "normalized_email",
                "email_verified",
                "display_name",
                "role",
                "status",
            )
        }
        return user, data

    def get_session_by_hash_for_test(self, session_id_hash: str) -> dict:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM auth_sessions WHERE session_id_hash=?",
                (session_id_hash,),
            ).fetchone()
        if row is None:
            raise KeyError(session_id_hash)
        return dict(row)

    def session_csrf_matches(self, session_id: str, csrf_token: str | None) -> bool:
        if not csrf_token:
            return False
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT csrf_token_hash FROM auth_sessions
                WHERE session_id_hash=? AND revoked_at=''
                """,
                (self.hash_secret(session_id),),
            ).fetchone()
        return bool(
            row and secrets.compare_digest(row[0], self.hash_secret(csrf_token))
        )

    def revoke_session(self, session_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE auth_sessions SET revoked_at=? WHERE session_id_hash=?",
                (utc_now(), self.hash_secret(session_id)),
            )

    def revoke_user_sessions(self, user_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE auth_sessions SET revoked_at=? WHERE user_id=? AND revoked_at=''",
                (utc_now(), user_id),
            )

    def _record_auth_audit_on_connection(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        *,
        actor_user_id: str = "",
        target_user_id: str = "",
        target_resource_type: str = "",
        target_resource_id: str = "",
        result: str = "success",
        correlation_id: str = "",
        ip_hash: str = "",
        metadata: dict | None = None,
    ) -> None:
        import json

        safe_metadata = metadata or {}
        connection.execute(
            """
            INSERT INTO auth_audit_events(
                event_id,event_type,actor_user_id,target_user_id,
                target_resource_type,target_resource_id,result,correlation_id,
                ip_hash,metadata_json,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                event_type,
                actor_user_id,
                target_user_id,
                target_resource_type,
                target_resource_id,
                result,
                correlation_id,
                ip_hash,
                json.dumps(safe_metadata, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )

    def record_auth_audit(self, event_type: str, **kwargs) -> None:
        with self.transaction() as connection:
            self._record_auth_audit_on_connection(connection, event_type, **kwargs)

    def list_auth_audit_events(self) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM auth_audit_events ORDER BY created_at DESC, event_id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def create_guest_session(self, *, lifetime_seconds: int, ip_hash: str = "") -> dict:
        from datetime import timedelta

        guest_session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        guest_scope_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=lifetime_seconds)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO guest_sessions(
                    guest_scope_id,session_id_hash,csrf_token_hash,created_at,
                    expires_at,last_seen_at,ip_hash
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    guest_scope_id,
                    self.hash_secret(guest_session_id),
                    self.hash_secret(csrf_token),
                    now.isoformat(),
                    expires.isoformat(),
                    now.isoformat(),
                    ip_hash,
                ),
            )
        return {
            "guest_session_id": guest_session_id,
            "csrf_token": csrf_token,
            "guest_scope_id": guest_scope_id,
            "expires_at": expires.isoformat(),
        }

    def get_guest_session(self, guest_session_id: str) -> dict | None:
        now = utc_now()
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM guest_sessions
                WHERE session_id_hash=? AND expires_at > ?
                """,
                (self.hash_secret(guest_session_id), now),
            ).fetchone()
        return dict(row) if row else None

    def guest_has_active_job(self, guest_scope_id: str) -> bool:
        with closing(self.connect()) as connection:
            return bool(
                connection.execute(
                    """
                    SELECT 1 FROM guest_jobs
                    WHERE guest_scope_id=? AND status IN ('pending','processing','ready')
                      AND claimed_at='' AND deleted_at='' AND expires_at > ?
                    LIMIT 1
                    """,
                    (guest_scope_id, utc_now()),
                ).fetchone()
            )

    def create_guest_job(self, values: dict) -> dict:
        now = utc_now()
        job_id = values.get("job_id") or uuid.uuid4().hex
        result_token = values.get("result_token") or secrets.token_urlsafe(24)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO guest_jobs(
                    job_id,guest_scope_id,original_pdf_name,stored_pdf_path,
                    stored_docx_path,result_token_hash,status,page_count,size_bytes,
                    provider_policy,created_at,expires_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    values["guest_scope_id"],
                    values["original_pdf_name"],
                    values["stored_pdf_path"],
                    values.get("stored_docx_path", ""),
                    self.hash_secret(result_token),
                    values.get("status", "pending"),
                    int(values["page_count"]),
                    int(values["size_bytes"]),
                    "local_free_only",
                    now,
                    values["expires_at"],
                ),
            )
        row = self.get_guest_job(job_id)
        if row is None:
            raise RuntimeError("Guest job was not persisted")
        row["result_token"] = result_token
        return row

    def get_guest_job(
        self, job_id: str, *, guest_scope_id: str | None = None
    ) -> dict | None:
        sql = "SELECT * FROM guest_jobs WHERE job_id=? AND deleted_at=''"
        params: tuple = (job_id,)
        if guest_scope_id is not None:
            sql += " AND guest_scope_id=? AND expires_at > ?"
            params += (guest_scope_id, utc_now())
        with closing(self.connect()) as connection:
            row = connection.execute(sql, params).fetchone()
        return dict(row) if row else None

    def issue_guest_claim_token(self, guest_scope_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE guest_sessions SET claim_token_hash=?
                WHERE guest_scope_id=? AND expires_at > ?
                """,
                (self.hash_secret(token), guest_scope_id, utc_now()),
            )
        return token

    def guest_csrf_matches(self, guest_session_id: str, csrf_token: str | None) -> bool:
        if not csrf_token:
            return False
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT csrf_token_hash FROM guest_sessions
                WHERE session_id_hash=? AND expires_at > ?
                """,
                (self.hash_secret(guest_session_id), utc_now()),
            ).fetchone()
        return bool(
            row and secrets.compare_digest(row[0], self.hash_secret(csrf_token))
        )

    def claim_guest_job(
        self, job_id: str, guest_scope_id: str, claim_token: str, owner_user_id: str
    ) -> dict:
        now = utc_now()
        with self.transaction() as connection:
            guest = connection.execute(
                """
                SELECT * FROM guest_sessions
                WHERE guest_scope_id=? AND expires_at > ?
                """,
                (guest_scope_id, now),
            ).fetchone()
            if guest is None:
                raise PermissionError("Invalid guest claim")
            if guest["claimed_at"]:
                raise KeyError(job_id)
            if not secrets.compare_digest(
                guest["claim_token_hash"], self.hash_secret(claim_token)
            ):
                raise PermissionError("Invalid guest claim")
            job = connection.execute(
                """
                SELECT * FROM guest_jobs
                WHERE job_id=? AND guest_scope_id=? AND claimed_at='' AND deleted_at=''
                  AND expires_at > ?
                """,
                (job_id, guest_scope_id, now),
            ).fetchone()
            if job is None:
                raise KeyError(job_id)
            connection.execute(
                """
                UPDATE guest_jobs SET owner_user_id=?, claimed_at=?, status='claimed'
                WHERE job_id=?
                """,
                (owner_user_id, now, job_id),
            )
            connection.execute(
                "UPDATE guest_sessions SET claimed_at=? WHERE guest_scope_id=?",
                (now, guest_scope_id),
            )
            self._record_auth_audit_on_connection(
                connection,
                "guest_job_claimed",
                actor_user_id=owner_user_id,
                target_user_id=owner_user_id,
                target_resource_type="guest_job",
                target_resource_id=job_id,
                result="success",
            )
            row = connection.execute(
                "SELECT * FROM guest_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row)

    def expire_guest_jobs(self, *, now_offset_seconds: int = 0) -> int:
        from datetime import timedelta

        now = datetime.now(timezone.utc) + timedelta(seconds=now_offset_seconds)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE guest_jobs SET deleted_at=?
                WHERE deleted_at='' AND claimed_at='' AND expires_at <= ?
                """,
                (now.isoformat(), now.isoformat()),
            )
        return cursor.rowcount

    def consume_daily_quota(
        self, *, scope_type: str, scope_id: str, endpoint: str, limit: int
    ) -> bool:
        if limit <= 0:
            return True
        now_dt = datetime.now(timezone.utc)
        window_start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(amount), 0) FROM auth_usage_events
                    WHERE scope_type=? AND scope_id=? AND endpoint=?
                      AND created_at >= ?
                    """,
                    (scope_type, scope_id, endpoint, window_start.isoformat()),
                ).fetchone()[0]
            )
            if count >= limit:
                return False
            connection.execute(
                """
                INSERT INTO auth_usage_events(
                    usage_id, scope_type, scope_id, endpoint, amount, created_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    uuid.uuid4().hex,
                    scope_type,
                    scope_id,
                    endpoint,
                    1,
                    now_dt.isoformat(),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return True

    def consume_window_quota(
        self,
        *,
        scope_type: str,
        scope_id: str,
        endpoint: str,
        limit: int,
        window_seconds: int,
    ) -> bool:
        if limit <= 0:
            return True
        now_dt = datetime.now(timezone.utc)
        window_start = now_dt - timedelta(seconds=window_seconds)
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(amount), 0) FROM auth_usage_events
                    WHERE scope_type=? AND scope_id=? AND endpoint=?
                      AND created_at >= ?
                    """,
                    (scope_type, scope_id, endpoint, window_start.isoformat()),
                ).fetchone()[0]
            )
            if count >= limit:
                return False
            connection.execute(
                """
                INSERT INTO auth_usage_events(
                    usage_id, scope_type, scope_id, endpoint, amount, created_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    uuid.uuid4().hex,
                    scope_type,
                    scope_id,
                    endpoint,
                    1,
                    now_dt.isoformat(),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return True

    def mark_guest_job_result(
        self, job_id: str, *, status: str, stored_docx_path: str
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE guest_jobs
                SET status=?, stored_docx_path=?
                WHERE job_id=? AND deleted_at=''
                """,
                (status, stored_docx_path, job_id),
            )

    def consume_guest_result_token(
        self, job_id: str, guest_scope_id: str, token: str
    ) -> dict | None:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM guest_jobs
                WHERE job_id=? AND guest_scope_id=? AND deleted_at=''
                  AND expires_at > ? AND downloaded_at=''
                """,
                (job_id, guest_scope_id, now),
            ).fetchone()
            if row is None:
                return None
            if not secrets.compare_digest(
                row["result_token_hash"], self.hash_secret(token)
            ):
                return None
            connection.execute(
                "UPDATE guest_jobs SET downloaded_at=? WHERE job_id=?",
                (now, job_id),
            )
        return dict(row)

    def list_owner_conversions(self, owner_user_id: str) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversions
                WHERE owner_user_id=? AND hidden=0
                ORDER BY created_at DESC
                """,
                (owner_user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_owner_conversion(self, job_id: str, owner_user_id: str) -> dict | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM conversions
                WHERE job_id=? AND owner_user_id=? AND hidden=0
                """,
                (job_id, owner_user_id),
            ).fetchone()
        return dict(row) if row else None

    def login(self, username: str) -> None:
        clean = username.strip()
        if not clean:
            raise ValueError("اسم المستخدم مطلوب")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO users(username, created_at, last_login)
                VALUES (?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET last_login = excluded.last_login
                """,
                (clean, now, now),
            )

    def mark_incomplete_as_interrupted(self) -> int:
        """Legacy compatibility; distributed pending jobs must survive UI restarts."""
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE conversions
                SET status = 'pending', updated_at = ?,
                    error_message = COALESCE(error_message, 'ستعاد المهمة إلى عامل المعالجة')
                WHERE status = 'interrupted'
                """,
                (utc_now(),),
            )
            return cursor.rowcount

    def create_conversion(self, values: dict) -> int:
        invalid = set(values) - CONVERSION_INSERT_FIELDS
        if invalid:
            raise ValueError(f"Unsupported conversion fields: {sorted(invalid)}")
        fields = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"INSERT INTO conversions ({fields}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Conversion insert did not return an id")
            return int(cursor.lastrowid)

    def list_conversions(
        self, username: str | None, include_hidden: bool = False
    ) -> list[dict]:
        hidden_clause = "" if include_hidden else "AND hidden = 0"
        where_clause = "WHERE username = ?" if username is not None else "WHERE 1 = 1"
        params = (username,) if username is not None else ()
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM conversions
                {where_clause} {hidden_clause}
                ORDER BY created_at DESC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversion(self, job_id: str, username: str | None = None) -> dict | None:
        sql = "SELECT * FROM conversions WHERE job_id = ?"
        params: tuple = (job_id,)
        if username is not None:
            sql += " AND username = ?"
            params = (job_id, username)
        with closing(self.connect()) as connection:
            row = connection.execute(sql, params).fetchone()
        return dict(row) if row else None

    def list_pending_conversions(self) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM conversions WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_conversion(self, conversion_id: int, values: dict) -> None:
        allowed = {
            "stored_docx_path",
            "file_type",
            "text_quality_score",
            "layout_quality_score",
            "final_quality_score",
            "winning_engine",
            "winning_model",
            "total_cost",
            "processing_time",
            "status",
            "hidden",
            "updated_at",
            "error_message",
            "corrected_docx_path",
            "actual_char_accuracy",
            "actual_word_accuracy",
            "attempt_count",
            "started_at",
            "completed_at",
            "worker_name",
            "last_heartbeat",
            "rq_job_id",
            "claim_token",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE conversions SET {assignments} WHERE id = ?",
                (*updates.values(), conversion_id),
            )

    def transition_conversion(
        self,
        job_id: str,
        target_status: str,
        *,
        worker_name: str | None = None,
        error_message: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        if target_status not in ALLOWED_STATUS_TRANSITIONS:
            raise ValueError(f"Unsupported status: {target_status}")
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversions WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = row["status"]
            if (
                current != target_status
                and target_status not in ALLOWED_STATUS_TRANSITIONS.get(current, set())
            ):
                raise ValueError(
                    f"Invalid status transition: {current} -> {target_status}"
                )
            values: dict[str, Any] = {
                "status": target_status,
                "updated_at": now,
                "error_message": error_message,
            }
            if worker_name is not None:
                values["worker_name"] = worker_name
            if target_status == "processing":
                values["started_at"] = now
                values["last_heartbeat"] = now
                values["attempt_count"] = int(row["attempt_count"] or 0) + 1
                values["claim_token"] = uuid.uuid4().hex
            if target_status in FINAL_STATUSES:
                values["completed_at"] = now
            for key, value in (extra or {}).items():
                if key in {
                    "stored_docx_path",
                    "file_type",
                    "text_quality_score",
                    "layout_quality_score",
                    "final_quality_score",
                    "winning_engine",
                    "winning_model",
                    "total_cost",
                    "processing_time",
                    "rq_job_id",
                    "last_heartbeat",
                    "claim_token",
                }:
                    values[key] = value
            assignments = ", ".join(f"{key} = ?" for key in values)
            connection.execute(
                f"UPDATE conversions SET {assignments} WHERE id = ?",
                (*values.values(), row["id"]),
            )
        result = self.get_conversion(job_id)
        assert result is not None
        return result

    def finalize_conversion_if_owned(
        self,
        job_id: str,
        target_status: str,
        *,
        worker_name: str,
        claim_token: str | None,
        error_message: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        if target_status not in FINAL_STATUSES:
            raise ValueError(f"Unsupported final status: {target_status}")
        now = utc_now()
        values: dict[str, Any] = {
            "status": target_status,
            "updated_at": now,
            "completed_at": now,
            "worker_name": worker_name,
            "error_message": error_message,
        }
        for key, value in (extra or {}).items():
            if key in {
                "stored_docx_path",
                "file_type",
                "text_quality_score",
                "layout_quality_score",
                "final_quality_score",
                "winning_engine",
                "winning_model",
                "total_cost",
                "processing_time",
            }:
                values[key] = value
        assignments = ", ".join(f"{key} = ?" for key in values)
        expected_token = claim_token or LEGACY_EMPTY_CLAIM_TOKEN
        with self.transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE conversions SET {assignments}
                WHERE job_id=? AND status='processing' AND worker_name=?
                  AND claim_token=?
                """,
                (*values.values(), job_id, worker_name, expected_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("Worker does not own this processing job")
            row = connection.execute(
                "SELECT * FROM conversions WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def prepare_conversion_finalization(
        self,
        job_id: str,
        target_status: str,
        *,
        worker_name: str,
        claim_token: str | None,
        error_message: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        if target_status not in {"completed", "manual_review"}:
            raise ValueError(f"Unsupported final status: {target_status}")
        now = utc_now()
        values: dict[str, Any] = {
            "status": "finalizing",
            "updated_at": now,
            "completed_at": "",
            "worker_name": worker_name,
            "error_message": error_message,
            "intended_final_status": target_status,
        }
        for key, value in (extra or {}).items():
            if key in {
                "stored_docx_path",
                "file_type",
                "text_quality_score",
                "layout_quality_score",
                "final_quality_score",
                "winning_engine",
                "winning_model",
                "total_cost",
                "processing_time",
            }:
                values[key] = value
        assignments = ", ".join(f"{key} = ?" for key in values)
        expected_token = claim_token or LEGACY_EMPTY_CLAIM_TOKEN
        with self.transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE conversions SET {assignments}
                WHERE job_id=? AND status='processing' AND worker_name=?
                  AND claim_token=?
                """,
                (*values.values(), job_id, worker_name, expected_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("Worker does not own this processing job")
            row = connection.execute(
                "SELECT * FROM conversions WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def complete_conversion_finalization(
        self,
        job_id: str,
        target_status: str,
        *,
        worker_name: str,
        claim_token: str | None,
    ) -> dict:
        if target_status not in {"completed", "manual_review"}:
            raise ValueError(f"Unsupported final status: {target_status}")
        now = utc_now()
        expected_token = claim_token or LEGACY_EMPTY_CLAIM_TOKEN
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE conversions
                SET status=?,
                    updated_at=?,
                    completed_at=?,
                    intended_final_status=''
                WHERE job_id=? AND status='finalizing' AND worker_name=?
                  AND claim_token=?
                """,
                (target_status, now, now, job_id, worker_name, expected_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("Worker does not own this finalizing job")
            row = connection.execute(
                "SELECT * FROM conversions WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def rollback_conversion_finalization(
        self,
        job_id: str,
        *,
        worker_name: str,
        claim_token: str | None,
    ) -> None:
        expected_token = claim_token or LEGACY_EMPTY_CLAIM_TOKEN
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE conversions
                SET status='processing',
                    updated_at=?,
                    completed_at='',
                    error_message=NULL,
                    intended_final_status='',
                    file_type=NULL,
                    text_quality_score=NULL,
                    layout_quality_score=NULL,
                    final_quality_score=NULL,
                    winning_engine=NULL,
                    winning_model=NULL,
                    total_cost=0,
                    processing_time=0
                WHERE job_id=? AND status='finalizing' AND worker_name=?
                  AND claim_token=?
                """,
                (utc_now(), job_id, worker_name, expected_token),
            )

    def abandon_conversion_finalization(
        self, job_id: str, *, observed_updated_at: str
    ) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE conversions
                SET status='pending',
                    updated_at=?,
                    completed_at='',
                    worker_name=NULL,
                    last_heartbeat=NULL,
                    claim_token='',
                    error_message=NULL,
                    intended_final_status='',
                    file_type=NULL,
                    text_quality_score=NULL,
                    layout_quality_score=NULL,
                    final_quality_score=NULL,
                    winning_engine=NULL,
                    winning_model=NULL,
                    total_cost=0,
                    processing_time=0
                WHERE job_id=? AND status='finalizing' AND updated_at=?
                """,
                (utc_now(), job_id, observed_updated_at),
            )
            return cursor.rowcount == 1

    def abandon_stale_processing(
        self, job_id: str, *, observed_updated_at: str
    ) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE conversions
                SET status='pending',
                    updated_at=?,
                    worker_name=NULL,
                    last_heartbeat=NULL,
                    claim_token='',
                    error_message='Worker heartbeat expired; job returned to pending.'
                WHERE job_id=? AND status='processing' AND updated_at=?
                """,
                (utc_now(), job_id, observed_updated_at),
            )
            return cursor.rowcount == 1

    def heartbeat(self, job_id: str, worker_name: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE conversions
                SET last_heartbeat = ?, updated_at = ?, worker_name = ?
                WHERE job_id = ? AND status = 'processing'
                """,
                (utc_now(), utc_now(), worker_name, job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Job is not processing")

    def claim_matches(
        self, row: dict, worker_name: str, claim_token: str | None = None
    ) -> bool:
        if row.get("worker_name") != worker_name:
            return False
        expected = row.get("claim_token") or LEGACY_EMPTY_CLAIM_TOKEN
        if not expected:
            return True
        return bool(claim_token) and claim_token == expected

    def record_attempt(self, values: dict) -> int:
        invalid = set(values) - ATTEMPT_INSERT_FIELDS
        if invalid:
            raise ValueError(f"Unsupported attempt fields: {sorted(invalid)}")
        fields = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"INSERT INTO attempts ({fields}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            connection.execute(
                """
                UPDATE conversions
                SET total_cost = (
                    SELECT COALESCE(SUM(cost), 0) FROM attempts WHERE conversion_id = ?
                ), updated_at = ?
                WHERE id = ?
                """,
                (values["conversion_id"], utc_now(), values["conversion_id"]),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Attempt insert did not return an id")
            return int(cursor.lastrowid)

    def list_attempts(self, conversion_id: int) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM attempts WHERE conversion_id = ? ORDER BY attempt_number, id",
                (conversion_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_setting_rows(self) -> dict[str, dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT * FROM project_settings").fetchall()
        return {row["key"]: dict(row) for row in rows}

    def set_setting(self, key: str, value: str, value_type: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO project_settings(key, value, type, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    type = excluded.type,
                    updated_at = excluded.updated_at
                """,
                (key, value, value_type, utc_now()),
            )

    def clear_settings(self) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM project_settings")

    def daily_cost(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(cost), 0) AS value FROM attempts WHERE created_at >= ?",
                (today,),
            ).fetchone()
        return float(row["value"] or 0)

    def statistics(
        self,
        username: str | None = None,
        file_type: str | None = None,
        engine: str | None = None,
    ) -> dict:
        clauses: list[str] = []
        params_list: list = []
        if username:
            clauses.append("username = ?")
            params_list.append(username)
        if file_type:
            clauses.append("file_type = ?")
            params_list.append(file_type)
        if engine:
            clauses.append("winning_engine = ?")
            params_list.append(engine)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params = tuple(params_list)
        with closing(self.connect()) as connection:
            summary = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN created_at >= date('now') THEN 1 ELSE 0 END) AS today,
                    SUM(CASE WHEN created_at >= date('now', '-6 days') THEN 1 ELSE 0 END) AS week,
                    SUM(CASE WHEN created_at >= date('now', 'start of month') THEN 1 ELSE 0 END) AS month,
                    AVG(text_quality_score) AS avg_text,
                    AVG(layout_quality_score) AS avg_layout,
                    AVG(final_quality_score) AS avg_final,
                    AVG(processing_time) AS avg_time,
                    SUM(total_cost) AS total_cost,
                    SUM(CASE WHEN created_at >= date('now') THEN total_cost ELSE 0 END) AS cost_today,
                    SUM(CASE WHEN created_at >= date('now', '-6 days') THEN total_cost ELSE 0 END) AS cost_week,
                    SUM(CASE WHEN created_at >= date('now', 'start of month') THEN total_cost ELSE 0 END) AS cost_month,
                    SUM(CASE WHEN status = 'completed' AND text_quality_score >= 90 THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN text_quality_score < 90 THEN 1 ELSE 0 END) AS below_text_threshold,
                    SUM(CASE WHEN file_type = 'digital' THEN 1 ELSE 0 END) AS digital,
                    SUM(CASE WHEN file_type = 'scan' THEN 1 ELSE 0 END) AS scan
                FROM conversions {where}
                """,
                params,
            ).fetchone()
            winner_where = (
                f"{where} {'AND' if where else 'WHERE'} winning_engine IS NOT NULL"
            )
            winner = connection.execute(
                f"""
                SELECT winning_engine, COUNT(*) AS uses
                FROM conversions {winner_where}
                GROUP BY winning_engine ORDER BY uses DESC LIMIT 1
                """,
                params,
            ).fetchone()
            failure = connection.execute("""
                SELECT failure_reason, COUNT(*) AS uses
                FROM attempts
                WHERE success = 0 AND failure_reason IS NOT NULL
                GROUP BY failure_reason ORDER BY uses DESC LIMIT 1
                """).fetchone()
            failed_attempts = connection.execute(
                "SELECT COUNT(*) AS value FROM attempts WHERE success = 0"
            ).fetchone()
        result = dict(summary)
        total = int(result.get("total") or 0)
        result["success_rate"] = (
            (int(result.get("completed") or 0) / total * 100.0) if total else 0.0
        )
        result["top_engine"] = winner["winning_engine"] if winner else None
        result["failed_attempts"] = int(failed_attempts["value"] or 0)
        result["top_failure"] = failure["failure_reason"] if failure else None
        return result

    def export_conversions_csv(self, username: str | None = None) -> str:
        import csv
        import io

        rows = self.list_conversions(username, include_hidden=True) if username else []
        if username is None:
            with closing(self.connect()) as connection:
                rows = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM conversions ORDER BY created_at DESC"
                    )
                ]
        if not rows:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: sanitize_spreadsheet_cell(value) for key, value in row.items()}
            for row in rows
        )
        return output.getvalue()

    def add_correction_rule(
        self, pattern: str, replacement: str, rule_type: str
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO correction_rules(pattern, replacement, rule_type, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(pattern, replacement, rule_type)
                DO UPDATE SET occurrences = occurrences + 1
                """,
                (pattern, replacement, rule_type, utc_now()),
            )

    def list_correction_rules(self) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM correction_rules ORDER BY occurrences DESC, id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def enabled_correction_rules(self) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute("""
                SELECT * FROM correction_rules
                WHERE approved = 1 AND enabled = 1
                ORDER BY occurrences DESC, id
                """).fetchall()
        return [dict(row) for row in rows]

    def set_correction_rule_state(
        self, rule_id: int, approved: bool, enabled: bool
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE correction_rules SET approved = ?, enabled = ? WHERE id = ?",
                (int(approved), int(enabled and approved), rule_id),
            )

    def delete_correction_rule(self, rule_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM correction_rules WHERE id = ?", (rule_id,))

    def create_correction_revision(
        self,
        job_id: str,
        username: str,
        stored_path: str,
        file_hash: str,
        file_size: int,
        learning_status: str = "review_only",
    ) -> dict:
        row = self.get_conversion(job_id, username)
        if row is None:
            raise PermissionError("Conversion does not belong to this user")
        with self.transaction() as connection:
            version = int(
                connection.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM correction_revisions WHERE job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO correction_revisions(
                    job_id,user_id,version,stored_path,file_hash,file_size,uploaded_at,
                    uploaded_by,comparison_status,learning_status
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    username,
                    version,
                    stored_path,
                    file_hash,
                    file_size,
                    utc_now(),
                    username,
                    "pending",
                    learning_status,
                ),
            )
            revision_id = cursor.lastrowid
            if revision_id is None:
                raise RuntimeError("Correction revision insert did not return an id")
        revision = self.get_correction_revision(int(revision_id), username)
        if revision is None:
            raise RuntimeError("Correction revision was not persisted")
        return revision

    def get_correction_revision(
        self, revision_id: int, username: str | None = None
    ) -> dict | None:
        sql = "SELECT * FROM correction_revisions WHERE id = ?"
        params: tuple = (revision_id,)
        if username is not None:
            sql += " AND user_id = ?"
            params += (username,)
        with closing(self.connect()) as connection:
            row = connection.execute(sql, params).fetchone()
        return dict(row) if row else None

    def list_correction_revisions(self, job_id: str, username: str) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM correction_revisions
                WHERE job_id = ? AND user_id = ? ORDER BY version DESC
                """,
                (job_id, username),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_correction_revision(self, revision_id: int, **values) -> None:
        allowed = {
            "stored_path",
            "comparison_status",
            "learning_status",
            "change_count",
        }
        clean = {key: value for key, value in values.items() if key in allowed}
        if not clean:
            return
        assignments = ", ".join(f"{key} = ?" for key in clean)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE correction_revisions SET {assignments} WHERE id = ?",
                (*clean.values(), revision_id),
            )

    def replace_revision_examples(
        self, revision_id: int, job_id: str, username: str, examples: list[dict]
    ) -> None:
        if self.get_correction_revision(revision_id, username) is None:
            raise PermissionError("Revision does not belong to this user")
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM correction_examples WHERE revision_id = ?", (revision_id,)
            )
            for example in examples:
                connection.execute(
                    """
                    INSERT INTO correction_examples(
                        revision_id,job_id,user_id,wrong_text,correct_text,context_before,
                        context_after,paragraph_index,sentence_index,change_type,language,
                        is_sensitive,status,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        revision_id,
                        job_id,
                        username,
                        example.get("wrong_text", ""),
                        example.get("correct_text", ""),
                        example.get("context_before", ""),
                        example.get("context_after", ""),
                        example.get("paragraph_index"),
                        example.get("sentence_index"),
                        example["change_type"],
                        example.get("language", "unknown"),
                        int(bool(example.get("is_sensitive"))),
                        "proposed",
                        utc_now(),
                    ),
                )
        self.update_correction_revision(
            revision_id, comparison_status="completed", change_count=len(examples)
        )

    def list_correction_examples(
        self, revision_id: int | None = None, status: str | None = None
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if revision_id is not None:
            clauses.append("revision_id = ?")
            params.append(revision_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM correction_examples" + where + " ORDER BY id", params
            ).fetchall()
        return [dict(row) for row in rows]

    def review_correction_example(
        self, example_id: int, action: str, reviewer: str
    ) -> None:
        if action not in {"approved", "rejected", "ignored"}:
            raise ValueError("Invalid review action")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT e.*,r.learning_status AS revision_learning_status
                FROM correction_examples e
                JOIN correction_revisions r ON r.id=e.revision_id
                WHERE e.id = ?
                """,
                (example_id,),
            ).fetchone()
            if row is None:
                raise KeyError(example_id)
            if row["user_id"].casefold() != reviewer.casefold():
                raise PermissionError("Example does not belong to this user")
            approved_at = utc_now() if action == "approved" else None
            rejected_at = utc_now() if action == "rejected" else None
            connection.execute(
                """
                UPDATE correction_examples SET status=?,approved_by=?,approved_at=?,rejected_at=?
                WHERE id=?
                """,
                (
                    action,
                    reviewer if action == "approved" else None,
                    approved_at,
                    rejected_at,
                    example_id,
                ),
            )
            if (
                action == "approved"
                and row["revision_learning_status"] == "learning"
                and not row["is_sensitive"]
                and row["wrong_text"]
                and row["correct_text"]
                and not str(row["change_type"]).startswith("formatting:")
            ):
                context = (row["context_before"] or "")[-30:]
                connection.execute(
                    """
                    INSERT INTO correction_memory_rules(
                        wrong_text,correct_text,context_pattern,scope,language,source_count,
                        occurrence_count,accepted_count,confidence,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(wrong_text,correct_text,context_pattern,scope,scope_value)
                    DO UPDATE SET
                        occurrence_count=occurrence_count+1,
                        accepted_count=accepted_count+1,
                        source_count=(
                            SELECT COUNT(DISTINCT job_id) FROM correction_examples
                            WHERE wrong_text=excluded.wrong_text AND correct_text=excluded.correct_text
                              AND status='approved'
                        ),
                        updated_at=excluded.updated_at
                    """,
                    (
                        row["wrong_text"],
                        row["correct_text"],
                        context,
                        "global",
                        row["language"],
                        1,
                        1,
                        1,
                        0.0,
                        utc_now(),
                    ),
                )
        self.recalculate_correction_confidence()

    def recalculate_correction_confidence(self) -> None:
        with self.transaction() as connection:
            connection.execute("""
                UPDATE correction_memory_rules SET confidence =
                    MIN(0.99, (accepted_count * 1.0 / MAX(1, accepted_count + rejected_count))
                    * MIN(1.0, source_count / 3.0))
                """)

    def list_memory_rules(self) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM correction_memory_rules ORDER BY confidence DESC,id"
            ).fetchall()
        return [dict(row) for row in rows]

    def set_memory_rule_state(
        self, rule_id: int, approved: bool, enabled: bool, reviewer: str
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE correction_memory_rules
                SET approved=?,enabled=?,approved_by=?,approved_at=?,updated_at=?
                WHERE id=? AND is_sensitive=0
                """,
                (
                    int(approved),
                    int(approved and enabled),
                    reviewer if approved else None,
                    utc_now() if approved else None,
                    utc_now(),
                    rule_id,
                ),
            )

    def active_memory_rules(
        self, min_sources: int = 3, threshold: float = 0.9
    ) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM correction_memory_rules
                WHERE approved=1 AND enabled=1 AND is_sensitive=0
                  AND source_count>=? AND confidence>=?
                ORDER BY confidence DESC,id
                """,
                (min_sources, threshold),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_correction_applications(
        self, job_id: str, applications: list[dict]
    ) -> None:
        with self.transaction() as connection:
            for item in applications:
                connection.execute(
                    """
                    INSERT INTO correction_applications(
                        job_id,rule_id,before_text,after_text,confidence,context_match,applied_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        job_id,
                        item["rule_id"],
                        item["before"],
                        item["after"],
                        item["confidence"],
                        int(bool(item["context_match"])),
                        utc_now(),
                    ),
                )

    def correction_readiness(self) -> dict:
        with closing(self.connect()) as connection:
            corrected = connection.execute(
                "SELECT COUNT(DISTINCT job_id) FROM correction_revisions"
            ).fetchone()[0]
            learning = connection.execute(
                "SELECT COUNT(DISTINCT job_id) FROM correction_revisions WHERE learning_status='learning'"
            ).fetchone()[0]
            examples = connection.execute(
                "SELECT COUNT(*) FROM correction_examples"
            ).fetchone()[0]
            trusted = connection.execute(
                "SELECT COUNT(*) FROM correction_memory_rules WHERE approved=1 AND enabled=1"
            ).fetchone()[0]
            evaluation = connection.execute(
                "SELECT COUNT(*) FROM correction_evaluations WHERE dataset_split='evaluation'"
            ).fetchone()[0]
        label = (
            "insufficient"
            if corrected < 10
            else "early" if corrected < 50 else "evaluation_ready"
        )
        return {
            "corrected_files": corrected,
            "learning_files": learning,
            "examples": examples,
            "trusted_rules": trusted,
            "evaluation_size": evaluation,
            "readiness": label,
        }

    def record_correction_evaluation(
        self,
        revision_id: int,
        job_id: str,
        dataset_split: str,
        pre_metrics: dict,
        post_metrics: dict,
    ) -> None:
        if dataset_split not in {"training", "evaluation"}:
            raise ValueError("Invalid dataset split")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO correction_evaluations(
                    revision_id,job_id,dataset_split,pre_word_accuracy,pre_char_accuracy,
                    post_word_accuracy,post_char_accuracy,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    revision_id,
                    job_id,
                    dataset_split,
                    pre_metrics.get("word_accuracy"),
                    pre_metrics.get("char_accuracy"),
                    post_metrics.get("word_accuracy"),
                    post_metrics.get("char_accuracy"),
                    utc_now(),
                ),
            )

    def correction_evaluation_summary(self) -> dict:
        with closing(self.connect()) as connection:
            row = connection.execute("""
                SELECT COUNT(*) AS samples,AVG(pre_word_accuracy) AS before_accuracy,
                       AVG(post_word_accuracy) AS after_accuracy
                FROM correction_evaluations WHERE dataset_split='evaluation'
                """).fetchone()
        result = dict(row)
        before = result.get("before_accuracy")
        after = result.get("after_accuracy")
        result["improvement"] = (
            None if before is None or after is None else after - before
        )
        return result

    def export_approved_dataset(self) -> str:
        import hashlib
        import json

        with closing(self.connect()) as connection:
            rows = connection.execute("""
                SELECT e.wrong_text,e.correct_text,e.context_before,e.context_after,e.language,
                       e.change_type,r.job_id
                FROM correction_examples e
                JOIN correction_revisions r ON r.id=e.revision_id
                WHERE e.status='approved' AND e.is_sensitive=0
                ORDER BY e.id
                """).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            job_id = item.pop("job_id")
            item["context"] = (
                f"{item.pop('context_before')} … {item.pop('context_after')}".strip()
            )
            item["scope"] = "global"
            item["document_type"] = ""
            item["source_job_hash"] = hashlib.sha256(job_id.encode()).hexdigest()[:16]
            output.append(item)
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in output)

    def record_backup(
        self,
        path: str,
        status: str,
        size_bytes: int = 0,
        error_message: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO backup_history(backup_path, status, size_bytes, error_message, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (path, status, size_bytes, error_message, utc_now()),
            )

    def last_backup(self) -> dict | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM backup_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def hide_conversion(self, job_id: str, username: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE conversions
                SET hidden = 1, updated_at = ?
                WHERE job_id = ? AND username = ?
                """,
                (utc_now(), job_id, username),
            )
            return cursor.rowcount == 1

    def hide_owner_conversion(self, job_id: str, owner_user_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE conversions
                SET hidden = 1, updated_at = ?
                WHERE job_id = ? AND owner_user_id = ?
                """,
                (utc_now(), job_id, owner_user_id),
            )
            return cursor.rowcount == 1
