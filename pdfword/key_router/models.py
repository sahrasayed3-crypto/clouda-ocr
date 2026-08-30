from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .credential_refs import redact_secret_ref
from .enums import (
    AccountState,
    CircuitState,
    KeyRouterMode,
    Modality,
    PrivacyClassification,
    ReservationState,
)


@dataclass(frozen=True)
class ProviderAccount:
    account_id: str
    provider: str
    pool_id: str
    display_name: str
    secret_ref: str
    enabled: bool = False
    environment: str = "local"
    region: str = ""
    project_id: str = ""
    tenant_id: str = ""
    priority: int = 0
    weight: int = 1
    allows_free: bool = True
    allows_paid: bool = False
    allows_vision: bool = False
    allows_text: bool = True
    monthly_cost_limit: float = 0.0
    daily_cost_limit: float = 0.0
    daily_request_limit: int = 0
    daily_token_limit: int = 0
    max_concurrency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    version: int = 0
    account_state: AccountState = AccountState.ACTIVE
    organization_id: str = ""
    allowed_privacy: tuple[str, ...] = (
        PrivacyClassification.PUBLIC.value,
        PrivacyClassification.INTERNAL.value,
    )
    cooldown_until: str = ""

    def __repr__(self) -> str:
        return (
            "ProviderAccount("
            f"account_id={self.account_id!r}, provider={self.provider!r}, "
            f"pool_id={self.pool_id!r}, "
            f"secret_ref={redact_secret_ref(self.secret_ref)!r}, "
            f"enabled={self.enabled!r})"
        )


@dataclass(frozen=True)
class DispatchResult:
    text: str
    provider: str = ""
    provider_model_id: str = ""
    account_id: str = ""
    attempts: int = 0
    used_local_result: bool = False
    manual_review_required: bool = False
    final_error_type: str = ""


@dataclass(frozen=True)
class AccountRuntime:
    router_id: str
    provider: str
    pool_id: str
    state: AccountState = AccountState.DISABLED
    active_account_id: str = ""
    pending_account_id: str = ""
    account_epoch: int = 0
    switch_id: str = ""
    drain_deadline: str = ""
    closed_at: str = ""
    switch_started_at: str = ""
    cooldown_until: str = ""
    last_error_code: str = ""
    last_error_at: str = ""
    version: int = 0
    updated_at: str = ""


@dataclass(frozen=True)
class QuotaRuntime:
    provider: str
    pool_id: str
    account_id: str
    model: str
    window_type: str
    window_started_at: str
    request_limit: int = 0
    token_limit: int = 0
    cost_limit: float = 0.0
    reserved_requests: int = 0
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    reserved_cost: float = 0.0
    consumed_requests: int = 0
    consumed_input_tokens: int = 0
    consumed_output_tokens: int = 0
    consumed_cost: float = 0.0
    cooldown_until: str = ""
    circuit_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_success_at: str = ""
    last_failure_at: str = ""
    version: int = 0
    updated_at: str = ""


@dataclass(frozen=True)
class RequestReservation:
    reservation_id: str
    idempotency_key: str
    provider: str
    pool_id: str
    account_id: str
    model: str
    state: ReservationState
    reservation_token: str
    created_at: str
    expires_at: str
    reserved_requests: int = 1
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost: float = 0.0
    settled_at: str = ""
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    actual_cost: float = 0.0
    error_code: str = ""
    request_id: str = ""
    parent_request_id: str = ""
    attempt_id: str = ""
    provider_model_id: str = ""
    policy_id: str = ""
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    reserved_cost: float = 0.0
    request_count_consumed: bool = False
    token_cost_consumed: bool = False
    monetary_cost_consumed: bool = False
    dispatched_at: str = ""
    sent_at: str = ""
    fencing_token: int = 1
    uncertainty_reason: str = ""


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    operation_id: str
    task_type: str
    modality: Modality | str
    requested_model: str
    allowed_providers: tuple[str, ...] = ()
    free_only: bool = True
    max_cost: float = 0.0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost: float | None = None
    page_quality: float | None = None
    document_type: str = ""
    requires_vision: bool = False
    requires_structured_output: bool = False
    router_id: str = "clouda-local"
    pool_id: str = "primary"
    canonical_model_id: str = ""
    policy_id: str = ""
    mime_type: str = ""
    payload_bytes: int = 0
    image_width: int = 0
    image_height: int = 0
    image_count: int = 0
    privacy_classification: PrivacyClassification | str = PrivacyClassification.INTERNAL
    deadline_at: str = ""
    teacher_role: str = ""
    region_type: str = ""
    required_training_output_policy_status: str = ""

    @property
    def normalized_modality(self) -> str:
        value = (
            self.modality.value
            if isinstance(self.modality, Modality)
            else self.modality
        )
        return str(value).strip().lower()

    @property
    def normalized_privacy(self) -> str:
        value = (
            self.privacy_classification.value
            if isinstance(self.privacy_classification, PrivacyClassification)
            else self.privacy_classification
        )
        return str(value).strip().lower()


@dataclass(frozen=True)
class SelectionDecision:
    mode: KeyRouterMode
    selected_account: ProviderAccount | None
    provider: str
    pool_id: str
    model: str
    decision_reason: str
    eligible_count: int = 0
    rejected: tuple[dict[str, str], ...] = ()
    score: float = 0.0
    secret_fingerprint: str = ""
    endpoint_id: str = ""
    canonical_model_id: str = ""
    reason_codes: tuple[str, ...] = ()

    def sanitized(self) -> dict[str, Any]:
        account_id = self.selected_account.account_id if self.selected_account else ""
        return {
            "mode": self.mode.value,
            "provider": self.provider,
            "pool_id": self.pool_id,
            "account_id": account_id,
            "model": self.model,
            "decision_reason": self.decision_reason,
            "eligible_count": self.eligible_count,
            "rejected": list(self.rejected),
            "score": self.score,
            "secret_fingerprint": self.secret_fingerprint,
            "endpoint_id": self.endpoint_id,
            "canonical_model_id": self.canonical_model_id,
            "reason_codes": list(self.reason_codes),
        }


def metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)


def metadata_from_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
