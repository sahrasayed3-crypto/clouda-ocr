from __future__ import annotations

import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pdfword.database import Database, SCHEMA_VERSION
from pdfword import openrouter_client
from pdfword.key_router.capabilities import ModelEndpointCapability
from pdfword.key_router.audit import sanitize_metadata
from pdfword.key_router.config import KeyRouterConfig
from pdfword.key_router.credential_refs import SecretHandle
from pdfword.key_router.enums import (
    AccountState,
    CapabilityVerificationStatus,
    ErrorScope,
    KeyRouterMode,
    Modality,
    ProviderErrorType,
    ReservationState,
)
from pdfword.key_router.errors import normalize_provider_error
from pdfword.key_router.exceptions import (
    FencingTokenError,
    KeyRouterConfigurationError,
    KeyRouterUnavailable,
)
from pdfword.key_router.models import ProviderAccount, RequestContext
from pdfword.key_router.policies import choose_ocr_policy, get_policy
from pdfword.key_router.providers.base import (
    ProviderRequest,
    ProviderResponse,
    UnifiedProviderRequest,
)
from pdfword.key_router.providers.google_gemini import GoogleGeminiAdapter
from pdfword.key_router.providers.registry import get_provider_adapter
from pdfword.key_router.repository import KeyRouterRepository
from pdfword.key_router.retry import RetryCoordinator, RetryLimits
from pdfword.key_router.service import KeyRouter
from pdfword.worker_api import app


def phase2_account(account_id: str = "account-a", **overrides) -> ProviderAccount:
    values: dict[str, Any] = {
        "account_id": account_id,
        "provider": "openrouter",
        "pool_id": "primary",
        "display_name": account_id,
        "secret_ref": "env:PHASE2_FAKE_KEY",  # pragma: allowlist secret
        "enabled": True,
        "allows_free": True,
        "allows_paid": False,
        "allows_vision": True,
        "allows_text": True,
        "daily_request_limit": 10,
        "daily_token_limit": 10000,
        "max_concurrency": 5,
        "priority": 10,
        "account_state": AccountState.ACTIVE,
        "allowed_privacy": ("public", "internal"),
    }
    values.update(overrides)
    return ProviderAccount(**values)


def phase2_endpoint(
    endpoint_id: str = "endpoint-a", **overrides
) -> ModelEndpointCapability:
    values: dict[str, Any] = {
        "endpoint_id": endpoint_id,
        "canonical_model_id": "canonical-ocr",
        "provider": "openrouter",
        "provider_model_id": "provider/vision-model",
        "supports_text": True,
        "supports_vision_declared": True,
        "supports_vision_verified": True,
        "supports_base64": True,
        "supported_mime_types": ("image/png", "image/jpeg"),
        "max_payload_bytes": 1024,
        "supports_structured_json": True,
        "verified_status": CapabilityVerificationStatus.VERIFIED,
        "capability_source": "restricted-test-fixture",
        "enabled": True,
        "quality_score": 0.9,
    }
    values.update(overrides)
    return ModelEndpointCapability(**values)


def phase2_context(**overrides) -> RequestContext:
    values: dict[str, Any] = {
        "request_id": "request-phase2",
        "operation_id": "operation-phase2",
        "task_type": "OCR_ACCURATE_VISION",
        "policy_id": "OCR_ACCURATE_VISION",
        "modality": Modality.IMAGE,
        "requested_model": "canonical-ocr",
        "canonical_model_id": "canonical-ocr",
        "allowed_providers": ("openrouter",),
        "free_only": True,
        "estimated_input_tokens": 10,
        "estimated_output_tokens": 20,
        "estimated_cost": 0.0,
        "requires_vision": True,
        "mime_type": "image/png",
        "payload_bytes": 128,
        "image_width": 10,
        "image_height": 10,
        "image_count": 1,
    }
    values.update(overrides)
    return RequestContext(**values)


def configured_router(
    tmp_path,
    monkeypatch,
    *,
    mode: KeyRouterMode = KeyRouterMode.ACTIVE,
    config: KeyRouterConfig | None = None,
    accounts: tuple[ProviderAccount, ...] | None = None,
    endpoints: tuple[ModelEndpointCapability, ...] | None = None,
) -> KeyRouter:
    monkeypatch.setenv("PHASE2_FAKE_KEY", "synthetic-secret-value")
    database = Database(tmp_path / "phase2.sqlite3")
    router = KeyRouter(database, config or KeyRouterConfig(mode=mode))
    for account in accounts or (phase2_account(),):
        router.repository.upsert_account(account)
        router.repository.ensure_runtime(
            router_id=router.config.router_id,
            provider=account.provider,
            pool_id=account.pool_id,
            state=AccountState.ACTIVE,
        )
    for endpoint in endpoints or (phase2_endpoint(),):
        router.repository.upsert_endpoint(endpoint)
    return router


class FakeTransport:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.requests: list[ProviderRequest] = []

    def __call__(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def success_response(text: str = "cloud text") -> ProviderResponse:
    return ProviderResponse(
        status_code=200,
        headers={},
        json_body={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 11, "cost": 0.0},
        },
    )


def test_disabled_route_has_no_side_effects(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch, mode=KeyRouterMode.DISABLED)
    before = router.repository.recent_audit()
    decision = router.select_route(phase2_context())
    assert decision.decision_reason == "key_router_disabled"
    assert router.repository.recent_audit() == before


def test_shadow_records_decision_without_reservation_or_transport(
    tmp_path, monkeypatch
):
    router = configured_router(tmp_path, monkeypatch, mode=KeyRouterMode.SHADOW)
    decision = router.select_route(phase2_context())
    assert decision.selected_account is not None
    assert router.repository.list_reservations() == []
    assert router.repository.recent_audit()[0]["event_type"] == "shadow_route_decision"


def test_shadow_router_failure_does_not_break_legacy_request(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "empty-shadow.sqlite3"))
    monkeypatch.setenv("KEY_ROUTER_MODE", "shadow")
    monkeypatch.setattr(openrouter_client, "estimate_model_cost", lambda *_args: 0.0)
    monkeypatch.setattr(
        openrouter_client,
        "_post_with_retries",
        lambda payload, headers: {"choices": [{"message": {"content": "legacy"}}]},
    )
    assert (
        openrouter_client.openrouter_chat_text(
            "legacy-key", "model/free", "system", "user", max_tokens=5
        )
        == "legacy"
    )


def test_active_legacy_entry_uses_reservation_aware_dispatcher(monkeypatch):
    monkeypatch.setenv("KEY_ROUTER_MODE", "active")
    monkeypatch.setattr(openrouter_client, "estimate_model_cost", lambda *_args: 0.0)
    monkeypatch.setattr(
        "pdfword.key_router.facade.dispatch_active_legacy_request",
        lambda **_kwargs: "active-dispatch",
    )
    monkeypatch.setattr(
        openrouter_client,
        "_post_with_retries",
        lambda *_args, **_kwargs: pytest.fail("legacy transport must not run"),
    )
    assert (
        openrouter_client.openrouter_chat_text(
            "legacy-key", "canonical", "system", "user", max_tokens=5
        )
        == "active-dispatch"
    )


def test_active_selects_endpoint_then_account(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch)
    decision = router.select_route(phase2_context())
    assert decision.endpoint_id == "endpoint-a"
    assert decision.model == "provider/vision-model"
    assert decision.selected_account.account_id == "account-a"
    assert "endpoint_capabilities_satisfied" in decision.reason_codes


def test_multiple_accounts_are_deterministic(tmp_path, monkeypatch):
    accounts = (
        phase2_account("b", priority=5),
        phase2_account("a", priority=5),
    )
    router = configured_router(tmp_path, monkeypatch, accounts=accounts)
    assert router.select_route(phase2_context()).selected_account.account_id == "a"


def test_multiple_providers_select_by_endpoint_quality(tmp_path, monkeypatch):
    monkeypatch.setenv("SECOND_FAKE_KEY", "synthetic-secret-two")
    accounts = (
        phase2_account(priority=1),
        phase2_account(
            "google-a",
            provider="google_gemini",
            secret_ref="env:SECOND_FAKE_KEY",  # pragma: allowlist secret
            priority=1,
        ),
    )
    endpoints = (
        phase2_endpoint(quality_score=0.8),
        phase2_endpoint(
            "google-endpoint",
            provider="google_gemini",
            provider_model_id="gemini-test",
            quality_score=0.95,
        ),
    )
    router = configured_router(
        tmp_path, monkeypatch, accounts=accounts, endpoints=endpoints
    )
    decision = router.select_route(
        phase2_context(allowed_providers=("openrouter", "google_gemini"))
    )
    assert decision.provider == "google_gemini"


def test_vision_excludes_text_only_endpoint(tmp_path, monkeypatch):
    endpoints = (
        phase2_endpoint(
            "text-only",
            supports_vision_declared=False,
            supports_vision_verified=False,
            verified_status=CapabilityVerificationStatus.VERIFIED,
        ),
    )
    router = configured_router(tmp_path, monkeypatch, endpoints=endpoints)
    with pytest.raises(KeyRouterUnavailable, match="no_eligible_endpoint"):
        router.select_route(phase2_context())


def test_declared_only_vision_rejected_by_default(tmp_path, monkeypatch):
    endpoint = phase2_endpoint(
        supports_vision_verified=False,
        verified_status=CapabilityVerificationStatus.DECLARED,
    )
    router = configured_router(tmp_path, monkeypatch, endpoints=(endpoint,))
    with pytest.raises(KeyRouterUnavailable):
        router.select_route(phase2_context())


def test_declared_only_vision_can_be_explicitly_allowed(tmp_path, monkeypatch):
    config = KeyRouterConfig(
        mode=KeyRouterMode.ACTIVE,
        require_verified_vision=False,
        allow_declared_only_capabilities=True,
    )
    endpoint = phase2_endpoint(
        supports_vision_verified=False,
        verified_status=CapabilityVerificationStatus.DECLARED,
    )
    router = configured_router(
        tmp_path, monkeypatch, config=config, endpoints=(endpoint,)
    )
    assert router.select_route(phase2_context()).endpoint_id == "endpoint-a"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"mime_type": "image/gif"}, "unsupported_mime_type"),
        ({"payload_bytes": 2048}, "payload_too_large"),
        ({"image_count": 2}, "too_many_images"),
    ],
)
def test_payload_constraints_fail_before_dispatch(
    tmp_path, monkeypatch, overrides, reason
):
    router = configured_router(tmp_path, monkeypatch)
    with pytest.raises(KeyRouterUnavailable):
        router.select_route(phase2_context(**overrides))
    assert any(
        item["reason"] == reason
        for item in router.endpoint_selector.select(
            context=phase2_context(**overrides),
            policy=get_policy("OCR_ACCURATE_VISION"),
            endpoints=router.repository.list_endpoints(enabled_only=True),
            require_verified_vision=True,
            allow_declared_only=False,
            open_circuits=set(),
        )[1]
    )


def test_model_mapping_is_separate_and_provider_specific():
    endpoint = phase2_endpoint()
    assert endpoint.mapping.canonical_model_id == "canonical-ocr"
    assert endpoint.mapping.provider_model_id == "provider/vision-model"


@pytest.mark.parametrize(
    ("status", "payload", "expected", "scope"),
    [
        (
            401,
            {"error": {"code": "invalid_api_key"}},
            ProviderErrorType.AUTH_FAILED,
            ErrorScope.KEY,
        ),
        (
            403,
            {"error": {"message": "denied"}},
            ProviderErrorType.PERMISSION_DENIED,
            ErrorScope.ACCOUNT,
        ),
        (
            429,
            {"error": {"code": "insufficient_quota"}},
            ProviderErrorType.QUOTA_EXHAUSTED,
            ErrorScope.ACCOUNT,
        ),
        (
            429,
            {"error": {"scope": "model", "message": "rate limit for model"}},
            ProviderErrorType.RATE_LIMITED_TEMPORARY,
            ErrorScope.MODEL,
        ),
        (
            402,
            {"error": {"message": "billing required"}},
            ProviderErrorType.BILLING_REQUIRED,
            ErrorScope.ORGANIZATION,
        ),
        (
            404,
            {"error": {"message": "model missing"}},
            ProviderErrorType.MODEL_NOT_FOUND,
            ErrorScope.MODEL,
        ),
        (
            503,
            {"error": {"message": "overloaded"}},
            ProviderErrorType.PROVIDER_OVERLOADED,
            ErrorScope.PROVIDER,
        ),
    ],
)
def test_provider_error_payloads_normalize(status, payload, expected, scope):
    error = normalize_provider_error(
        provider="fake", http_status=status, payload=payload, request_sent=True
    )
    assert error.error_type is expected
    assert error.scope is scope


def test_retry_after_is_parsed_with_fake_clock():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    error = normalize_provider_error(
        provider="fake",
        http_status=429,
        payload={"error": {"message": "temporary rate limit"}},
        headers={"Retry-After": "12"},
        now=now,
    )
    assert error.retry_after_seconds == 12


def test_error_message_redacts_token_and_base64():
    error = normalize_provider_error(
        provider="fake",
        http_status=400,
        payload={
            "error": {
                "message": "Bearer top-secret data:image/png;base64,AAAA token=hidden"
            }
        },
    )
    assert "top-secret" not in error.sanitized_message
    assert "AAAA" not in error.sanitized_message
    assert "hidden" not in error.sanitized_message


def test_audit_redacts_base64_document_text_and_signed_urls():
    sanitized = sanitize_metadata(
        {
            "image_b64": "AAAA",
            "document_text": "private OCR text",
            "url": "https://example.invalid/file?token=signed-value&safe=1",
            "message": "data:image/png;base64,BBBB",
        }
    )
    serialized = str(sanitized)
    assert "AAAA" not in serialized
    assert "private OCR text" not in serialized
    assert "signed-value" not in serialized
    assert "BBBB" not in serialized


def test_auth_failure_quarantines_only_selected_account(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch)
    transport = FakeTransport(
        ProviderResponse(401, {}, {"error": {"code": "invalid_api_key"}})
    )
    result = router.dispatch(
        phase2_context(),
        UnifiedProviderRequest(model="ignored", image_b64="AAAA"),
        transport=transport,
        local_result="local",
        local_quality=95,
        sleeper=lambda _seconds: None,
    )
    assert result.used_local_result
    account = router.repository.get_account("account-a")
    assert account.account_state is AccountState.QUARANTINED


def test_model_scoped_429_does_not_disable_account(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch)
    transport = FakeTransport(
        ProviderResponse(
            429,
            {"Retry-After": "3"},
            {"error": {"scope": "model", "message": "model rate limit"}},
        )
    )
    router.dispatch(
        phase2_context(),
        UnifiedProviderRequest(model="ignored", image_b64="AAAA"),
        transport=transport,
        local_result="local",
        sleeper=lambda _seconds: None,
    )
    account = router.repository.get_account("account-a")
    assert account.enabled
    assert account.account_state is AccountState.ACTIVE


def test_account_scoped_429_does_not_cycle_same_provider_accounts(
    tmp_path, monkeypatch
):
    router = configured_router(
        tmp_path,
        monkeypatch,
        accounts=(phase2_account("first"), phase2_account("second", priority=5)),
    )
    transport = FakeTransport(
        ProviderResponse(429, {}, {"error": {"message": "temporary rate limit"}}),
        success_response("must not run"),
    )
    result = router.dispatch(
        phase2_context(),
        UnifiedProviderRequest(model="ignored", image_b64="AAAA"),
        transport=transport,
        local_result="local",
        sleeper=lambda _seconds: None,
    )
    assert result.used_local_result
    assert len(transport.requests) == 1


def test_provider_503_opens_provider_and_model_circuits(tmp_path, monkeypatch):
    config = KeyRouterConfig(mode=KeyRouterMode.ACTIVE, failure_threshold=1)
    router = configured_router(tmp_path, monkeypatch, config=config)
    transport = FakeTransport(ProviderResponse(503, {}, {"error": {"message": "busy"}}))
    router.dispatch(
        phase2_context(),
        UnifiedProviderRequest(model="ignored", image_b64="AAAA"),
        transport=transport,
        local_result="local",
        sleeper=lambda _seconds: None,
    )
    levels = {
        item["level"]
        for item in router.repository.list_circuits()
        if item["state"] == "OPEN"
    }
    assert {"provider", "model"} <= levels


def test_half_open_probe_limit_is_atomic(tmp_path):
    repository = KeyRouterRepository(Database(tmp_path / "circuit.sqlite3"))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    circuit = repository.record_circuit_failure(
        level="provider",
        provider="fake",
        failure_threshold=1,
        open_seconds=1,
        now=start,
    )

    def probe(_index: int) -> bool:
        return repository.acquire_half_open_probe(
            circuit["circuit_id"], probe_limit=1, now=start + timedelta(seconds=2)
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(probe, range(4)))
    assert results.count(True) == 1


def test_account_circuit_level_and_rolling_window(tmp_path):
    repository = KeyRouterRepository(Database(tmp_path / "account-circuit.sqlite3"))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = repository.record_circuit_failure(
        level="account",
        provider="fake",
        account_id="account-a",
        failure_threshold=2,
        open_seconds=10,
        failure_window_seconds=5,
        now=start,
    )
    assert first["state"] == "CLOSED"
    reset_window = repository.record_circuit_failure(
        level="account",
        provider="fake",
        account_id="account-a",
        failure_threshold=2,
        open_seconds=10,
        failure_window_seconds=5,
        now=start + timedelta(seconds=6),
    )
    assert reset_window["failure_count"] == 1
    assert reset_window["state"] == "CLOSED"


def test_local_validation_release_success_and_timeouts(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch)
    decision = router.select_route(phase2_context())
    reservation = router.reserve(phase2_context(), decision, idempotency_key="release")
    assert (
        router.release(reservation, reason="local_validation").state
        is ReservationState.RELEASED
    )

    success = router.reserve(phase2_context(), decision, idempotency_key="success")
    success = router.mark_dispatching(success)
    success = router.mark_sent(success)
    assert (
        router.settle_success(
            success, actual_input_tokens=1, actual_output_tokens=2, actual_cost=0
        ).state
        is ReservationState.SETTLED_SUCCESS
    )

    before_send = router.reserve(phase2_context(), decision, idempotency_key="before")
    before_send = router.mark_dispatching(before_send)
    assert (
        router.release(before_send, reason="TIMEOUT_BEFORE_SEND").state
        is ReservationState.RELEASED
    )

    after_send = router.reserve(phase2_context(), decision, idempotency_key="after")
    after_send = router.mark_dispatching(after_send)
    assert (
        router.mark_uncertain(after_send, error_code="TIMEOUT_AFTER_SEND").state
        is ReservationState.UNCERTAIN
    )


def test_reconciliation_expires_reserved_and_marks_stale_dispatch_uncertain(
    tmp_path, monkeypatch
):
    router = configured_router(tmp_path, monkeypatch)
    decision = router.select_route(phase2_context())
    reserved = router.reserve(phase2_context(), decision, idempotency_key="expired")
    dispatching = router.reserve(phase2_context(), decision, idempotency_key="stale")
    dispatching = router.mark_dispatching(dispatching)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    decisions = router.repository.reconcile_reservations(
        now=future, stale_after_seconds=1
    )
    states = {
        item.reservation_id: item.state
        for item in router.repository.list_reservations()
    }
    assert states[reserved.reservation_id] is ReservationState.EXPIRED
    assert states[dispatching.reservation_id] is ReservationState.UNCERTAIN
    assert len(decisions) == 2


def test_double_settlement_and_stale_fencing_are_prevented(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch)
    decision = router.select_route(phase2_context())
    reservation = router.reserve(phase2_context(), decision, idempotency_key="fence")
    settled = router.settle_success(
        reservation, actual_input_tokens=0, actual_output_tokens=0, actual_cost=0
    )
    assert settled.state is ReservationState.SETTLED_SUCCESS
    with pytest.raises(FencingTokenError):
        router.settle_success(
            reservation, actual_input_tokens=0, actual_output_tokens=0, actual_cost=0
        )


def test_atomic_reservation_prevents_multi_worker_oversubscription(
    tmp_path, monkeypatch
):
    router = configured_router(
        tmp_path,
        monkeypatch,
        accounts=(phase2_account(daily_request_limit=1),),
    )
    decision = router.select_route(phase2_context())

    def reserve(index: int) -> str:
        try:
            router.reserve(
                phase2_context(), decision, idempotency_key=f"worker-{index}"
            )
            return "ok"
        except Exception as exc:
            return exc.__class__.__name__

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(reserve, range(4)))
    assert results.count("ok") == 1


def test_retry_budget_and_exclusion_prevent_loops():
    coordinator = RetryCoordinator(
        RetryLimits(maximum_attempts=2, retry_budget=2, maximum_account_failovers=1)
    )
    state = coordinator.start("parent", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    coordinator.begin_attempt(state, provider="p", model="m", account_id="a")
    error = normalize_provider_error(
        provider="p", provider_account_id="a", http_status=401
    )
    coordinator.record_failure(
        state, provider="p", model="m", account_id="a", error=error
    )
    assert ("p", "m", "a") in state.excluded_candidates
    with pytest.raises(ValueError):
        coordinator.begin_attempt(state, provider="p", model="m", account_id="a")


def test_attempt_lineage_and_account_failover(tmp_path, monkeypatch):
    accounts = (
        phase2_account("first", priority=20),
        phase2_account("second", priority=10),
    )
    router = configured_router(tmp_path, monkeypatch, accounts=accounts)
    transport = FakeTransport(
        ProviderResponse(401, {}, {"error": {"code": "invalid_api_key"}}),
        success_response("second worked"),
    )
    result = router.dispatch(
        phase2_context(),
        UnifiedProviderRequest(model="ignored", image_b64="AAAA"),
        transport=transport,
        sleeper=lambda _seconds: None,
    )
    assert result.text == "second worked"
    attempts = router.repository.list_attempts("request-phase2")
    assert [item["attempt_number"] for item in attempts] == [1, 2]
    assert attempts[0]["account_id"] != attempts[1]["account_id"]


def test_exhausted_routes_preserve_best_local_result(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch)
    transport = FakeTransport(ProviderResponse(503, {}, {"error": {"message": "busy"}}))
    result = router.dispatch(
        phase2_context(),
        UnifiedProviderRequest(model="ignored", image_b64="AAAA"),
        transport=transport,
        local_result="best local OCR",
        local_quality=91,
        sleeper=lambda _seconds: None,
    )
    assert result.text == "best local OCR"
    assert result.used_local_result
    assert not result.manual_review_required


def test_provider_400_consumes_request_only(tmp_path, monkeypatch):
    router = configured_router(tmp_path, monkeypatch)
    transport = FakeTransport(
        ProviderResponse(400, {}, {"error": {"message": "invalid request"}})
    )
    router.dispatch(
        phase2_context(),
        UnifiedProviderRequest(model="ignored", image_b64="AAAA"),
        transport=transport,
        local_result="local",
        sleeper=lambda _seconds: None,
    )
    reservation = router.repository.list_reservations()[0]
    assert reservation.request_count_consumed
    assert not reservation.token_cost_consumed
    assert not reservation.monetary_cost_consumed


def test_openai_and_google_adapters_translate_and_redact():
    openai = get_provider_adapter("openrouter")
    request = openai.build_request(
        UnifiedProviderRequest(
            model="m", user_text="text", image_b64="AAAA", structured_json=True
        ),
        secret_value="synthetic-secret",  # pragma: allowlist secret
    )
    sanitized = openai.redact_request_for_logging(request)
    assert request.json_body["messages"][0]["content"][1]["image_url"]["url"].endswith(
        "AAAA"
    )
    assert "synthetic-secret" not in str(sanitized)
    assert "AAAA" not in str(sanitized)

    definition = get_provider_adapter("google_gemini").definition
    google = GoogleGeminiAdapter(definition)
    translated = google.build_request(
        UnifiedProviderRequest(model="gemini", user_text="text", image_b64="BBBB"),
        secret_value="synthetic-google",  # pragma: allowlist secret
    )
    assert "inlineData" in str(translated.json_body)
    assert "synthetic-google" not in str(google.redact_request_for_logging(translated))


def test_privacy_policy_blocks_disallowed_provider(tmp_path, monkeypatch):
    router = configured_router(
        tmp_path,
        monkeypatch,
        accounts=(phase2_account(allowed_privacy=("public",)),),
    )
    with pytest.raises(KeyRouterUnavailable):
        router.select_route(phase2_context(privacy_classification="restricted"))


def test_ocr_policy_does_not_send_born_digital_text_to_vision():
    assert (
        choose_ocr_policy(
            born_digital=True,
            local_text_usable=True,
            page_quality=99,
        ).policy_id
        == "DIGITAL_TEXT_VALIDATE"
    )
    assert choose_ocr_policy(
        born_digital=False,
        local_text_usable=False,
        page_quality=50,
    ).vision_required


def test_migration_keeps_existing_data_and_adds_phase2_schema(tmp_path):
    path = tmp_path / "migration.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE schema_meta(version INTEGER NOT NULL);
        INSERT INTO schema_meta(version) VALUES (4);
        CREATE TABLE conversions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE provider_accounts(
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
        INSERT INTO provider_accounts(
            account_id, provider, pool_id, display_name, secret_ref,
            created_at, updated_at
        ) VALUES (
            'old-account', 'openrouter', 'primary', 'Old',
            'OPENROUTER_API_KEY_PRIMARY', 'before', 'before'
        );
        """)
    connection.commit()
    connection.close()
    upgraded = Database(path)
    with upgraded.connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = connection.execute("SELECT version FROM schema_meta").fetchone()[0]
        accounts = connection.execute(
            "SELECT account_id, account_state FROM provider_accounts"
        ).fetchall()
    assert {
        "capability_endpoints",
        "key_router_circuits",
        "key_router_errors",
        "key_router_attempts",
        "teacher_roles",
        "teacher_jobs",
    } <= tables
    assert version == SCHEMA_VERSION
    assert SCHEMA_VERSION >= 6
    assert [tuple(row) for row in accounts] == [("old-account", "ACTIVE")]


def test_internal_operational_api_is_authenticated_and_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "worker-synthetic")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setenv("KEY_ROUTER_MODE", "disabled")
    monkeypatch.setenv("PHASE2_FAKE_KEY", "never-return-this")
    router = KeyRouter(Database(tmp_path / "api.sqlite3"))
    router.repository.upsert_account(phase2_account())
    router.repository.upsert_endpoint(phase2_endpoint())
    client = TestClient(app)
    for path in (
        "/internal/key-router/status",
        "/internal/key-router/accounts",
        "/internal/key-router/capabilities",
        "/internal/key-router/reservations",
        "/internal/key-router/audit",
    ):
        assert client.get(path).status_code == 401
        response = client.get(path, headers={"X-Worker-API-Key": "worker-synthetic"})
        assert response.status_code == 200
        assert "never-return-this" not in response.text
        assert "secret_ref" not in response.text


def test_admin_actions_are_validated_protected_and_audited(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "worker-synthetic")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "admin.sqlite3"))
    monkeypatch.setenv("KEY_ROUTER_MODE", "disabled")
    repository = KeyRouterRepository(Database(tmp_path / "admin.sqlite3"))
    repository.upsert_account(phase2_account())
    client = TestClient(app)
    headers = {"X-Worker-API-Key": "worker-synthetic"}
    assert (
        client.post(
            "/internal/key-router/accounts/invalid$id/disable", headers=headers
        ).status_code
        == 400
    )
    response = client.post(
        "/internal/key-router/accounts/account-a/quarantine", headers=headers
    )
    assert response.status_code == 200
    assert response.json()["state"] == "QUARANTINED"
    assert repository.recent_audit()[0]["event_type"] == "account_admin_action"


def test_secret_references_and_repr_never_expose_values(caplog):
    caplog.set_level(logging.DEBUG)
    handle = SecretHandle("env:PHASE2_FAKE_KEY", "synthetic-sensitive-value")
    logging.getLogger("phase2").debug("%r", handle)
    account = phase2_account()
    assert "synthetic-sensitive-value" not in repr(handle)
    assert "synthetic-sensitive-value" not in caplog.text
    assert "PHASE2_FAKE_KEY" not in repr(account)


def test_active_validation_fails_closed_but_disabled_tolerates_optional_bad_env(
    tmp_path, monkeypatch
):
    active = KeyRouter(
        Database(tmp_path / "active.sqlite3"),
        KeyRouterConfig(mode=KeyRouterMode.ACTIVE),
    )
    with pytest.raises(KeyRouterConfigurationError):
        active.validate_active_configuration()

    monkeypatch.setenv("KEY_ROUTER_MODE", "disabled")
    monkeypatch.setenv("KEY_ROUTER_MAX_ATTEMPTS", "not-an-integer")
    assert KeyRouterConfig.from_env().mode is KeyRouterMode.DISABLED


@pytest.mark.skipif(
    os.getenv("KEY_ROUTER_LIVE_TESTS_ENABLED", "false").lower() != "true",
    reason="live provider tests require explicit opt-in and credentials",
)
def test_live_provider_requests_require_explicit_opt_in():
    pytest.skip("No live provider request is implemented in this test suite")
