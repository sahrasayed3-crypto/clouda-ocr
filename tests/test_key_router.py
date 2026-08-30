from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import TypedDict, Unpack

import pytest
from fastapi.testclient import TestClient

from pdfword import openrouter_client
from pdfword.database import Database
from pdfword.key_router.circuit_breaker import (
    CircuitBreaker,
    classify_error,
    parse_retry_after,
)
from pdfword.key_router.config import KeyRouterConfig
from pdfword.key_router.enums import (
    AccountState,
    CircuitState,
    ErrorCategory,
    KeyRouterMode,
    Modality,
    ReservationState,
)
from pdfword.key_router.exceptions import (
    FencingTokenError,
    InvalidStateTransition,
    KeyRouterUnavailable,
    QuotaExceeded,
)
from pdfword.key_router.models import ProviderAccount, RequestContext
from pdfword.key_router.providers.registry import PROVIDER_REGISTRY
from pdfword.key_router.credential_refs import SecretHandle
from pdfword.key_router.service import KeyRouter
from pdfword.key_router.state_machine import validate_transition
from pdfword.worker_api import app


class ContextOverrides(TypedDict, total=False):
    allowed_providers: tuple[str, ...]
    estimated_cost: float
    free_only: bool
    requires_vision: bool


class AccountOverrides(TypedDict, total=False):
    account_id: str
    allows_paid: bool
    allows_vision: bool
    daily_cost_limit: float
    daily_request_limit: int
    enabled: bool
    max_concurrency: int
    priority: int
    provider: str
    secret_ref: str
    weight: int


def context(**overrides: Unpack[ContextOverrides]) -> RequestContext:
    defaults = RequestContext(
        request_id="req-1",
        operation_id="op-1",
        task_type="ocr",
        modality=Modality.TEXT,
        requested_model="model/free",
        allowed_providers=("openrouter",),
        free_only=True,
        estimated_input_tokens=10,
        estimated_output_tokens=20,
        estimated_cost=0.0,
        requires_vision=False,
    )
    return replace(defaults, **overrides)


def account(**overrides: Unpack[AccountOverrides]) -> ProviderAccount:
    defaults = ProviderAccount(
        account_id="openrouter-primary",
        provider="openrouter",
        pool_id="primary",
        display_name="OpenRouter Primary",
        secret_ref="OPENROUTER_API_KEY_PRIMARY",  # pragma: allowlist secret
        enabled=True,
        allows_free=True,
        allows_paid=False,
        allows_vision=True,
        allows_text=True,
        daily_request_limit=10,
        daily_token_limit=1000,
        daily_cost_limit=0.0,
        max_concurrency=2,
        priority=10,
        weight=1,
    )
    return replace(defaults, **overrides)


def router(tmp_path, monkeypatch, mode: KeyRouterMode) -> KeyRouter:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "key-router.sqlite3"))
    monkeypatch.setenv("KEY_ROUTER_MODE", mode.value)
    return KeyRouter(
        Database(tmp_path / "key-router.sqlite3"), KeyRouterConfig(mode=mode)
    )


def add_active_account(router: KeyRouter, provider_account: ProviderAccount) -> None:
    router.repository.upsert_account(provider_account)
    router.repository.ensure_runtime(
        router_id=router.config.router_id,
        provider=provider_account.provider,
        pool_id=provider_account.pool_id,
        state=AccountState.ACTIVE,
    )


def test_disabled_mode_does_not_select_or_write(tmp_path, monkeypatch):
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.DISABLED)
    decision = key_router.select_account(context())
    assert decision.selected_account is None
    assert decision.decision_reason == "key_router_disabled"
    assert key_router.repository.recent_audit() == []


def test_shadow_selects_and_audits_without_using_selected_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_PRIMARY", "shadow-secret-value")
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.SHADOW)
    add_active_account(key_router, account())

    decision = key_router.select_account(context())

    assert decision.selected_account is not None
    assert decision.secret_fingerprint
    events = key_router.repository.recent_audit()
    assert events[0]["event_type"] == "shadow_selection"
    assert "shadow-secret-value" not in str(events)


def test_active_rejects_missing_configuration(tmp_path, monkeypatch):
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.ACTIVE)
    with pytest.raises(KeyRouterUnavailable):
        key_router.select_account(context())


def test_secret_handle_and_logs_do_not_expose_value(caplog):
    caplog.set_level(logging.INFO)
    handle = SecretHandle("OPENROUTER_API_KEY_PRIMARY", "dummy-sensitive-value")
    logging.getLogger("test").info("handle=%r", handle)
    assert "dummy-sensitive-value" not in repr(handle)
    assert "dummy-sensitive-value" not in caplog.text
    assert handle.fingerprint != "cret"


def test_secret_value_is_not_stored_in_database(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_PRIMARY", "database-secret")
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.ACTIVE)
    add_active_account(key_router, account())
    with key_router.database.connect() as connection:
        dump = "\n".join(
            "".join(str(value) for value in row)
            for row in connection.execute("SELECT * FROM provider_accounts")
        )
    assert "database-secret" not in dump
    assert "OPENROUTER_API_KEY_PRIMARY" in dump


def test_database_upgrade_adds_key_router_tables_without_losing_existing_rows(tmp_path):
    database_path = tmp_path / "upgrade.sqlite3"
    database = Database(database_path)
    database.login("alice")

    upgraded = Database(database_path)
    with upgraded.connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        user_count = connection.execute(
            "SELECT COUNT(*) AS count FROM users"
        ).fetchone()

    assert {
        "provider_accounts",
        "account_runtime",
        "quota_runtime",
        "request_reservations",
        "key_router_audit",
    } <= tables
    assert user_count["count"] == 1


def test_selection_excludes_unsafe_or_ineligible_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_PRIMARY", "secret")
    monkeypatch.setenv("TOGETHER_API_KEY_PRIMARY", "secret2")
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.ACTIVE)
    candidates = [
        account(account_id="disabled", enabled=False, priority=100),
        account(account_id="missing-secret", secret_ref="MISSING_SECRET", priority=99),
        account(account_id="no-vision", allows_vision=False, priority=98),
        account(
            account_id="paid-blocked",
            allows_paid=False,
            daily_cost_limit=10,
            priority=97,
        ),
        account(
            account_id="winner",
            provider="together",
            secret_ref="TOGETHER_API_KEY_PRIMARY",  # pragma: allowlist secret
            allows_paid=True,
            daily_cost_limit=10,
            priority=1,
        ),
    ]
    for candidate in candidates:
        key_router.repository.upsert_account(candidate)
        key_router.repository.ensure_runtime(
            router_id=key_router.config.router_id,
            provider=candidate.provider,
            pool_id=candidate.pool_id,
            state=AccountState.ACTIVE,
        )

    decision = key_router.select_account(
        context(
            allowed_providers=("openrouter", "together"),
            requires_vision=True,
            free_only=False,
            estimated_cost=1.0,
        )
    )

    assert decision.selected_account is not None
    assert decision.selected_account.account_id == "winner"
    reasons = {item["account_id"]: item["reason"] for item in decision.rejected}
    assert reasons["disabled"] == "account_disabled"
    assert reasons["missing-secret"] == "secret_missing"  # pragma: allowlist secret
    assert reasons["no-vision"] == "vision_not_allowed"
    assert reasons["paid-blocked"] == "paid_not_allowed"


def test_selection_prefers_free_priority_weight_and_is_deterministic(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY_PRIMARY", "secret")
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.ACTIVE)
    for candidate in [
        account(account_id="b", priority=1, weight=1),
        account(account_id="a", priority=1, weight=1),
        account(account_id="paid", allows_paid=True, daily_cost_limit=10, priority=0),
    ]:
        key_router.repository.upsert_account(candidate)
        key_router.repository.ensure_runtime(
            router_id=key_router.config.router_id,
            provider=candidate.provider,
            pool_id=candidate.pool_id,
            state=AccountState.ACTIVE,
        )

    choices = [
        key_router.select_account(context()).selected_account.account_id
        for _ in range(5)
    ]
    assert choices == ["a"] * 5


def test_quota_reservation_settlement_and_uncertain_fencing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_PRIMARY", "secret")
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.ACTIVE)
    add_active_account(
        key_router,
        account(allows_paid=True, daily_cost_limit=2.0, daily_request_limit=2),
    )
    req = context(free_only=False, estimated_cost=0.5)
    decision = key_router.select_account(req)
    reservation = key_router.reserve(req, decision, idempotency_key="idem-1")
    sent = key_router.mark_sent(reservation)
    assert sent.state is ReservationState.SENT

    with pytest.raises(FencingTokenError):
        key_router.repository.settle_reservation(
            reservation_id=reservation.reservation_id,
            reservation_token="stale",
            state=ReservationState.SUCCEEDED,
        )

    uncertain = key_router.mark_uncertain(sent, error_code="timeout")
    assert uncertain.state is ReservationState.UNCERTAIN
    assert (
        key_router.settle_failure(uncertain, error_code="timeout").state
        is ReservationState.FAILED
    )

    second = key_router.reserve(req, decision, idempotency_key="idem-2")
    assert (
        key_router.settle_success(
            second,
            actual_input_tokens=8,
            actual_output_tokens=10,
            actual_cost=0.4,
        ).state
        is ReservationState.SUCCEEDED
    )


def test_quota_prevents_parallel_oversubscription(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_PRIMARY", "secret")
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.ACTIVE)
    add_active_account(
        key_router,
        account(daily_request_limit=1, max_concurrency=5),
    )
    req = context()
    decision = key_router.select_account(req)

    def reserve_once(index: int) -> str:
        try:
            key_router.reserve(req, decision, idempotency_key=f"parallel-{index}")
            return "reserved"
        except QuotaExceeded:
            return "quota_exceeded"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve_once, [1, 2]))

    assert sorted(results) == ["quota_exceeded", "reserved"]


def test_account_switching_and_stale_epoch_are_fenced(tmp_path, monkeypatch):
    key_router = router(tmp_path, monkeypatch, KeyRouterMode.ACTIVE)
    provider = "openrouter"
    pool_id = "primary"
    key_router.repository.ensure_runtime(
        router_id=key_router.config.router_id,
        provider=provider,
        pool_id=pool_id,
        state=AccountState.ACTIVE,
    )

    draining = key_router.start_drain(provider=provider, pool_id=pool_id)
    assert draining.state is AccountState.DRAINING
    closed = key_router.close_drained(provider=provider, pool_id=pool_id)
    assert closed.state is AccountState.CLOSED
    switching = key_router.begin_switch(
        provider=provider, pool_id=pool_id, pending_account_id="new-account"
    )
    assert switching.state is AccountState.SWITCHING

    with pytest.raises(QuotaExceeded):
        key_router.complete_switch(
            provider=provider,
            pool_id=pool_id,
            switch_id="stale",
            account_epoch=switching.account_epoch,
        )
    active = key_router.complete_switch(
        provider=provider,
        pool_id=pool_id,
        switch_id=switching.switch_id,
        account_epoch=switching.account_epoch,
    )
    assert active.state is AccountState.ACTIVE
    assert active.active_account_id == "new-account"


def test_invalid_account_state_transition_is_rejected():
    with pytest.raises(InvalidStateTransition):
        validate_transition(AccountState.ACTIVE, AccountState.SWITCHING)


def test_circuit_breaker_and_429_parsing():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=30)
    state, failures = breaker.after_failure(
        state=CircuitState.CLOSED,
        consecutive_failures=0,
        error_category=ErrorCategory.NETWORK,
    )
    assert state is CircuitState.CLOSED
    state, failures = breaker.after_failure(
        state=state,
        consecutive_failures=failures,
        error_category=ErrorCategory.RATE_LIMIT,
    )
    assert state is CircuitState.OPEN
    assert breaker.after_failure(
        state=state,
        consecutive_failures=failures,
        error_category=ErrorCategory.INVALID_REQUEST,
    ) == (CircuitState.OPEN, failures)
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert (
        breaker.state_for_time(state=CircuitState.OPEN, cooldown_until=past)
        is CircuitState.HALF_OPEN
    )
    assert breaker.after_success(state=CircuitState.HALF_OPEN) == (
        CircuitState.CLOSED,
        0,
    )
    retry_at = parse_retry_after("2", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert retry_at == datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    assert classify_error(429) is ErrorCategory.RATE_LIMIT


def test_provider_registry_contains_required_configuration_ready_providers():
    required = {
        "openrouter",
        "google_gemini",
        "alibaba_dashscope",
        "requesty",
        "huggingface",
        "nvidia_nim",
        "sambanova",
        "github_models",
        "cloudflare_workers_ai",
        "siliconflow",
        "deepinfra",
        "fireworks",
        "together",
        "nebius",
        "replicate",
        "novita",
    }
    assert required <= set(PROVIDER_REGISTRY)


def test_openrouter_disabled_path_uses_existing_key_without_network(monkeypatch):
    monkeypatch.setenv("KEY_ROUTER_MODE", "disabled")
    calls = []

    def fake_post(payload, headers):
        calls.append((payload, headers))
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(openrouter_client, "_post_with_retries", fake_post)
    assert (
        openrouter_client.openrouter_chat_text(
            "legacy-key", "model", "system", "user", max_tokens=5
        )
        == "done"
    )
    assert calls[0][1]["Authorization"] == "Bearer legacy-key"


def test_openrouter_shadow_path_audits_but_uses_existing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "shadow.sqlite3"))
    monkeypatch.setenv("KEY_ROUTER_MODE", "shadow")
    monkeypatch.setenv("OPENROUTER_API_KEY_PRIMARY", "selected-secret")
    monkeypatch.setattr(openrouter_client, "estimate_model_cost", lambda *_args: 0.0)
    key_router = KeyRouter(
        Database(tmp_path / "shadow.sqlite3"),
        KeyRouterConfig(mode=KeyRouterMode.SHADOW),
    )
    add_active_account(key_router, account())
    calls = []

    def fake_post(payload, headers):
        calls.append((payload, headers))
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(openrouter_client, "_post_with_retries", fake_post)
    assert (
        openrouter_client.openrouter_chat_text(
            "legacy-key", "model/free", "system", "user", max_tokens=5
        )
        == "done"
    )
    assert calls[0][1]["Authorization"] == "Bearer legacy-key"
    assert "selected-secret" not in str(key_router.repository.recent_audit())


def test_internal_key_router_status_is_authenticated_and_sanitized(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WORKER_API_KEY", "worker-secret")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setenv("KEY_ROUTER_MODE", "disabled")
    client = TestClient(app)

    assert client.get("/internal/key-router/status").status_code == 401
    response = client.get(
        "/internal/key-router/status",
        headers={"X-Worker-API-Key": "worker-secret"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "disabled"
    assert "secret_ref" not in response.text
    assert "worker-secret" not in response.text
