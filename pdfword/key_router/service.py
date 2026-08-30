from __future__ import annotations

import uuid
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable

from ..database import Database, utc_now
from .audit import audit_event, sanitize_metadata
from .config import KeyRouterConfig
from .enums import (
    AccountState,
    ErrorScope,
    KeyRouterMode,
    ProviderErrorType,
    ReservationState,
)
from .errors import NormalizedProviderError
from .exceptions import (
    KeyRouterConfigurationError,
    KeyRouterUnavailable,
    ProviderDispatchError,
    QuotaExceeded,
)
from .health import provider_capability_snapshot
from .models import (
    AccountRuntime,
    DispatchResult,
    RequestContext,
    RequestReservation,
    SelectionDecision,
)
from .policies import get_policy
from .providers.base import Transport, UnifiedProviderRequest
from .providers.registry import get_provider_adapter
from .quota import quota_limits_for
from .repository import KeyRouterRepository
from .credential_refs import EnvSecretResolver
from .retry import RetryCoordinator, RetryLimits
from .selector import AccountSelector, EndpointSelector
from .state_machine import validate_transition


class KeyRouter:
    def __init__(
        self,
        database: Database | None = None,
        config: KeyRouterConfig | None = None,
        secret_resolver: EnvSecretResolver | None = None,
    ) -> None:
        self.database = database or Database()
        self.config = config or KeyRouterConfig.from_env()
        self.secret_resolver = secret_resolver or EnvSecretResolver()
        self.repository = KeyRouterRepository(self.database)
        self.selector = AccountSelector(self.secret_resolver)
        self.endpoint_selector = EndpointSelector()

    def select_account(self, context: RequestContext) -> SelectionDecision:
        if self.config.mode is KeyRouterMode.DISABLED:
            return SelectionDecision(
                mode=self.config.mode,
                selected_account=None,
                provider="",
                pool_id=context.pool_id,
                model=context.requested_model,
                decision_reason="key_router_disabled",
            )
        accounts = self.repository.list_accounts()
        runtimes: dict[tuple[str, str, str], AccountRuntime] = {}
        active_counts: dict[str, int] = {}
        for account in accounts:
            runtime = self.repository.ensure_runtime(
                router_id=context.router_id,
                provider=account.provider,
                pool_id=account.pool_id,
                state=(
                    AccountState.ACTIVE
                    if account.enabled and account.account_state is AccountState.ACTIVE
                    else AccountState.DISABLED
                ),
            )
            runtimes[(context.router_id, account.provider, account.pool_id)] = runtime
            active_counts[account.account_id] = (
                self.repository.active_reservation_count(account.account_id)
            )
        decision = self.selector.select(
            mode=self.config.mode,
            context=context,
            accounts=accounts,
            runtimes=runtimes,
            active_counts=active_counts,
        )
        if decision.selected_account:
            handle = self.secret_resolver.get(decision.selected_account.secret_ref)
            if handle:
                decision = SelectionDecision(
                    mode=decision.mode,
                    selected_account=decision.selected_account,
                    provider=decision.provider,
                    pool_id=decision.pool_id,
                    model=decision.model,
                    decision_reason=decision.decision_reason,
                    eligible_count=decision.eligible_count,
                    rejected=decision.rejected,
                    score=decision.score,
                    secret_fingerprint=handle.fingerprint,
                )
        if self.config.mode is KeyRouterMode.SHADOW:
            self.repository.record_audit(
                audit_event(
                    event_type="shadow_selection",
                    provider=decision.provider,
                    pool_id=decision.pool_id,
                    account_id=(
                        decision.selected_account.account_id
                        if decision.selected_account
                        else ""
                    ),
                    model=decision.model,
                    decision_reason=decision.decision_reason,
                    request_id=context.request_id,
                    metadata=decision.sanitized(),
                )
            )
        if self.config.mode is KeyRouterMode.ACTIVE and not decision.selected_account:
            raise KeyRouterUnavailable("No eligible provider account")
        return decision

    def validate_active_configuration(self) -> None:
        accounts = self.repository.list_accounts()
        endpoints = self.repository.list_endpoints(enabled_only=True)
        dispatchable_account_providers = {
            account.provider
            for account in accounts
            if account.enabled
            and account.account_state is AccountState.ACTIVE
            and get_provider_adapter(account.provider) is not None
            and self.secret_resolver.get(account.secret_ref) is not None
        }
        dispatchable_endpoint_providers = {
            endpoint.provider
            for endpoint in endpoints
            if get_provider_adapter(endpoint.provider) is not None
        }
        self.config.validate_active_safety(
            has_enabled_accounts=bool(
                dispatchable_account_providers & dispatchable_endpoint_providers
            ),
            has_enabled_endpoints=bool(dispatchable_endpoint_providers),
        )

    def select_route(
        self,
        context: RequestContext,
        *,
        excluded_candidates: set[tuple[str, str, str]] | None = None,
        excluded_providers: set[str] | None = None,
        excluded_models: set[tuple[str, str]] | None = None,
    ) -> SelectionDecision:
        if self.config.mode is KeyRouterMode.DISABLED:
            return SelectionDecision(
                mode=self.config.mode,
                selected_account=None,
                provider="",
                pool_id=context.pool_id,
                model=context.requested_model,
                decision_reason="key_router_disabled",
            )
        if self.config.mode is KeyRouterMode.ACTIVE:
            self.validate_active_configuration()
        policy = get_policy(context.policy_id or context.task_type)
        circuits = self.repository.list_circuits()
        open_circuits: set[tuple[str, str]] = set()
        for item in circuits:
            if item["state"] not in {"OPEN", "HALF_OPEN"}:
                continue
            if not self.repository.acquire_half_open_probe(
                str(item["circuit_id"]),
                probe_limit=self.config.half_open_probe_count,
            ):
                open_circuits.add(
                    (str(item["provider"]), str(item["provider_model_id"]))
                )
        endpoints = [
            endpoint
            for endpoint in self.repository.list_endpoints(enabled_only=True)
            if get_provider_adapter(endpoint.provider) is not None
        ]
        endpoint, endpoint_rejected = self.endpoint_selector.select(
            context=context,
            policy=policy,
            endpoints=endpoints,
            require_verified_vision=self.config.require_verified_vision,
            allow_declared_only=self.config.allow_declared_only_capabilities,
            open_circuits=open_circuits,
            excluded_providers=excluded_providers,
            excluded_models=excluded_models,
        )
        if endpoint is None:
            decision = SelectionDecision(
                mode=self.config.mode,
                selected_account=None,
                provider="",
                pool_id=context.pool_id,
                model=context.requested_model,
                decision_reason="no_eligible_endpoint",
                rejected=endpoint_rejected,
                reason_codes=("no_eligible_endpoint",),
            )
        else:
            route_context = replace(
                context,
                requested_model=endpoint.provider_model_id,
                allowed_providers=(endpoint.provider,),
                requires_vision=context.requires_vision or policy.vision_required,
                requires_structured_output=(
                    context.requires_structured_output
                    or policy.structured_output_required
                ),
            )
            excluded_candidates = excluded_candidates or set()
            accounts = [
                account
                for account in self.repository.list_accounts()
                if (
                    endpoint.provider,
                    endpoint.provider_model_id,
                    account.account_id,
                )
                not in excluded_candidates
            ]
            runtimes: dict[tuple[str, str, str], AccountRuntime] = {}
            active_counts: dict[str, int] = {}
            for account in accounts:
                runtime = self.repository.ensure_runtime(
                    router_id=route_context.router_id,
                    provider=account.provider,
                    pool_id=account.pool_id,
                    state=(
                        AccountState.ACTIVE
                        if account.enabled
                        and account.account_state is AccountState.ACTIVE
                        else AccountState.DISABLED
                    ),
                )
                runtimes[
                    (route_context.router_id, account.provider, account.pool_id)
                ] = runtime
                active_counts[account.account_id] = (
                    self.repository.active_reservation_count(account.account_id)
                )
            account_decision = self.selector.select(
                mode=self.config.mode,
                context=route_context,
                accounts=accounts,
                runtimes=runtimes,
                active_counts=active_counts,
            )
            decision = replace(
                account_decision,
                endpoint_id=endpoint.endpoint_id,
                canonical_model_id=endpoint.canonical_model_id,
                rejected=endpoint_rejected + account_decision.rejected,
                reason_codes=(
                    (
                        "endpoint_capabilities_satisfied",
                        "account_policy_satisfied",
                        "deterministic_score_selected",
                    )
                    if account_decision.selected_account
                    else ("no_eligible_account",)
                ),
            )
            if decision.selected_account:
                handle = self.secret_resolver.get(decision.selected_account.secret_ref)
                if handle:
                    decision = replace(decision, secret_fingerprint=handle.fingerprint)
        if self.config.mode is KeyRouterMode.SHADOW:
            self.repository.record_audit(
                audit_event(
                    event_type="shadow_route_decision",
                    provider=decision.provider,
                    pool_id=decision.pool_id,
                    account_id=(
                        decision.selected_account.account_id
                        if decision.selected_account
                        else ""
                    ),
                    model=decision.model,
                    decision_reason=decision.decision_reason,
                    request_id=context.request_id,
                    metadata=decision.sanitized(),
                )
            )
        if self.config.mode is KeyRouterMode.ACTIVE and not decision.selected_account:
            raise KeyRouterUnavailable(decision.decision_reason)
        return decision

    def reserve(
        self,
        context: RequestContext,
        decision: SelectionDecision,
        *,
        idempotency_key: str,
        attempt_id: str = "",
        parent_request_id: str = "",
    ) -> RequestReservation:
        if self.config.mode is not KeyRouterMode.ACTIVE:
            raise KeyRouterUnavailable("Reservations require KEY_ROUTER_MODE=active")
        if decision.selected_account is None:
            raise KeyRouterUnavailable("No selected account to reserve")
        request_limit, token_limit, cost_limit = quota_limits_for(
            decision.selected_account, context
        )
        return self.repository.reserve_quota(
            idempotency_key=idempotency_key,
            provider=decision.selected_account.provider,
            pool_id=decision.selected_account.pool_id,
            account_id=decision.selected_account.account_id,
            model=context.requested_model,
            request_limit=request_limit,
            token_limit=token_limit,
            cost_limit=cost_limit,
            estimated_input_tokens=context.estimated_input_tokens,
            estimated_output_tokens=context.estimated_output_tokens,
            estimated_cost=float(context.estimated_cost or 0),
            ttl_seconds=self.config.reservation_ttl_seconds,
            request_id=context.request_id,
            parent_request_id=parent_request_id or context.request_id,
            attempt_id=attempt_id,
            provider_model_id=decision.model,
            policy_id=context.policy_id or context.task_type,
        )

    def mark_dispatching(self, reservation: RequestReservation) -> RequestReservation:
        return self.repository.mark_dispatching(
            reservation.reservation_id, reservation.reservation_token
        )

    def mark_sent(self, reservation: RequestReservation) -> RequestReservation:
        return self.repository.mark_sent(
            reservation.reservation_id, reservation.reservation_token
        )

    def settle_success(
        self,
        reservation: RequestReservation,
        *,
        actual_input_tokens: int,
        actual_output_tokens: int,
        actual_cost: float,
    ) -> RequestReservation:
        return self.repository.settle_reservation(
            reservation_id=reservation.reservation_id,
            reservation_token=reservation.reservation_token,
            state=ReservationState.SETTLED_SUCCESS,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            actual_cost=actual_cost,
        )

    def settle_failure(
        self,
        reservation: RequestReservation,
        *,
        error_code: str,
        request_count_consumed: bool | None = None,
        token_cost_consumed: bool | None = None,
        monetary_cost_consumed: bool | None = None,
    ) -> RequestReservation:
        return self.repository.settle_reservation(
            reservation_id=reservation.reservation_id,
            reservation_token=reservation.reservation_token,
            state=ReservationState.SETTLED_FAILURE,
            error_code=error_code,
            request_count_consumed=request_count_consumed,
            token_cost_consumed=token_cost_consumed,
            monetary_cost_consumed=monetary_cost_consumed,
        )

    def release(
        self, reservation: RequestReservation, *, reason: str
    ) -> RequestReservation:
        return self.repository.settle_reservation(
            reservation_id=reservation.reservation_id,
            reservation_token=reservation.reservation_token,
            state=ReservationState.RELEASED,
            error_code=reason,
        )

    def mark_uncertain(
        self, reservation: RequestReservation, *, error_code: str
    ) -> RequestReservation:
        return self.repository.mark_uncertain(
            reservation.reservation_id, reservation.reservation_token, error_code
        )

    def _apply_normalized_error(self, error: NormalizedProviderError) -> None:
        self.repository.record_normalized_error(error)
        if error.account_action == "quarantine" and error.provider_account_id:
            self.repository.set_account_state(
                error.provider_account_id,
                AccountState.QUARANTINED,
                enabled=False,
            )
        elif (
            error.account_action == "cooldown"
            and error.provider_account_id
            and error.scope is not ErrorScope.MODEL
        ):
            cooldown_until = (
                datetime.now(timezone.utc)
                + timedelta(
                    seconds=error.retry_after_seconds
                    or self.config.default_cooldown_seconds
                )
            ).isoformat()
            self.repository.set_account_cooldown(
                error.provider_account_id, cooldown_until
            )
        if error.provider_action == "record_circuit_failure":
            self.repository.record_circuit_failure(
                level="provider",
                provider=error.provider,
                failure_threshold=self.config.failure_threshold,
                open_seconds=self.config.circuit_open_seconds,
                failure_window_seconds=self.config.circuit_failure_window_seconds,
            )
            if error.provider_model_id:
                self.repository.record_circuit_failure(
                    level="model",
                    provider=error.provider,
                    provider_model_id=error.provider_model_id,
                    failure_threshold=self.config.failure_threshold,
                    open_seconds=self.config.circuit_open_seconds,
                    failure_window_seconds=self.config.circuit_failure_window_seconds,
                )
        self.repository.record_audit(
            audit_event(
                event_type="provider_error_normalized",
                provider=error.provider,
                account_id=error.provider_account_id,
                model=error.provider_model_id,
                decision_reason=error.error_type.value,
                metadata={
                    "error_type": error.error_type.value,
                    "scope": error.scope.value,
                    "retryable": error.retryable,
                    "safe_to_failover": error.safe_to_failover,
                    "retry_after_seconds": error.retry_after_seconds,
                    "raw_error_fingerprint": error.raw_error_fingerprint,
                },
            )
        )

    def dispatch(
        self,
        context: RequestContext,
        request: UnifiedProviderRequest,
        *,
        transport: Transport,
        local_result: str = "",
        local_quality: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> DispatchResult:
        if self.config.mode is not KeyRouterMode.ACTIVE:
            raise KeyRouterUnavailable("Dispatch requires KEY_ROUTER_MODE=active")
        policy = get_policy(context.policy_id or context.task_type)
        limits = RetryLimits(
            maximum_attempts=min(self.config.max_attempts, policy.maximum_attempts),
            maximum_provider_failovers=min(
                self.config.max_provider_failovers,
                policy.maximum_provider_failovers,
            ),
            maximum_account_failovers=min(
                self.config.max_account_failovers,
                policy.maximum_account_failovers,
            ),
            retry_budget=min(self.config.max_attempts, policy.maximum_attempts),
            total_timeout_seconds=self.config.total_timeout_seconds,
            allow_retry_after_send_timeout=policy.allow_retry_after_send_timeout,
        )
        coordinator = RetryCoordinator(limits)
        state = coordinator.start(context.request_id)
        last_error: NormalizedProviderError | None = None
        while state.attempts < limits.maximum_attempts:
            try:
                decision = self.select_route(
                    context,
                    excluded_candidates=state.excluded_candidates,
                    excluded_providers=state.excluded_providers,
                    excluded_models=state.excluded_models,
                )
            except (KeyRouterUnavailable, KeyRouterConfigurationError):
                break
            account = decision.selected_account
            if account is None:
                break
            adapter = get_provider_adapter(decision.provider)
            handle = self.secret_resolver.get(account.secret_ref)
            if adapter is None or handle is None:
                raise KeyRouterUnavailable("Selected route is not dispatchable")
            attempt_id = coordinator.begin_attempt(
                state,
                provider=decision.provider,
                model=decision.model,
                account_id=account.account_id,
            )
            provider_request = adapter.build_request(
                replace(request, model=decision.model), secret_value=handle.value
            )
            reservation = self.reserve(
                context,
                decision,
                idempotency_key=f"{context.request_id}:{attempt_id}",
                attempt_id=attempt_id,
                parent_request_id=context.request_id,
            )
            self.repository.record_attempt(
                attempt_id=attempt_id,
                parent_request_id=context.request_id,
                request_id=context.request_id,
                attempt_number=state.attempts,
                provider=decision.provider,
                provider_model_id=decision.model,
                account_id=account.account_id,
                reservation_id=reservation.reservation_id,
            )
            dispatching = self.mark_dispatching(reservation)
            try:
                response = adapter.send_request(provider_request, transport=transport)
                sent = self.mark_sent(dispatching)
            except Exception as exc:
                request_sent = bool(getattr(exc, "request_sent", False))
                error = adapter.normalize_error(
                    exception=exc,
                    provider_account_id=account.account_id,
                    provider_model_id=decision.model,
                    request_sent=request_sent,
                    default_cooldown_seconds=self.config.default_cooldown_seconds,
                )
                self._apply_normalized_error(error)
                if request_sent:
                    self.mark_uncertain(dispatching, error_code=error.error_type.value)
                else:
                    self.release(dispatching, reason=error.error_type.value)
                self.repository.complete_attempt(
                    attempt_id,
                    outcome="uncertain" if request_sent else "failed",
                    error_type=error.error_type.value,
                )
                coordinator.record_failure(
                    state,
                    provider=decision.provider,
                    model=decision.model,
                    account_id=account.account_id,
                    error=error,
                )
                last_error = error
                if not coordinator.can_retry(state, error):
                    break
                sleeper(coordinator.backoff_seconds(state, error))
                continue
            if response.status_code >= 400:
                error = adapter.normalize_error(
                    response=response,
                    provider_account_id=account.account_id,
                    provider_model_id=decision.model,
                    request_sent=True,
                    default_cooldown_seconds=self.config.default_cooldown_seconds,
                )
                self._apply_normalized_error(error)
                request_consumed = response.status_code in {400, 422}
                self.settle_failure(
                    sent,
                    error_code=error.error_type.value,
                    request_count_consumed=request_consumed,
                    token_cost_consumed=False,
                    monetary_cost_consumed=False,
                )
                self.repository.complete_attempt(
                    attempt_id, outcome="failed", error_type=error.error_type.value
                )
                coordinator.record_failure(
                    state,
                    provider=decision.provider,
                    model=decision.model,
                    account_id=account.account_id,
                    error=error,
                )
                last_error = error
                if not coordinator.can_retry(state, error):
                    break
                sleeper(coordinator.backoff_seconds(state, error))
                continue
            try:
                text = adapter.parse_response(response)
            except ValueError as exc:
                error = adapter.normalize_error(
                    response=response,
                    exception=exc,
                    provider_account_id=account.account_id,
                    provider_model_id=decision.model,
                    request_sent=True,
                )
                self._apply_normalized_error(error)
                self.settle_failure(sent, error_code=error.error_type.value)
                self.repository.complete_attempt(
                    attempt_id, outcome="failed", error_type=error.error_type.value
                )
                last_error = error
                break
            usage = adapter.extract_usage(response)
            self.settle_success(
                sent,
                actual_input_tokens=int(usage.get("input_tokens") or 0),
                actual_output_tokens=int(usage.get("output_tokens") or 0),
                actual_cost=float(usage.get("cost") or 0),
            )
            for circuit in self.repository.list_circuits():
                if circuit["provider"] != decision.provider:
                    continue
                if (
                    circuit["level"] == "model"
                    and circuit["provider_model_id"] != decision.model
                ):
                    continue
                if circuit["state"] == "HALF_OPEN":
                    self.repository.record_circuit_success(str(circuit["circuit_id"]))
            self.repository.complete_attempt(attempt_id, outcome="success")
            return DispatchResult(
                text=text,
                provider=decision.provider,
                provider_model_id=decision.model,
                account_id=account.account_id,
                attempts=state.attempts,
            )
        if local_result:
            return DispatchResult(
                text=local_result,
                attempts=state.attempts,
                used_local_result=True,
                manual_review_required=(
                    local_quality is None or float(local_quality) < 90.0
                ),
                final_error_type=(
                    last_error.error_type.value if last_error else "NO_ROUTE"
                ),
            )
        error_type = (
            last_error.error_type.value
            if last_error
            else ProviderErrorType.UNKNOWN_PROVIDER_ERROR.value
        )
        raise ProviderDispatchError(f"Cloud routes exhausted: {error_type}")

    def start_drain(
        self, *, provider: str, pool_id: str, drain_seconds: int = 30
    ) -> AccountRuntime:
        runtime = self.repository.ensure_runtime(
            router_id=self.config.router_id,
            provider=provider,
            pool_id=pool_id,
            state=AccountState.ACTIVE,
        )
        validate_transition(runtime.state, AccountState.DRAINING)
        deadline = (
            datetime.now(timezone.utc) + timedelta(seconds=max(0, drain_seconds))
        ).isoformat()
        return self.repository.compare_and_set_runtime(
            runtime=runtime,
            new_state=AccountState.DRAINING,
            drain_deadline=deadline,
        )

    def close_drained(self, *, provider: str, pool_id: str) -> AccountRuntime:
        runtime = self.repository.ensure_runtime(
            router_id=self.config.router_id, provider=provider, pool_id=pool_id
        )
        validate_transition(runtime.state, AccountState.CLOSED)
        return self.repository.compare_and_set_runtime(
            runtime=runtime,
            new_state=AccountState.CLOSED,
            closed_at=utc_now(),
            active_account_id="",
        )

    def begin_switch(
        self, *, provider: str, pool_id: str, pending_account_id: str
    ) -> AccountRuntime:
        runtime = self.repository.ensure_runtime(
            router_id=self.config.router_id, provider=provider, pool_id=pool_id
        )
        validate_transition(runtime.state, AccountState.SWITCHING)
        return self.repository.compare_and_set_runtime(
            runtime=runtime,
            new_state=AccountState.SWITCHING,
            pending_account_id=pending_account_id,
            switch_id=uuid.uuid4().hex,
            account_epoch=runtime.account_epoch + 1,
            switch_started_at=utc_now(),
        )

    def complete_switch(
        self,
        *,
        provider: str,
        pool_id: str,
        switch_id: str,
        account_epoch: int,
    ) -> AccountRuntime:
        runtime = self.repository.ensure_runtime(
            router_id=self.config.router_id, provider=provider, pool_id=pool_id
        )
        if runtime.switch_id != switch_id or runtime.account_epoch != account_epoch:
            raise QuotaExceeded("stale_switch")
        validate_transition(runtime.state, AccountState.ACTIVE)
        return self.repository.compare_and_set_runtime(
            runtime=runtime,
            new_state=AccountState.ACTIVE,
            active_account_id=runtime.pending_account_id,
            pending_account_id="",
        )

    def fail_switch(
        self, *, provider: str, pool_id: str, error_code: str
    ) -> AccountRuntime:
        runtime = self.repository.ensure_runtime(
            router_id=self.config.router_id, provider=provider, pool_id=pool_id
        )
        validate_transition(runtime.state, AccountState.ERROR)
        return self.repository.compare_and_set_runtime(
            runtime=runtime,
            new_state=AccountState.ERROR,
            last_error_code=error_code,
        )

    def get_status(self) -> dict:
        snapshot = self.repository.status_snapshot()
        accounts = snapshot["accounts"]
        provider_names = sorted({str(item["provider"]) for item in accounts})
        recent_events = self.repository.recent_audit(limit=20)
        payload = {
            "mode": self.config.mode.value,
            "router_id": self.config.router_id,
            "total_providers": len(provider_names),
            "total_accounts": len(accounts),
            "enabled_accounts": sum(bool(item["enabled"]) for item in accounts),
            "quarantined_accounts": sum(
                item.get("account_state") == AccountState.QUARANTINED.value
                for item in accounts
            ),
            "cooldown_accounts": sum(
                bool(item.get("cooldown_until")) for item in accounts
            ),
            "open_circuits": snapshot["open_circuits"],
            "pending_reservations": sum(
                snapshot["reservation_counts"].get(state, 0)
                for state in ("PENDING", "RESERVED", "DISPATCHING", "SENT")
            ),
            "uncertain_reservations": snapshot["reservation_counts"].get(
                "UNCERTAIN", 0
            ),
            "capability_registry_summary": {
                "enabled_endpoints": snapshot["enabled_endpoints"],
                "verification_states": snapshot["capability_status"],
            },
            "normalized_error_counters": snapshot["error_counts"],
            "provider_health_summaries": [
                {
                    "provider": provider,
                    "enabled_accounts": sum(
                        bool(item["enabled"])
                        for item in accounts
                        if item["provider"] == provider
                    ),
                }
                for provider in provider_names
            ],
            "providers_configured": [
                item["provider"] for item in provider_capability_snapshot()
            ],
            "pools": snapshot["runtimes"],
            "accounts": accounts,
            "quotas": {"active_reservations": snapshot["active_reservations"]},
            "recent_routing_decisions": recent_events,
            "recent_events": recent_events,
        }
        sanitized = sanitize_metadata(payload)
        return sanitized if isinstance(sanitized, dict) else {}

    def health_snapshot(self) -> dict:
        return {
            "mode": self.config.mode.value,
            "router_id": self.config.router_id,
            "providers": provider_capability_snapshot(),
            "status": self.get_status(),
        }
