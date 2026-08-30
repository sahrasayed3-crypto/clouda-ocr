from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace

from .capabilities import ModelEndpointCapability
from .enums import AccountState, CircuitState, KeyRouterMode
from .exceptions import UnsupportedCapability
from .models import AccountRuntime, ProviderAccount, RequestContext, SelectionDecision
from .policies import RoutingPolicy
from .providers.registry import get_provider_definition
from .credential_refs import EnvSecretResolver


def _future(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed > datetime.now(timezone.utc)


class AccountSelector:
    def __init__(self, secret_resolver: EnvSecretResolver | None = None) -> None:
        self.secret_resolver = secret_resolver or EnvSecretResolver()

    def select(
        self,
        *,
        mode: KeyRouterMode,
        context: RequestContext,
        accounts: list[ProviderAccount],
        runtimes: dict[tuple[str, str, str], AccountRuntime],
        active_counts: dict[str, int],
    ) -> SelectionDecision:
        allowed = {item.lower() for item in context.allowed_providers}
        rejected: list[dict[str, str]] = []
        eligible: list[tuple[float, str, ProviderAccount, str]] = []
        for account in accounts:
            reason = self._rejection_reason(
                context=context,
                account=account,
                runtime=runtimes.get(
                    (context.router_id, account.provider, account.pool_id)
                ),
                active_count=active_counts.get(account.account_id, 0),
                allowed=allowed,
            )
            if reason:
                rejected.append({"account_id": account.account_id, "reason": reason})
                continue
            score = self._score(
                account, context, active_counts.get(account.account_id, 0)
            )
            eligible.append(
                (
                    score,
                    f"{account.provider}:{account.pool_id}:{account.account_id}",
                    account,
                    "eligible",
                )
            )
        eligible.sort(key=lambda item: (-item[0], item[1]))
        if not eligible:
            return SelectionDecision(
                mode=mode,
                selected_account=None,
                provider="",
                pool_id=context.pool_id,
                model=context.requested_model,
                decision_reason="no_eligible_account",
                eligible_count=0,
                rejected=tuple(rejected),
            )
        score, _tie, account, reason = eligible[0]
        return SelectionDecision(
            mode=mode,
            selected_account=account,
            provider=account.provider,
            pool_id=account.pool_id,
            model=context.requested_model,
            decision_reason=reason,
            eligible_count=len(eligible),
            rejected=tuple(rejected),
            score=score,
        )

    def _rejection_reason(
        self,
        *,
        context: RequestContext,
        account: ProviderAccount,
        runtime: AccountRuntime | None,
        active_count: int,
        allowed: set[str],
    ) -> str:
        if allowed and account.provider not in allowed:
            return "provider_not_allowed"
        if not account.enabled:
            return "account_disabled"
        if account.account_state is AccountState.QUARANTINED:
            return "account_quarantined"
        if account.account_state is AccountState.COOLDOWN:
            return "account_cooldown"
        if account.account_state not in {AccountState.ACTIVE}:
            return f"account_{account.account_state.value.lower()}"
        if _future(account.cooldown_until):
            return "account_cooldown"
        if (
            not account.secret_ref
            or self.secret_resolver.get(account.secret_ref) is None
        ):
            return "secret_missing"
        if runtime and runtime.state not in {AccountState.ACTIVE}:
            return f"runtime_{runtime.state.value.lower()}"
        if runtime and _future(runtime.cooldown_until):
            return "cooldown"
        if active_count >= account.max_concurrency:
            return "max_concurrency"
        if context.requires_vision and not account.allows_vision:
            return "vision_not_allowed"
        if not account.allows_text:
            return "text_not_allowed"
        if context.normalized_privacy not in set(account.allowed_privacy):
            return "privacy_not_allowed"
        definition = get_provider_definition(account.provider)
        if definition is None:
            return "unknown_provider"
        if context.requires_vision and not definition.capabilities.supports_image_input:
            return "provider_no_vision"
        if (
            context.requires_structured_output
            and not definition.capabilities.supports_structured_output
        ):
            return "provider_no_structured_output"
        if context.estimated_cost is None and context.free_only:
            return "unknown_price_free_only"
        estimated_cost = float(context.estimated_cost or 0)
        if context.free_only and estimated_cost > 0:
            return "paid_model_blocked"
        if estimated_cost > 0 and not account.allows_paid:
            return "paid_not_allowed"
        if estimated_cost > 0 and account.daily_cost_limit <= 0:
            return "paid_without_budget"
        if context.max_cost and estimated_cost > context.max_cost:
            return "request_cost_limit"
        if account.metadata.get("circuit_state") == CircuitState.OPEN.value:
            return "circuit_open"
        return ""

    def _score(
        self, account: ProviderAccount, context: RequestContext, active_count: int
    ) -> float:
        estimated_cost = float(context.estimated_cost or 0)
        score = float(account.priority * 1000 + account.weight * 10)
        if account.allows_free and estimated_cost == 0:
            score += 250
        if account.daily_cost_limit:
            score += max(0.0, account.daily_cost_limit - estimated_cost)
        score -= active_count * 50
        score -= estimated_cost * 100
        if context.page_quality is not None:
            score += max(0.0, min(100.0, context.page_quality)) / 100.0
        return score


class EndpointSelector:
    def select(
        self,
        *,
        context: RequestContext,
        policy: RoutingPolicy,
        endpoints: list[ModelEndpointCapability],
        require_verified_vision: bool,
        allow_declared_only: bool,
        open_circuits: set[tuple[str, str]],
        excluded_providers: set[str] | None = None,
        excluded_models: set[tuple[str, str]] | None = None,
    ) -> tuple[ModelEndpointCapability | None, tuple[dict[str, str], ...]]:
        rejected: list[dict[str, str]] = []
        eligible: list[tuple[float, str, ModelEndpointCapability]] = []
        allowed = {item.lower() for item in policy.allowed_providers}
        blocked = {item.lower() for item in policy.blocked_providers}
        allowed_models = set(policy.allowed_models)
        context_allowed = {item.lower() for item in context.allowed_providers}
        excluded_providers = excluded_providers or set()
        excluded_models = excluded_models or set()
        for endpoint in endpoints:
            reason = ""
            if not endpoint.enabled:
                reason = "endpoint_disabled"
            elif (
                context.canonical_model_id
                and endpoint.canonical_model_id != context.canonical_model_id
            ):
                reason = "canonical_model_mismatch"
            elif allowed and endpoint.provider not in allowed:
                reason = "provider_not_allowed_by_policy"
            elif context_allowed and endpoint.provider not in context_allowed:
                reason = "provider_not_allowed_by_request"
            elif (
                endpoint.provider in blocked or endpoint.provider in excluded_providers
            ):
                reason = "provider_blocked"
            elif allowed_models and endpoint.canonical_model_id not in allowed_models:
                reason = "model_not_allowed"
            elif (endpoint.provider, endpoint.provider_model_id) in excluded_models:
                reason = "model_excluded"
            elif (endpoint.provider, "") in open_circuits or (
                endpoint.provider,
                endpoint.provider_model_id,
            ) in open_circuits:
                reason = "circuit_open"
            elif (
                policy.maximum_cost
                and context.estimated_cost is not None
                and context.estimated_cost > policy.maximum_cost
            ):
                reason = "policy_cost_limit"
            elif (
                policy.maximum_latency_ms
                and endpoint.expected_latency_ms > policy.maximum_latency_ms
            ):
                reason = "policy_latency_limit"
            elif (
                "structured_json" in policy.required_capabilities
                and not endpoint.supports_structured_json
            ):
                reason = "structured_output_not_supported"
            elif (
                "system_prompt" in policy.required_capabilities
                and not endpoint.supports_system_prompt
            ):
                reason = "system_prompt_not_supported"
            else:
                effective_context = replace(
                    context,
                    requires_vision=context.requires_vision or policy.vision_required,
                    requires_structured_output=(
                        context.requires_structured_output
                        or policy.structured_output_required
                    ),
                )
                try:
                    endpoint.validate_request(
                        effective_context,
                        require_verified_vision=require_verified_vision,
                        allow_declared_only=allow_declared_only,
                    )
                except UnsupportedCapability as exc:
                    reason = str(exc)
            if reason:
                rejected.append({"endpoint_id": endpoint.endpoint_id, "reason": reason})
                continue
            score = endpoint.quality_score * 1000
            score -= endpoint.expected_latency_ms / 10
            score -= (
                endpoint.pricing_input
                + endpoint.pricing_output
                + endpoint.pricing_image
            )
            score += sum(
                10.0
                for capability in policy.preferred_capabilities
                if (
                    capability == "structured_json"
                    and endpoint.supports_structured_json
                )
            )
            tie = f"{endpoint.provider}:{endpoint.provider_model_id}:{endpoint.endpoint_id}"
            eligible.append((score, tie, endpoint))
        eligible.sort(key=lambda item: (-item[0], item[1]))
        return (eligible[0][2] if eligible else None, tuple(rejected))
