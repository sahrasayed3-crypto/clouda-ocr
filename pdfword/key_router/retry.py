from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .enums import ProviderErrorType
from .errors import NormalizedProviderError


@dataclass(frozen=True)
class RetryLimits:
    maximum_attempts: int = 3
    maximum_provider_failovers: int = 1
    maximum_account_failovers: int = 1
    retry_budget: int = 3
    total_timeout_seconds: int = 180
    exponential_backoff_seconds: float = 0.25
    maximum_backoff_seconds: float = 5.0
    jitter_fraction: float = 0.1
    allow_retry_after_send_timeout: bool = False


@dataclass
class RetryState:
    parent_request_id: str
    started_at: datetime
    attempts: int = 0
    provider_failovers: int = 0
    account_failovers: int = 0
    excluded_candidates: set[tuple[str, str, str]] = field(default_factory=set)
    excluded_providers: set[str] = field(default_factory=set)
    excluded_models: set[tuple[str, str]] = field(default_factory=set)
    attempt_ids: list[str] = field(default_factory=list)


class RetryCoordinator:
    def __init__(self, limits: RetryLimits) -> None:
        self.limits = limits

    def start(
        self, parent_request_id: str, *, now: datetime | None = None
    ) -> RetryState:
        return RetryState(
            parent_request_id=parent_request_id,
            started_at=now or datetime.now(timezone.utc),
        )

    def begin_attempt(
        self,
        state: RetryState,
        *,
        provider: str,
        model: str,
        account_id: str,
    ) -> str:
        candidate = (provider, model, account_id)
        if candidate in state.excluded_candidates:
            raise ValueError("Candidate is excluded for this request")
        state.attempts += 1
        material = f"{state.parent_request_id}:{state.attempts}:{provider}:{model}:{account_id}"
        attempt_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        state.attempt_ids.append(attempt_id)
        return attempt_id

    def record_failure(
        self,
        state: RetryState,
        *,
        provider: str,
        model: str,
        account_id: str,
        error: NormalizedProviderError,
    ) -> None:
        state.excluded_candidates.add((provider, model, account_id))
        if error.provider_action != "none":
            state.excluded_providers.add(provider)
            state.provider_failovers += 1
        if error.model_action != "none":
            state.excluded_models.add((provider, model))
        if error.account_action != "none":
            state.account_failovers += 1

    def can_retry(
        self,
        state: RetryState,
        error: NormalizedProviderError,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = now or datetime.now(timezone.utc)
        if state.attempts >= min(
            self.limits.maximum_attempts, self.limits.retry_budget
        ):
            return False
        if state.provider_failovers > self.limits.maximum_provider_failovers:
            return False
        if state.account_failovers > self.limits.maximum_account_failovers:
            return False
        if (
            current - state.started_at
        ).total_seconds() >= self.limits.total_timeout_seconds:
            return False
        if (
            error.error_type is ProviderErrorType.TIMEOUT_AFTER_SEND
            and not self.limits.allow_retry_after_send_timeout
        ):
            return False
        return bool(error.retryable or error.safe_to_failover)

    def backoff_seconds(
        self, state: RetryState, error: NormalizedProviderError
    ) -> float:
        if error.retry_after_seconds is not None:
            return float(error.retry_after_seconds)
        base = min(
            self.limits.maximum_backoff_seconds,
            self.limits.exponential_backoff_seconds * (2 ** max(0, state.attempts - 1)),
        )
        digest = hashlib.sha256(
            f"{state.parent_request_id}:{state.attempts}".encode("utf-8")
        ).digest()[0]
        deterministic_jitter = (digest / 255.0) * self.limits.jitter_fraction * base
        return base + deterministic_jitter
