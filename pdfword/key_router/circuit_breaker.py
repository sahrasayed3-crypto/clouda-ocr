from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from .enums import CircuitState, ErrorCategory


def parse_retry_after(
    value: str | None, *, now: datetime | None = None
) -> datetime | None:
    if not value:
        return None
    current = now or datetime.now(timezone.utc)
    raw = value.strip()
    try:
        seconds = max(0, int(raw))
        return current + timedelta(seconds=seconds)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def classify_error(status_code: int | None, message: str = "") -> ErrorCategory:
    msg = message.lower()
    if status_code in {401, 403} or "auth" in msg:
        return (
            ErrorCategory.AUTHENTICATION
            if status_code == 401
            else ErrorCategory.PERMISSION
        )
    if status_code == 429 or "rate limit" in msg:
        return ErrorCategory.RATE_LIMIT
    if "billing" in msg or "quota" in msg:
        return ErrorCategory.BILLING
    if status_code and 500 <= status_code <= 599:
        return ErrorCategory.PROVIDER_5XX
    if status_code == 408 or "timeout" in msg:
        return ErrorCategory.TIMEOUT
    if status_code and 400 <= status_code < 500:
        return ErrorCategory.INVALID_REQUEST
    return ErrorCategory.UNKNOWN


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout_seconds: int = 60
    half_open_probe_count: int = 1

    def after_failure(
        self,
        *,
        state: CircuitState,
        consecutive_failures: int,
        error_category: ErrorCategory,
    ) -> tuple[CircuitState, int]:
        if error_category is ErrorCategory.INVALID_REQUEST:
            return state, consecutive_failures
        failures = consecutive_failures + 1
        if failures >= self.failure_threshold:
            return CircuitState.OPEN, failures
        return state, failures

    def state_for_time(
        self,
        *,
        state: CircuitState,
        cooldown_until: datetime | None,
        now: datetime | None = None,
    ) -> CircuitState:
        current = now or datetime.now(timezone.utc)
        if state is CircuitState.OPEN and cooldown_until and current >= cooldown_until:
            return CircuitState.HALF_OPEN
        return state

    def after_success(self, *, state: CircuitState) -> tuple[CircuitState, int]:
        if state in {CircuitState.HALF_OPEN, CircuitState.CLOSED}:
            return CircuitState.CLOSED, 0
        return state, 0
