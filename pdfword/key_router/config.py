from __future__ import annotations

import os
from dataclasses import dataclass, field

from .enums import KeyRouterMode
from .exceptions import KeyRouterConfigurationError


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, strict: bool) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        if strict:
            raise KeyRouterConfigurationError(f"Invalid {name}") from exc
        return default


def _env_float(name: str, default: float, *, strict: bool) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        if strict:
            raise KeyRouterConfigurationError(f"Invalid {name}") from exc
        return default


@dataclass(frozen=True)
class KeyRouterConfig:
    mode: KeyRouterMode = KeyRouterMode.DISABLED
    router_id: str = "clouda-local"
    free_only_default: bool = True
    allow_paid_default: bool = False
    reservation_ttl_seconds: int = 300
    uncertain_ttl_seconds: int = 1800
    default_daily_cost_limit: float = 0.0
    failure_threshold: int = 3
    recovery_timeout_seconds: int = 60
    half_open_probe_count: int = 1
    providers: dict[str, dict] = field(default_factory=dict)
    require_verified_vision: bool = True
    allow_declared_only_capabilities: bool = False
    max_attempts: int = 3
    max_provider_failovers: int = 1
    max_account_failovers: int = 1
    total_timeout_seconds: int = 180
    default_cooldown_seconds: int = 60
    circuit_open_seconds: int = 60
    circuit_failure_window_seconds: int = 60
    uncertain_reconciliation_seconds: int = 1800
    live_tests_enabled: bool = False

    @classmethod
    def from_env(cls) -> "KeyRouterConfig":
        raw_mode = os.getenv("KEY_ROUTER_MODE", "disabled").strip().lower()
        try:
            mode = KeyRouterMode(raw_mode)
        except ValueError as exc:
            raise KeyRouterConfigurationError("Invalid KEY_ROUTER_MODE") from exc
        strict = mode is KeyRouterMode.ACTIVE
        return cls(
            mode=mode,
            router_id=os.getenv("KEY_ROUTER_ID", "clouda-local").strip()
            or "clouda-local",
            free_only_default=_env_bool("KEY_ROUTER_FREE_ONLY_DEFAULT", True),
            allow_paid_default=_env_bool("KEY_ROUTER_ALLOW_PAID_DEFAULT", False),
            reservation_ttl_seconds=max(
                1, _env_int("KEY_ROUTER_RESERVATION_TTL_SECONDS", 300, strict=strict)
            ),
            uncertain_ttl_seconds=max(
                1, _env_int("KEY_ROUTER_UNCERTAIN_TTL_SECONDS", 1800, strict=strict)
            ),
            default_daily_cost_limit=max(
                0.0,
                _env_float("KEY_ROUTER_DEFAULT_DAILY_COST_LIMIT", 0.0, strict=strict),
            ),
            failure_threshold=max(
                1,
                _env_int(
                    "KEY_ROUTER_CIRCUIT_FAILURE_THRESHOLD",
                    _env_int("KEY_ROUTER_FAILURE_THRESHOLD", 3, strict=strict),
                    strict=strict,
                ),
            ),
            recovery_timeout_seconds=max(
                1,
                _env_int("KEY_ROUTER_RECOVERY_TIMEOUT_SECONDS", 60, strict=strict),
            ),
            half_open_probe_count=max(
                1, _env_int("KEY_ROUTER_HALF_OPEN_PROBE_COUNT", 1, strict=strict)
            ),
            require_verified_vision=_env_bool(
                "KEY_ROUTER_REQUIRE_VERIFIED_VISION", True
            ),
            allow_declared_only_capabilities=_env_bool(
                "KEY_ROUTER_ALLOW_DECLARED_ONLY_CAPABILITIES", False
            ),
            max_attempts=max(1, _env_int("KEY_ROUTER_MAX_ATTEMPTS", 3, strict=strict)),
            max_provider_failovers=max(
                0,
                _env_int("KEY_ROUTER_MAX_PROVIDER_FAILOVERS", 1, strict=strict),
            ),
            max_account_failovers=max(
                0,
                _env_int("KEY_ROUTER_MAX_ACCOUNT_FAILOVERS", 1, strict=strict),
            ),
            total_timeout_seconds=max(
                1,
                _env_int("KEY_ROUTER_TOTAL_TIMEOUT_SECONDS", 180, strict=strict),
            ),
            default_cooldown_seconds=max(
                1,
                _env_int("KEY_ROUTER_DEFAULT_COOLDOWN_SECONDS", 60, strict=strict),
            ),
            circuit_open_seconds=max(
                1,
                _env_int("KEY_ROUTER_CIRCUIT_OPEN_SECONDS", 60, strict=strict),
            ),
            circuit_failure_window_seconds=max(
                1,
                _env_int(
                    "KEY_ROUTER_CIRCUIT_FAILURE_WINDOW_SECONDS",
                    60,
                    strict=strict,
                ),
            ),
            uncertain_reconciliation_seconds=max(
                1,
                _env_int(
                    "KEY_ROUTER_UNCERTAIN_RECONCILIATION_SECONDS",
                    1800,
                    strict=strict,
                ),
            ),
            live_tests_enabled=_env_bool("KEY_ROUTER_LIVE_TESTS_ENABLED", False),
        )

    @property
    def enabled(self) -> bool:
        return self.mode is not KeyRouterMode.DISABLED

    def validate_active_safety(
        self, *, has_enabled_accounts: bool, has_enabled_endpoints: bool
    ) -> None:
        if self.mode is not KeyRouterMode.ACTIVE:
            return
        if self.allow_declared_only_capabilities and self.require_verified_vision:
            raise KeyRouterConfigurationError(
                "Active configuration conflicts: declared-only capabilities cannot "
                "override verified vision"
            )
        if not has_enabled_accounts:
            raise KeyRouterConfigurationError(
                "Active Key Router requires at least one enabled provider account"
            )
        if not has_enabled_endpoints:
            raise KeyRouterConfigurationError(
                "Active Key Router requires at least one enabled capability endpoint"
            )
