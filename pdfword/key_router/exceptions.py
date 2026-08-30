from __future__ import annotations


class KeyRouterError(RuntimeError):
    """Base class for sanitized key-router failures."""


class KeyRouterConfigurationError(KeyRouterError):
    pass


class KeyRouterUnavailable(KeyRouterError):
    pass


class QuotaExceeded(KeyRouterError):
    pass


class InvalidStateTransition(KeyRouterError):
    pass


class FencingTokenError(KeyRouterError):
    pass


class UnsupportedCapability(KeyRouterError):
    pass


class ProviderDispatchError(KeyRouterError):
    """Sanitized provider failure raised by the experimental dispatcher."""


class RetryBudgetExhausted(KeyRouterError):
    pass
