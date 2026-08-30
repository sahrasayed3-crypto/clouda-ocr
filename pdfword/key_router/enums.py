from __future__ import annotations

from enum import Enum


class KeyRouterMode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"


class AccountState(str, Enum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    CLOSED = "CLOSED"
    SWITCHING = "SWITCHING"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    QUARANTINED = "QUARANTINED"
    COOLDOWN = "COOLDOWN"


class ReservationState(str, Enum):
    PENDING = "PENDING"
    RESERVED = "RESERVED"
    DISPATCHING = "DISPATCHING"
    SENT = "SENT"
    SETTLED_SUCCESS = "SETTLED_SUCCESS"
    SETTLED_FAILURE = "SETTLED_FAILURE"
    UNCERTAIN = "UNCERTAIN"
    EXPIRED = "EXPIRED"
    RELEASED = "RELEASED"
    CANCELLED = "CANCELLED"

    # Source-compatible aliases for the Phase 1 API.
    SUCCEEDED = "SETTLED_SUCCESS"
    FAILED = "SETTLED_FAILURE"


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class WindowType(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"
    PROVIDER = "provider"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"


class ErrorCategory(str, Enum):
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    BILLING = "billing"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PROVIDER_5XX = "provider_5xx"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_MODALITY = "unsupported_modality"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNKNOWN = "unknown"


class CapabilityVerificationStatus(str, Enum):
    DECLARED = "declared"
    VERIFIED = "verified"
    FAILED_VERIFICATION = "failed_verification"
    STALE = "stale"
    UNKNOWN = "unknown"


class PrivacyClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class ProviderErrorType(str, Enum):
    AUTH_FAILED = "AUTH_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMITED_TEMPORARY = "RATE_LIMITED_TEMPORARY"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    BILLING_REQUIRED = "BILLING_REQUIRED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    PROVIDER_OVERLOADED = "PROVIDER_OVERLOADED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    NETWORK_ERROR = "NETWORK_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    TIMEOUT_BEFORE_SEND = "TIMEOUT_BEFORE_SEND"
    TIMEOUT_AFTER_SEND = "TIMEOUT_AFTER_SEND"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    CONTENT_REFUSAL = "CONTENT_REFUSAL"
    SAFETY_REJECTION = "SAFETY_REJECTION"
    UNKNOWN_PROVIDER_ERROR = "UNKNOWN_PROVIDER_ERROR"


class ErrorScope(str, Enum):
    REQUEST = "request"
    KEY = "key"
    ACCOUNT = "account"
    MODEL = "model"
    PROVIDER = "provider"
    ORGANIZATION = "organization"
    GLOBAL = "global"
    UNKNOWN = "unknown"
