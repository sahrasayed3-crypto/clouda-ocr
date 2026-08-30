from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .circuit_breaker import parse_retry_after
from .enums import ErrorScope, ProviderErrorType

_SECRET_VALUE_RE = re.compile(
    r"(?i)(bearer\s+|api[-_ ]?key[=: ]+|token[=: ]+|secret[=: ]+)([^\s,;]+)"
)
_DATA_URL_RE = re.compile(
    r"(?i)data:(image|application)/[^;\s]+;base64,[A-Za-z0-9+/=]+"
)


def sanitize_provider_message(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = _SECRET_VALUE_RE.sub(lambda match: f"{match.group(1)}[redacted]", text)
    text = _DATA_URL_RE.sub("[redacted-image-payload]", text)
    return text[:500]


@dataclass(frozen=True)
class NormalizedProviderError:
    error_type: ProviderErrorType
    provider: str
    provider_account_id: str = ""
    provider_model_id: str = ""
    http_status: int | None = None
    provider_error_code: str = ""
    sanitized_message: str = ""
    retryable: bool = False
    safe_to_failover: bool = False
    retry_after_seconds: int | None = None
    scope: ErrorScope = ErrorScope.UNKNOWN
    credential_action: str = "none"
    account_action: str = "none"
    model_action: str = "none"
    provider_action: str = "none"
    reservation_action: str = "uncertain"
    request_may_have_been_processed: bool = False
    raw_error_fingerprint: str = ""
    occurred_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _payload_fields(payload: Any) -> tuple[str, str, str]:
    if not isinstance(payload, Mapping):
        return "", sanitize_provider_message(payload), ""
    error = payload.get("error", payload)
    if isinstance(error, Mapping):
        code = str(error.get("code") or error.get("type") or "")
        message = sanitize_provider_message(
            error.get("message") or error.get("detail") or error
        )
        scope = str(error.get("scope") or payload.get("scope") or "")
        return code, message, scope
    return "", sanitize_provider_message(error), ""


def _fingerprint(provider: str, status: int | None, code: str, message: str) -> str:
    material = f"{provider}|{status}|{code}|{message}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:20]


def normalize_provider_error(
    *,
    provider: str,
    provider_account_id: str = "",
    provider_model_id: str = "",
    http_status: int | None = None,
    payload: Any = None,
    headers: Mapping[str, str] | None = None,
    exception: BaseException | None = None,
    request_sent: bool = False,
    now: datetime | None = None,
    default_cooldown_seconds: int = 60,
) -> NormalizedProviderError:
    current = now or datetime.now(timezone.utc)
    code, payload_message, payload_scope = _payload_fields(payload)
    exception_name = exception.__class__.__name__.lower() if exception else ""
    message = payload_message or sanitize_provider_message(exception)
    signal = f"{code} {message}".lower()

    error_type = ProviderErrorType.UNKNOWN_PROVIDER_ERROR
    scope = ErrorScope.UNKNOWN
    retryable = False
    failover = False
    credential_action = account_action = model_action = provider_action = "none"
    reservation_action = "uncertain" if request_sent else "release"

    if "timeout" in exception_name or http_status == 408:
        error_type = (
            ProviderErrorType.TIMEOUT_AFTER_SEND
            if request_sent
            else ProviderErrorType.TIMEOUT_BEFORE_SEND
        )
        scope = ErrorScope.REQUEST
        retryable = not request_sent
        failover = not request_sent
    elif exception is not None and http_status is not None and 200 <= http_status < 300:
        if "empty" in signal:
            error_type = ProviderErrorType.EMPTY_RESPONSE
        elif "malformed" in signal:
            error_type = ProviderErrorType.MALFORMED_RESPONSE
        else:
            error_type = ProviderErrorType.INVALID_RESPONSE
        scope = ErrorScope.REQUEST
    elif "connection" in exception_name:
        error_type = ProviderErrorType.CONNECTION_ERROR
        scope = ErrorScope.PROVIDER
        retryable = True
        failover = True
    elif exception is not None and http_status is None:
        error_type = ProviderErrorType.NETWORK_ERROR
        scope = ErrorScope.PROVIDER
        retryable = True
        failover = True
    elif (
        http_status == 401 or "invalid_api_key" in signal or "authentication" in signal
    ):
        error_type = ProviderErrorType.AUTH_FAILED
        scope = ErrorScope.KEY
        credential_action = "inspect"
        account_action = "quarantine"
        failover = True
    elif http_status == 403:
        error_type = ProviderErrorType.PERMISSION_DENIED
        scope = ErrorScope.ACCOUNT
        account_action = "inspect"
    elif http_status == 402 or "billing" in signal or "payment_required" in signal:
        error_type = ProviderErrorType.BILLING_REQUIRED
        scope = ErrorScope.ORGANIZATION
        account_action = "disable_billing_route"
    elif http_status == 429 and (
        "insufficient_quota" in signal
        or "quota_exhausted" in signal
        or "credit" in signal
    ):
        error_type = ProviderErrorType.QUOTA_EXHAUSTED
        scope = ErrorScope.ACCOUNT
        account_action = "quota_exhausted"
        provider_action = "exclude_for_request"
        failover = True
    elif http_status == 429:
        error_type = ProviderErrorType.RATE_LIMITED_TEMPORARY
        scope = (
            ErrorScope.MODEL
            if "model" in signal or payload_scope == "model"
            else ErrorScope.ACCOUNT
        )
        retryable = True
        failover = True
        if scope is ErrorScope.MODEL:
            model_action = "cooldown"
        else:
            account_action = "cooldown"
            provider_action = "exclude_for_request"
    elif http_status == 404 and "model" in signal:
        error_type = ProviderErrorType.MODEL_NOT_FOUND
        scope = ErrorScope.MODEL
        model_action = "exclude"
        failover = True
    elif http_status in {400, 422} and (
        "mime" in signal or "payload" in signal or "image" in signal
    ):
        error_type = ProviderErrorType.INVALID_PAYLOAD
        scope = ErrorScope.REQUEST
        reservation_action = "provider_policy"
    elif http_status in {400, 422}:
        error_type = ProviderErrorType.INVALID_REQUEST
        scope = ErrorScope.REQUEST
        reservation_action = "provider_policy"
    elif http_status in {500, 502, 503, 504}:
        error_type = (
            ProviderErrorType.PROVIDER_OVERLOADED
            if http_status == 503 or "overload" in signal
            else ProviderErrorType.PROVIDER_UNAVAILABLE
        )
        scope = ErrorScope.PROVIDER
        retryable = True
        failover = True
        provider_action = "record_circuit_failure"
    elif "safety" in signal:
        error_type = ProviderErrorType.SAFETY_REJECTION
        scope = ErrorScope.REQUEST
    elif "refusal" in signal:
        error_type = ProviderErrorType.CONTENT_REFUSAL
        scope = ErrorScope.REQUEST

    retry_after = None
    if error_type is ProviderErrorType.RATE_LIMITED_TEMPORARY:
        header_value = ""
        for key, value in (headers or {}).items():
            if key.lower() == "retry-after":
                header_value = str(value)
                break
        retry_at = parse_retry_after(header_value, now=current)
        retry_after = (
            max(0, int((retry_at - current).total_seconds()))
            if retry_at is not None
            else max(1, default_cooldown_seconds)
        )

    return NormalizedProviderError(
        error_type=error_type,
        provider=provider,
        provider_account_id=provider_account_id,
        provider_model_id=provider_model_id,
        http_status=http_status,
        provider_error_code=code[:120],
        sanitized_message=message,
        retryable=retryable,
        safe_to_failover=failover,
        retry_after_seconds=retry_after,
        scope=scope,
        credential_action=credential_action,
        account_action=account_action,
        model_action=model_action,
        provider_action=provider_action,
        reservation_action=reservation_action,
        request_may_have_been_processed=request_sent,
        raw_error_fingerprint=_fingerprint(provider, http_status, code, message),
        occurred_at=current.isoformat(),
    )
