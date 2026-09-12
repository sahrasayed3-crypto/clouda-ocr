from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|(?:^|[_-])key(?:$|[_-])|token|secret|password|passwd|"
    r"cookie|authorization|"
    r"credential|service[_-]?account|session|redis_url)",
    re.IGNORECASE,
)
_XML_FORBIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff]")
_BIDI_OVERRIDE = re.compile("[\u202a-\u202e\u2066-\u2069]")
_SPREADSHEET_PREFIX = ("=", "+", "-", "@")


def redact_text(text: str, environment: Mapping[str, str] | None = None) -> str:
    """Remove configured credential values from diagnostic text."""
    env = os.environ if environment is None else environment
    redacted = text
    for name, raw in env.items():
        if not _SENSITIVE_KEY.search(name) or not isinstance(raw, str) or len(raw) < 4:
            continue
        redacted = redacted.replace(raw, "<redacted>")
    return redacted


def redact_value(value: Any, environment: Mapping[str, str] | None = None) -> Any:
    """Recursively redact configured secrets in a diagnostic payload."""
    if isinstance(value, str):
        return redact_text(value, environment)
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(str(key))
                else redact_value(item, environment)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact_value(item, environment) for item in value]
    return value


def redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive keys and configured secret values recursively."""
    return {
        key: ("[REDACTED]" if _SENSITIVE_KEY.search(key) else redact_value(item))
        for key, item in value.items()
    }


def sanitize_document_text(value: object, *, max_characters: int = 5_000_000) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    if len(text) > max_characters:
        raise ValueError("Document text exceeds the configured character limit")
    return _BIDI_OVERRIDE.sub("", _XML_FORBIDDEN.sub("", text))


def sanitize_spreadsheet_cell(value: object) -> object:
    if not isinstance(value, str):
        return value
    cleaned = sanitize_document_text(value, max_characters=100_000)
    if cleaned.lstrip().startswith(_SPREADSHEET_PREFIX) or cleaned.startswith(
        ("\t", "\r", "\n")
    ):
        return "'" + cleaned
    return cleaned


def may_use_user_document_for_training(
    *,
    explicit_document_consent: bool,
    approved_consent_policy: bool,
) -> bool:
    """Require two independent approvals; both default false at all call sites."""
    return explicit_document_consent and approved_consent_policy
