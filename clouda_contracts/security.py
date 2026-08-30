from __future__ import annotations

import re
import unicodedata
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|token|secret|password|cookie|authorization)",
    re.IGNORECASE,
)
_XML_FORBIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff]")
_BIDI_OVERRIDE = re.compile("[\u202a-\u202e\u2066-\u2069]")
_SPREADSHEET_PREFIX = ("=", "+", "-", "@")


def redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            "[REDACTED]"
            if _SENSITIVE_KEY.search(key)
            else redact_mapping(item) if isinstance(item, dict) else item
        )
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
