from __future__ import annotations

import json
import re
import uuid
from typing import Any

from ..database import utc_now

SENSITIVE_KEY_RE = re.compile(
    r"(api[-_]?key|token|authorization|secret|password|credential|signed)",
    re.IGNORECASE,
)
CONTENT_KEY_RE = re.compile(
    r"(base64|image|document[-_]?text|full[-_]?text|payload|prompt|content)",
    re.IGNORECASE,
)
DATA_URL_RE = re.compile(
    r"data:(?:image|application)/[^;\s]+;base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)
SIGNED_QUERY_RE = re.compile(
    r"(?i)([?&](?:token|api_key|signature|credential)=)[^&\s]+"
)


def sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[redacted]"
                if SENSITIVE_KEY_RE.search(str(key)) or CONTENT_KEY_RE.search(str(key))
                else sanitize_metadata(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value[:100]]
    if isinstance(value, str):
        clean = value.replace("\r", "\\r").replace("\n", "\\n")
        clean = DATA_URL_RE.sub("[redacted-payload]", clean)
        clean = SIGNED_QUERY_RE.sub(r"\1[redacted]", clean)
        if SENSITIVE_KEY_RE.search(clean):
            return "[redacted]"
        return clean[:500]
    return value


def audit_event(
    *,
    event_type: str,
    provider: str = "",
    pool_id: str = "",
    account_id: str = "",
    model: str = "",
    decision_reason: str = "",
    old_state: str = "",
    new_state: str = "",
    request_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "provider": provider,
        "pool_id": pool_id,
        "account_id": account_id,
        "model": model,
        "decision_reason": decision_reason,
        "old_state": old_state,
        "new_state": new_state,
        "request_id": request_id,
        "created_at": utc_now(),
        "metadata_json": json.dumps(
            sanitize_metadata(metadata or {}), ensure_ascii=False, sort_keys=True
        ),
    }
