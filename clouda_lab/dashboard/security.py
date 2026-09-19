from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from clouda_contracts.security import redact_value
from clouda_contracts.storage import validate_relative_components

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,255}$")
_OUTPUT_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")
_INLINE_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*[^\s,;]+"
)
_EMBEDDED_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/][^\s,;\)\]\}]+|/(?:[^/\s]+/)+[^\s,;\)\]\}]+)"
)


def safe_identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("invalid canonical identifier")
    return value


def safe_output_label(value: str) -> str:
    if not isinstance(value, str) or not _OUTPUT_LABEL.fullmatch(value):
        raise ValueError("invalid output label")
    validate_relative_components(Path(value))
    return value


def safe_relative_label(path: str | Path, roots: tuple[Path, ...]) -> str | None:
    candidate = Path(path).expanduser().resolve(strict=False)
    for root in roots:
        resolved_root = root.expanduser().resolve(strict=False)
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError:
            continue
        return relative.as_posix() or "."
    return None


def sanitize_payload(value: Any) -> Any:
    redacted = redact_value(value)
    if isinstance(redacted, str):
        return _INLINE_SECRET.sub(
            lambda match: f"{match.group(1)}=[REDACTED]", redacted
        )
    if isinstance(redacted, dict):
        return {str(key): sanitize_payload(item) for key, item in redacted.items()}
    if isinstance(redacted, list):
        return [sanitize_payload(item) for item in redacted]
    return redacted


def browser_safe(value: Any, roots: tuple[Path, ...]) -> Any:
    """Redact secrets and replace absolute paths before browser serialization."""

    cleaned = sanitize_payload(value)
    if isinstance(cleaned, Path):
        return safe_relative_label(cleaned, roots) or "[PRIVATE PATH]"
    if isinstance(cleaned, dict):
        return {str(key): browser_safe(item, roots) for key, item in cleaned.items()}
    if isinstance(cleaned, list):
        return [browser_safe(item, roots) for item in cleaned]
    if isinstance(cleaned, str):
        candidate = Path(cleaned)
        if candidate.is_absolute():
            return safe_relative_label(candidate, roots) or "[PRIVATE PATH]"
        text = cleaned
        for root in roots:
            for raw in (str(root), root.as_posix()):
                text = text.replace(raw, ".")
        return _EMBEDDED_ABSOLUTE_PATH.sub("[PRIVATE PATH]", text)
    return cleaned


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in {"testclient", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_loopback(request: Request) -> None:
    settings = request.app.state.lab_settings
    if settings.local_only and not is_loopback(
        request.client.host if request.client else None
    ):
        raise HTTPException(status_code=403, detail="Clouda Lab is loopback-only")
