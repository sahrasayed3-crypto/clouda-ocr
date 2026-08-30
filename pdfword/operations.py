from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?:authorization|cookie|password|secret|token|api[_-]?key)",
    re.IGNORECASE,
)


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SENSITIVE_FIELD_PATTERN.search(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def structured_log(event: str, **fields) -> None:
    logging.getLogger("clouda.operations").info(
        json.dumps(
            {"event": event, **redact(fields)}, ensure_ascii=False, sort_keys=True
        )
    )


class SecretSource(Protocol):
    def get(self, name: str) -> str | None: ...


class EnvironmentSecretSource:
    def get(self, name: str) -> str | None:
        value = os.getenv(name)
        return value if value else None


class FileSecretSource:
    """Read a secret from an explicitly configured external file."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def get(self, name: str) -> str | None:
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid secret name")
        path = (self.root / name).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("Secret path escaped root") from exc
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8").strip()


class IdentityProvider(Protocol):
    def authenticate(self, authorization_header: str) -> dict[str, str]: ...


class ExternalOIDCRequired:
    def authenticate(self, authorization_header: str) -> dict[str, str]:
        del authorization_header
        raise RuntimeError(
            "Production OIDC is not configured. Keep the service private."
        )


@dataclass(frozen=True)
class RedisSecurityConfig:
    url: str
    tls_required: bool
    certificate_reqs: str
    health_check_interval: int
    retry_on_timeout: bool

    @classmethod
    def from_env(cls) -> "RedisSecurityConfig":
        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        tls_required = os.getenv("CLOUDA_REDIS_TLS_REQUIRED", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if tls_required and not url.startswith("rediss://"):
            raise ValueError("Redis TLS is required but REDIS_URL is not rediss://")
        return cls(
            url=url,
            tls_required=tls_required,
            certificate_reqs=os.getenv("CLOUDA_REDIS_CERT_REQS", "required"),
            health_check_interval=max(
                5, int(os.getenv("CLOUDA_REDIS_HEALTH_INTERVAL", "30"))
            ),
            retry_on_timeout=True,
        )

    def client_kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "decode_responses": True,
            "socket_connect_timeout": 2,
            "socket_timeout": 2,
            "health_check_interval": self.health_check_interval,
            "retry_on_timeout": self.retry_on_timeout,
        }
        if self.url.startswith("rediss://"):
            values["ssl_cert_reqs"] = self.certificate_reqs
        return values


class SlidingWindowRateLimiter:
    def __init__(self, limit: int = 120, window_seconds: int = 60) -> None:
        self.limit = max(1, limit)
        self.window_seconds = max(1, window_seconds)
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(current)
            return True


class OperationsMetrics:
    def __init__(self) -> None:
        self.requests: Counter[tuple[str, int]] = Counter()
        self._lock = threading.Lock()

    def observe_request(self, method: str, status: int) -> None:
        with self._lock:
            self.requests[(method, status)] += 1

    def prometheus(self, *, queue_depth: int = 0, failed_jobs: int = 0) -> str:
        lines = [
            "# HELP clouda_http_requests_total HTTP requests.",
            "# TYPE clouda_http_requests_total counter",
        ]
        with self._lock:
            for (method, status), count in sorted(self.requests.items()):
                lines.append(
                    f'clouda_http_requests_total{{method="{method}",status="{status}"}} {count}'
                )
        lines.extend(
            [
                "# HELP clouda_queue_depth Current queue depth.",
                "# TYPE clouda_queue_depth gauge",
                f"clouda_queue_depth {queue_depth}",
                "# HELP clouda_failed_jobs Current failed job count.",
                "# TYPE clouda_failed_jobs gauge",
                f"clouda_failed_jobs {failed_jobs}",
            ]
        )
        return "\n".join(lines) + "\n"


def credential_matches(provided: str | None, current: str, previous: str = "") -> bool:
    return bool(
        provided
        and any(
            hmac.compare_digest(provided, candidate)
            for candidate in (current, previous)
            if candidate
        )
    )
