from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from clouda_training.experiments.io import atomic_write_json, read_json

from .security import browser_safe, safe_identifier

CONFIRMATION_SCHEMA_VERSION = "clouda.lab.confirmation.v1"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConfirmationStore:
    """Persistent, expiring, single-use confirmation plans."""

    def __init__(self, root: Path, *, browser_roots: tuple[Path, ...]) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.browser_roots = tuple(path.resolve() for path in browser_roots)

    def _path(self, plan_id: str) -> Path:
        return self.root / f"{safe_identifier(plan_id)}.json"

    def issue(
        self,
        kind: str,
        target_id: str,
        details: dict[str, Any],
        *,
        expires_minutes: int = 15,
    ) -> dict[str, Any]:
        safe_identifier(target_id)
        plan_id = f"confirm-{secrets.token_hex(16)}"
        created = _now()
        payload = {
            "schema_version": CONFIRMATION_SCHEMA_VERSION,
            "plan_id": plan_id,
            "kind": kind,
            "target_id": target_id,
            **details,
            "confirmation_token": secrets.token_urlsafe(24),
            "created_at": created.isoformat(),
            "expires_at": (created + timedelta(minutes=expires_minutes)).isoformat(),
            "used": False,
        }
        atomic_write_json(self._path(plan_id), payload)
        safe_payload = browser_safe(payload, self.browser_roots)
        # This random, short-lived, single-use value is deliberately returned
        # to the same loopback client that requested the confirmation plan.
        # Restore only this field after generic secret redaction; operation
        # results and diagnostic payloads continue to redact token-like keys.
        safe_payload["confirmation_token"] = payload["confirmation_token"]
        return safe_payload

    def consume(self, plan_id: str, token: str, *, kind: str) -> dict[str, Any]:
        path = self._path(plan_id)
        if not path.is_file():
            raise KeyError(f"Unknown confirmation plan: {plan_id}")
        payload = read_json(path)
        if payload.get("kind") != kind:
            raise PermissionError("confirmation plan operation mismatch")
        if payload.get("used"):
            raise PermissionError("confirmation plan was already used")
        if _now() >= datetime.fromisoformat(str(payload["expires_at"])):
            raise PermissionError("confirmation plan expired")
        if not token or not secrets.compare_digest(
            str(payload.get("confirmation_token", "")), token
        ):
            raise PermissionError("confirmation token does not match")
        payload["used"] = True
        payload["used_at"] = _now().isoformat()
        atomic_write_json(path, payload)
        return payload
