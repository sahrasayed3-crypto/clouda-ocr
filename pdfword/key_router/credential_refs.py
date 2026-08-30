from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

SECRET_REF_RE = re.compile(r"^(?:env:)?[A-Z][A-Z0-9_]{0,127}$")


def validate_secret_ref(secret_ref: str) -> str:
    value = (secret_ref or "").strip()
    if not SECRET_REF_RE.fullmatch(value):
        raise ValueError("Invalid secret reference")
    return value


def secret_env_name(secret_ref: str) -> str:
    value = validate_secret_ref(secret_ref)
    return value[4:] if value.startswith("env:") else value


def redact_secret_ref(secret_ref: str) -> str:
    value = (secret_ref or "").strip()
    if not value:
        return ""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[-8:]
    return f"env:***:{digest}" if value.startswith("env:") else f"***:{digest}"


@dataclass(frozen=True)
class SecretHandle:
    secret_ref: str
    _value: str = field(repr=False)

    @property
    def value(self) -> str:
        return self._value

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self._value.encode("utf-8")).hexdigest()[-8:]

    def __repr__(self) -> str:
        return (
            "SecretHandle("
            f"secret_ref={redact_secret_ref(self.secret_ref)!r}, "
            f"fingerprint={self.fingerprint!r})"
        )

    __str__ = __repr__


class EnvSecretResolver:
    def get(self, secret_ref: str) -> SecretHandle | None:
        ref = validate_secret_ref(secret_ref)
        value = os.getenv(secret_env_name(ref), "")
        if not value.strip():
            return None
        return SecretHandle(secret_ref=ref, _value=value.strip())
