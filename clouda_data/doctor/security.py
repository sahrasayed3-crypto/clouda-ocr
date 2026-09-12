"""Secret-safety helpers for the Environment Doctor.

The Doctor may report *whether* environment variables are set, never their
values. Variable names come from a curated allowlist discovered from the
canonical repository (runtime config, storage roots, server, worker, model
providers). Anything matching credential-ish patterns is redacted even if
someone later extends the list.

``build_env_report`` is the single choke point every env-var check must use,
and its output is audited by tests/security-style tests in tests/doctor.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from clouda_contracts.security import redact_text as _redact_text
from clouda_contracts.security import redact_value as _redact_value

# Names discovered by scanning os.environ/getenv usage across the canonical
# repository (clouda_data, clouda_training, clouda_contracts, clouda_models,
# pdfword, tools). Only SET/NOT-SET is ever reported for these.
RELEVANT_ENV_VARS: tuple[str, ...] = (
    # project runtime / storage
    "CLOUDA_PROJECT_ROOT",
    "CLOUDA_STATE_HOME",
    "CLOUDA_RUNTIME_ROOT",
    "CLOUDA_DATASET_ROOT",
    "CLOUDA_ARTIFACT_ROOT",
    "CLOUDA_MODEL_ROOT",
    "CLOUDA_CACHE_ROOT",
    "CLOUDA_DATABASE_PATH",
    "CLOUDA_DATASET_MANIFEST_DATABASE_PATH",
    "DATABASE_PATH",
    "STORAGE_ROOT",
    "TEMP_ROOT",
    # data foundation / factory
    "CLOUDA_MAX_IMAGE_PIXELS",
    "CLOUDA_MAX_PDF_BYTES",
    "CLOUDA_MAX_PDF_PAGES",
    # runtime server
    "CLOUDA_ENV",
    "CLOUDA_ALLOWED_HOSTS",
    "APP_ROLE",
    "AUTH_MODE",
    "SERVER_BASE_URL",
    "WEB_CONCURRENCY",
    "FIREBASE_PROJECT_ID",
    "FIREBASE_AUTH_EMULATOR_HOST",
    # worker
    "REDIS_URL",
    "RQ_QUEUE_NAME",
    "WORKER_NAME",
    "WORKER_CONCURRENCY",
    "JOB_TIMEOUT_SECONDS",
    "JOB_RETRY_COUNT",
    # model providers / inference (secrets: SET-only)
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "FIREBASE_CLIENT_API_KEY",
    "WORKER_API_KEY",
    "CLOUDA_WORKER_API_KEY_PREVIOUS",
    "FIREBASE_SERVICE_ACCOUNT_JSON_PATH",
    # local OCR runner
    "CLOUDA_LOCAL_OCR_ENABLED",
    "CLOUDA_LOCAL_OCR_COMMAND",
    "CLOUDA_LOCAL_OCR_ALLOWED_EXECUTABLES",
    "CLOUDA_ALLOW_MOCK_OCR",
    # tooling
    "GITHUB_ACTIONS",
    "PYTHONUSERBASE",
    "PYTHONSTARTUP",
)

# Fallback redaction net: any name matching these patterns is treated as
# secret-bearing even if it appears in future allowlist extensions.
_SECRET_NAME_PATTERN = re.compile(
    r"(?i)key|token|secret|password|passwd|credential|auth|api[-_]?key|"
    r"service[-_]?account|session"
)
_SECRET_ENV_NAMES = {"REDIS_URL"}


def is_secret_var(name: str) -> bool:
    """True when a variable name pattern-matches a credential class."""
    return name in _SECRET_ENV_NAMES or bool(_SECRET_NAME_PATTERN.search(name))


def redact_text(text: str, environment: Mapping[str, str] | None = None) -> str:
    """Use the canonical contracts-layer diagnostic redaction policy."""
    return _redact_text(text, environment)


def redact_value(value: Any) -> Any:
    """Use the canonical contracts-layer recursive redaction policy."""
    return _redact_value(value)


def build_env_report(environment: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Return allowlisted env-var entries with SET/NOT-SET only.

    The value is read to decide set-ness and then discarded; it never appears
    in the returned structure.
    """
    env = os.environ if environment is None else environment
    entries: list[dict[str, Any]] = []
    for name in RELEVANT_ENV_VARS:
        raw = env.get(name)
        if raw is None:
            raw = None  # not present
        entries.append(
            {
                "name": name,
                "set": raw is not None and raw != "",
                "secret": is_secret_var(name),
            }
        )
    return entries
