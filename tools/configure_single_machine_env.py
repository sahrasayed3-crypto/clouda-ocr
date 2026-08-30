"""Safely adapt an existing .env file for single-machine Windows operation."""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED = {
    "APP_ROLE": "server",
    "LOCAL_PROCESSING_ENABLED": "true",
    "REDIS_URL": "redis://127.0.0.1:6379/0",
    "RQ_QUEUE_NAME": "pdf_conversion",
    "SERVER_BASE_URL": "http://127.0.0.1:8000",
    "WORKER_CONCURRENCY": "1",
    "WORKER_NAME": "clouda-local",
    "DATABASE_PATH": "data/clouda.sqlite3",
    "STORAGE_ROOT": "conversions",
    "TEMP_ROOT": "conversions/temporary",
    "JOB_TIMEOUT_SECONDS": "7200",
    "JOB_RETRY_COUNT": "2",
}


def configure(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    found: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in REQUIRED:
            output.append(f"{key}={REQUIRED[key]}")
            found.add(key)
        else:
            output.append(line)
    if output and output[-1]:
        output.append("")
    for key, value in REQUIRED.items():
        if key not in found:
            output.append(f"{key}={value}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return sorted(REQUIRED)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".env")
    changed = configure(target)
    print("Configured variable names: " + ", ".join(changed))
