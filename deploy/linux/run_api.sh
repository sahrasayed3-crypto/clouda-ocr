#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${CLOUDA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
API_PORT="${CLOUDA_API_PORT:-8000}"
BIND_ADDRESS="${CLOUDA_API_BIND_ADDRESS:-127.0.0.1}"
if [[ "${BIND_ADDRESS}" != "127.0.0.1" && "${BIND_ADDRESS}" != "::1" ]]; then
  if [[ "${CLOUDA_ALLOW_PUBLIC_BIND:-false}" != "true" ]]; then
    echo "Non-loopback worker API binding requires explicit opt-in." >&2
    exit 1
  fi
fi
cd "${PROJECT_DIR}"
exec "${PROJECT_DIR}/.venv/bin/python" -m uvicorn pdfword.worker_api:app \
  --host="${BIND_ADDRESS}" --port="${API_PORT}" --workers=1
