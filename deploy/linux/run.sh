#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${CLOUDA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PORT="${CLOUDA_PORT:-8501}"
BIND_ADDRESS="${CLOUDA_BIND_ADDRESS:-127.0.0.1}"
if [[ "${BIND_ADDRESS}" != "127.0.0.1" && "${BIND_ADDRESS}" != "::1" ]]; then
  if [[ "${CLOUDA_ALLOW_PUBLIC_BIND:-false}" != "true" || "${CLOUDA_EXTERNAL_AUTH_ENFORCED:-false}" != "true" ]]; then
    echo "Public binding requires explicit opt-in and enforced external authentication." >&2
    exit 1
  fi
fi
cd "${PROJECT_DIR}"
exec "${PROJECT_DIR}/.venv/bin/python" -m streamlit run app.py \
  --server.headless=true \
  --server.address="${BIND_ADDRESS}" \
  --server.port="${PORT}" \
  --browser.gatherUsageStats=false
