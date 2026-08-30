#!/usr/bin/env bash
set -euo pipefail
PORT="${CLOUDA_PORT:-8501}"
curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:${PORT}/_stcore/health"
