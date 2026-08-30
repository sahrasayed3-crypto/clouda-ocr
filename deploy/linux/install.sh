#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -eq 0 ]]; then
  echo "Run this installer as the dedicated application user, not root." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sudo apt-get update
sudo apt-get install -y \
  python3.11 python3.11-venv python3-pip redis-server curl

python3.11 -m venv "${PROJECT_DIR}/.venv"
"${PROJECT_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${PROJECT_DIR}/.venv/bin/pip" install -r "${PROJECT_DIR}/requirements-server.txt"
mkdir -p "${PROJECT_DIR}/data" "${PROJECT_DIR}/conversions" "${PROJECT_DIR}/logs"
echo "Install complete. Copy clouda.env.example to clouda.env and install the systemd unit."
