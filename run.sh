#!/usr/bin/env bash
# nanobot launcher — uses project .venv
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$HOME/.nanobot/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.nanobot/.env"
  set +a
fi

exec "$SCRIPT_DIR/.venv/bin/nanobot" "$@"
