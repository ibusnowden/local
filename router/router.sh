#!/usr/bin/env bash
# Start the LiteLLM proxy that fans requests out to per-model vLLM backends.
# Listens on 127.0.0.1:4000 by default. Point fabric/harness at:
#   LOCAL_BASE_URL=http://127.0.0.1:4000/v1
#
# Usage:
#   ./router.sh               # foreground, default port 4000
#   ./router.sh --port 4100   # override port
#   PORT=4100 ./router.sh     # same via env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/litellm.config.yaml"
PORT="${PORT:-4000}"
HOST="${HOST:-127.0.0.1}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:-}"; shift 2 ;;
        --port=*) PORT="${1#--port=}"; shift ;;
        --host) HOST="${2:-}"; shift 2 ;;
        --host=*) HOST="${1#--host=}"; shift ;;
        -h|--help)
            printf 'Usage: %s [--host HOST] [--port PORT]\n' "$0"
            exit 0
            ;;
        *) printf 'Unknown flag: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [ -f "/project/inniang/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "/project/inniang/.venv/bin/activate"
fi

if ! command -v litellm >/dev/null 2>&1; then
    printf 'litellm proxy not installed. Run: pip install "litellm[proxy]>=1.50"\n' >&2
    exit 2
fi

printf '[%s] Starting LiteLLM proxy on %s:%s (config=%s)\n' "$(date -Iseconds)" "${HOST}" "${PORT}" "${CONFIG}"
exec litellm --config "${CONFIG}" --host "${HOST}" --port "${PORT}"
