#!/usr/bin/env bash
# Point the `pi` coding agent at the live GLM-5.2 llama.cpp endpoint.
#
# Usage: /project/inniang/inference/wire-pi-glm52.sh [--set-default]
set -euo pipefail

ALIAS="glm-5.2"
SERVED_MODEL="glm-5.2"
MODELS_JSON="${HOME}/.pi/agent/models.json"

ENV_FILE=""
for cand in \
    "/project/inniang/inference/logs/current-${ALIAS}-llamacpp.env" \
    "/project/inniang/inference/logs/current-${SERVED_MODEL}-llamacpp.env"; do
    if [ -f "${cand}" ]; then
        ENV_FILE="${cand}"
        break
    fi
done

if [ -z "${ENV_FILE}" ]; then
    printf 'No llama.cpp endpoint file found. Start it with:\n' >&2
    printf '  sbatch /project/inniang/inference/serve-llamacpp.sh glm-5.2\n' >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"
HOST="${LLAMACPP_HOST:-${VLLM_HOST:?}}"
PORT="${LLAMACPP_PORT:-${VLLM_PORT:?}}"
BASE_URL="http://${HOST}:${PORT}/v1"
SERVED_MODEL="${LLAMACPP_SERVED_NAME:-${SERVED_NAME:-${SERVED_MODEL}}}"

mkdir -p "$(dirname "${MODELS_JSON}")"

/project/inniang/.venv/bin/python - "$MODELS_JSON" "$BASE_URL" "$SERVED_MODEL" <<'PY'
import json, sys
path, base_url, served_model = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as f:
        cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}
cfg.setdefault("providers", {})
cfg["providers"]["glm-5.2"] = {
    "name": "GLM-5.2 (llama.cpp 3-bit GGUF)",
    "baseUrl": base_url,
    "api": "openai-completions",
    "apiKey": "local",
    "compat": {
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
    },
    "models": [
        {
            "id": served_model,
            "name": "GLM-5.2 (Q3_K_XL)",
            "reasoning": True,
            "input": ["text"],
            # Must match the server's PER-SLOT context, not the total KV budget.
            # serve-llamacpp.sh runs CONTEXT_LENGTH=262144 over N_PARALLEL=1
            # slot, so the per-slot context is the full 262144 (256k). This
            # matches pincode's LOCAL_CONTEXT_WINDOW.
            "contextWindow": 262144,
            "maxTokens": 16384,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
    ],
}
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote provider 'glm-5.2' model {served_model} -> {base_url} into {path}")
PY

if [ "${1:-}" = "--set-default" ]; then
    SETTINGS="${HOME}/.pi/agent/settings.json"
    /project/inniang/.venv/bin/python - "$SETTINGS" "$SERVED_MODEL" <<'PY'
import json, sys
path, served_model = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        s = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    s = {}
s["defaultProvider"] = "glm-5.2"
s["defaultModel"] = served_model
with open(path, "w") as f:
    json.dump(s, f, indent=2)
print(f"Set default provider/model to glm-5.2/{served_model} in {path}")
PY
fi

printf '\nDone. In pi: /model -> GLM-5.2, or run:\n  bun %s --provider glm-5.2 --model %s\n' \
    "/home/inniang/.bun/install/global/node_modules/@earendil-works/pi-coding-agent/dist/cli.js" \
    "${SERVED_MODEL}"
