#!/usr/bin/env bash
# Smoke-test the GLM-5.2 llama.cpp endpoint.
# Usage: /project/inniang/inference/test-glm52.sh [base_url]
set -euo pipefail

ALIAS="glm-5.2"
SERVED_MODEL="glm-5.2"

ENV_FILE=""
for cand in \
    "/project/inniang/inference/logs/current-${ALIAS}-llamacpp.env" \
    "/project/inniang/inference/logs/current-${SERVED_MODEL}-llamacpp.env"; do
    if [ -f "${cand}" ]; then
        ENV_FILE="${cand}"
        break
    fi
done

if [ "$#" -ge 1 ]; then
    BASE_URL="$1"
elif [ -n "${ENV_FILE}" ]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    BASE_URL="http://${LLAMACPP_HOST:-${VLLM_HOST}}:${LLAMACPP_PORT:-${VLLM_PORT}}/v1"
    SERVED_MODEL="${LLAMACPP_SERVED_NAME:-${SERVED_NAME:-${SERVED_MODEL}}}"
else
    printf 'No endpoint env file and no base_url arg.\n' >&2
    exit 1
fi

echo "== Endpoint: ${BASE_URL} =="
echo "--- /v1/models ---"
curl -sS --max-time 30 -H "Authorization: Bearer local" "${BASE_URL}/models" | (jq . 2>/dev/null || cat)

echo
echo "--- /v1/chat/completions (coding prompt) ---"
curl -sS --max-time 300 -H "Authorization: Bearer local" -H "Content-Type: application/json" \
    "${BASE_URL}/chat/completions" -d "{
        \"model\": \"${SERVED_MODEL}\",
        \"messages\": [
            {\"role\": \"user\", \"content\": \"Write a Python function is_prime(n) that returns True iff n is prime. Only output the code.\"}
        ],
        \"temperature\": 0.6,
        \"max_tokens\": 2000,
        \"stream\": false
    }" | (jq -r '.choices[0].message.content // .' 2>/dev/null || cat)
echo
echo "== done =="
