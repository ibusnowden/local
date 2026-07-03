#!/usr/bin/env bash
# Regenerate litellm.config.yaml from logs/current-<alias>-<profile>.env files.
# Each env file records the live job's host + port for a model, so the router
# config tracks whichever compute node SLURM landed each model on. Removes the
# need for SSH tunnels when the login node has direct network access to the
# compute nodes (which is the normal case here).
#
# Supports the llama.cpp GLM-5.2 GGUF profile. Env files expose VLLM_HOST /
# VLLM_PORT aliases so older router consumers do not need special handling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${INFERENCE_DIR}/logs"
OUT="${SCRIPT_DIR}/litellm.config.yaml"

# Models to route. Add an alias here and give it a models/<alias>.sh + a
# current-<alias>-<profile>.env to make it routable.
MODELS=(glm-5.2)

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

# Read a variable from an endpoint env file.
envval() {
  local envf="$1" key="$2"
  awk -F= -v k="${key}" '$1==k{print $2; exit}' "${envf}" | tr -d "\"'"
}

{
  printf 'model_list:\n'
  for alias in "${MODELS[@]}"; do
    envf=""
    for cand in \
      "${LOGS_DIR}/current-${alias}-llamacpp.env" \
      "${LOGS_DIR}/current-${alias}-ktransformers.env"; do
      [ -f "${cand}" ] || continue
      jobid="$(envval "${cand}" LLAMACPP_JOB_ID)"
      [ -n "${jobid}" ] || jobid="$(envval "${cand}" KTRANSFORMERS_JOB_ID)"
      [ -n "${jobid}" ] || jobid="$(envval "${cand}" VLLM_JOB_ID)"
      [ -n "${jobid}" ] || continue
      state="$(squeue -h -j "${jobid}" -o '%T' 2>/dev/null | head -n1 || true)"
      if [ "${state}" = "RUNNING" ]; then envf="${cand}"; break; fi
    done
    if [ -z "${envf}" ]; then
      printf '  # %s — no RUNNING job found, skipped\n' "${alias}" >&2
      continue
    fi
    host="$(envval "${envf}" VLLM_HOST)"
    [ -n "${host}" ] || host="$(envval "${envf}" KTRANSFORMERS_HOST)"
    port="$(envval "${envf}" VLLM_PORT)"
    [ -n "${port}" ] || port="$(envval "${envf}" KTRANSFORMERS_PORT)"
    served="$(envval "${envf}" LLAMACPP_SERVED_NAME)"
    [ -n "${served}" ] || served="$(envval "${envf}" KTRANSFORMERS_SERVED_NAME)"
    [ -n "${served}" ] || served="$(envval "${envf}" SERVED_NAME)"
    [ -n "${served}" ] || served="${alias}"
    printf '  - model_name: %s\n' "${alias}"
    printf '    litellm_params:\n'
    printf '      model: openai/%s\n' "${served}"
    printf '      api_base: http://%s:%s/v1\n' "${host}" "${port}"
    printf '      api_key: local\n'
  done
  printf '\n'
  printf 'general_settings:\n'
  printf '  master_key: local\n\n'
  printf 'litellm_settings:\n'
  printf '  drop_params: true\n'
  printf '  request_timeout: 600\n'
} > "${tmp}"

mv "${tmp}" "${OUT}"
trap - EXIT
printf 'wrote %s\n' "${OUT}"
