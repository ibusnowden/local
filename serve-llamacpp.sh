#!/usr/bin/env bash
# Usage: sbatch /project/inniang/inference/serve-llamacpp.sh [glm-5.2]
#
# Serves the GLM-5.2 Unsloth 3-bit GGUF (UD-Q3_K_XL) with llama.cpp on one whole
# 8x RTX 6000 Ada node, via the prebuilt ggml-org llama.cpp CUDA server
# container (apptainer). llama.cpp implements GLM-5.2's MLA + DeepSeek sparse
# attention (arch glm-dsa) in portable CUDA that runs on SM89 — the SGLang
# FlashMLA path is Hopper/Blackwell-only. Exposes an OpenAI-compatible API on
# PORT and writes an endpoint env file the LiteLLM router + wire-pi consume.

#SBATCH --job-name=serve-glm52-llamacpp
#SBATCH --partition=bigTiger
# Pinned to itiger06: the other RTX 6000 nodes (e.g. itiger02) lack the user in
# their NSS/passwd, so apptainer aborts with "unknown userid". itiger06 resolves
# the user and runs the container fine (it's the original known-good serve node).
#SBATCH --nodelist=itiger06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=48
# 480G (of 515G; ~20G is MemSpecLimit-reserved): the 3-bit GGUF is ~320 GB and
# --no-mmap reads it fully into RAM, so it needs headroom above the model size
# or it OOMs near the end of load. (The old 2-bit config used 320G.)
#SBATCH --mem=480G
#SBATCH --signal=B:USR1@900
#SBATCH --output=/project/inniang/inference/logs/slurm-serve-llamacpp-%j.out
#SBATCH --error=/project/inniang/inference/logs/slurm-serve-llamacpp-%j.err
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="/project/inniang/inference"
PROFILE="llamacpp"
ALIAS="${1:-glm-5.2}"
MODEL_FILE="${SCRIPT_DIR}/models/${ALIAS}.sh"
SIF="${SIF:-${SCRIPT_DIR}/llama-cpp-server-cuda.sif}"
APPTAINER="${APPTAINER:-$(command -v apptainer || command -v singularity)}"

if [ ! -f "${MODEL_FILE}" ]; then
    printf 'Unknown model alias: %s (no %s)\n' "${ALIAS}" "${MODEL_FILE}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${MODEL_FILE}"

: "${MODEL_GGUF:?config must set MODEL_GGUF}"
: "${SERVED_NAME:?config must set SERVED_NAME}"
: "${PORT:?config must set PORT}"

if [ ! -f "${SIF}" ]; then
    printf 'llama.cpp container not found: %s\n' "${SIF}" >&2
    printf 'Pull it: /project/inniang/inference/pull-llamacpp.sh\n' >&2
    exit 2
fi
if [ ! -f "${MODEL_GGUF}" ]; then
    printf 'GGUF not found: %s\n' "${MODEL_GGUF}" >&2
    printf 'Download it: /project/inniang/inference/download-glm52-gguf.sh\n' >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'Refusing to start: nvidia-smi not available (no GPU on this node).\n' >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "${SCRIPT_DIR}/logs"
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
JOB_ID="${SLURM_JOB_ID:-manual}"
N_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
DIRECT_BASE_URL="http://${NODE_NAME}:${PORT}/v1"
LOCAL_BASE_URL="http://127.0.0.1:${PORT}/v1"

write_endpoint_file() {
    local alias="$1"
    local endpoint_file="${SCRIPT_DIR}/logs/current-${alias}-${PROFILE}.env"
    {
        printf 'LLAMACPP_ALIAS=%q\n' "${alias}"
        printf 'LLAMACPP_SERVED_NAME=%q\n' "${SERVED_NAME}"
        printf 'LLAMACPP_JOB_ID=%q\n' "${JOB_ID}"
        printf 'LLAMACPP_PROFILE=%q\n' "${PROFILE}"
        printf 'LLAMACPP_HOST=%q\n' "${NODE_NAME}"
        printf 'LLAMACPP_PORT=%q\n' "${PORT}"
        printf 'LLAMACPP_GPUS=%q\n' "${N_GPUS}"
        # Generic aliases the LiteLLM router + wire-pi already read.
        printf 'VLLM_HOST=%q\n' "${NODE_NAME}"
        printf 'VLLM_PORT=%q\n' "${PORT}"
        printf 'VLLM_JOB_ID=%q\n' "${JOB_ID}"
        printf 'SERVED_NAME=%q\n' "${SERVED_NAME}"
        printf 'OPENAI_BASE_URL=%q\n' "${DIRECT_BASE_URL}"
        printf 'LOCAL_OPENAI_BASE_URL=%q\n' "${LOCAL_BASE_URL}"
    } > "${endpoint_file}"
}

if declare -p ALIASES >/dev/null 2>&1; then
    for route_alias in "${ALIASES[@]}"; do
        write_endpoint_file "${route_alias}"
    done
else
    write_endpoint_file "${ALIAS}"
fi

printf '[%s] llama.cpp (container) serving %s from %s on %s:%s across %s GPUs\n' \
    "$(date -Iseconds)" "${SERVED_NAME}" "${MODEL_GGUF}" "${NODE_NAME}" "${PORT}" "${N_GPUS}"
printf '[%s] Direct OpenAI base URL: %s\n' "$(date -Iseconds)" "${DIRECT_BASE_URL}"
printf '[%s] ngl=%s n-cpu-moe=%s ctx=%s parallel=%s kv=%s spec=%s\n' \
    "$(date -Iseconds)" "${N_GPU_LAYERS}" "${N_CPU_MOE}" "${CONTEXT_LENGTH}" "${N_PARALLEL}" "${KV_CACHE_TYPE}" "${SPEC_TYPE:-none}"

# llama-server args (the container binary lives at /app/llama-server).
SERVER_ARGS=(
    --model "${MODEL_GGUF}"
    --alias "${SERVED_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --n-gpu-layers "${N_GPU_LAYERS}"
    --ctx-size "${CONTEXT_LENGTH}"
    --parallel "${N_PARALLEL}"
    --batch-size "${BATCH_SIZE}"
    --ubatch-size "${UBATCH_SIZE}"
    --flash-attn "${FLASH_ATTN}"
    --cache-type-k "${KV_CACHE_TYPE}"
    --cache-type-v "${KV_CACHE_TYPE}"
    --jinja
    --metrics
    --no-mmap
    # -fit off disables the device-memory auto-fit probe, which otherwise hangs
    # the load for ~30 min on a model this size (see inference/README.md).
    --fit off
)
if [ "${N_CPU_MOE:-0}" != "0" ]; then
    SERVER_ARGS+=(--n-cpu-moe "${N_CPU_MOE}")
fi
if [ "${SPEC_TYPE:-none}" != "none" ]; then
    SERVER_ARGS+=(--spec-type "${SPEC_TYPE}")
    case ",${SPEC_TYPE}," in
        *,ngram-mod,*)
            SERVER_ARGS+=(
                --spec-ngram-mod-n-match "${SPEC_NGRAM_MOD_N_MATCH:-24}"
                --spec-ngram-mod-n-min "${SPEC_NGRAM_MOD_N_MIN:-48}"
                --spec-ngram-mod-n-max "${SPEC_NGRAM_MOD_N_MAX:-64}"
            )
            ;;
    esac
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    SERVER_ARGS+=("${EXTRA[@]}")
fi

# NATIVE=1 runs the from-source CUDA build instead of the container. Required
# for --spec-type draft-mtp on GLM-5.2: the glm-dsa DECODER_MTP graph is a
# local port (llama.cpp commit ebd048f + local patches), not in the prebuilt
# container image.
NATIVE_BIN="${NATIVE_BIN:-${SCRIPT_DIR}/llama.cpp/build-cuda/bin/llama-server}"
if [ "${NATIVE:-0}" = "1" ]; then
    if [ ! -x "${NATIVE_BIN}" ]; then
        printf 'NATIVE=1 but native build not found: %s\n' "${NATIVE_BIN}" >&2
        exit 2
    fi
    LAUNCH_CMD=("${NATIVE_BIN}" "${SERVER_ARGS[@]}")
else
    # The container's llama-server needs /app on its library path.
    export APPTAINERENV_LD_LIBRARY_PATH="/app"
    LAUNCH_CMD=("${APPTAINER}" exec --nv
        --bind "${SCRIPT_DIR}"
        "${SIF}"
        /app/llama-server "${SERVER_ARGS[@]}")
fi

# The devs association caps jobs at MaxWall=2-00:00:00, so a single job cannot
# serve indefinitely. SLURM sends USR1 to this shell 900s before the limit
# (--signal=B:USR1@900); we submit a successor job that pends on the same node
# and takes over when this one is killed. RESUBMIT=0 breaks the chain.
# Absolute path required: inside a job this script runs from the SLURM spool dir.
SCRIPT_PATH="/project/inniang/inference/serve-llamacpp.sh"
resubmit_successor() {
    printf '[%s] wall-time limit approaching — submitting successor job\n' "$(date -Iseconds)"
    sbatch "${SCRIPT_PATH}" || printf 'successor submission FAILED\n' >&2
}
if [ "${RESUBMIT:-1}" = "1" ]; then
    trap resubmit_successor USR1
fi

"${LAUNCH_CMD[@]}" &
SERVER_PID=$!
rc=0
while :; do
    if wait "${SERVER_PID}"; then rc=0; else rc=$?; fi
    # wait returns >128 when interrupted by the trapped USR1; loop back to
    # keep waiting on the still-running server. Exit only when it is gone.
    kill -0 "${SERVER_PID}" 2>/dev/null || break
done
exit "${rc}"
