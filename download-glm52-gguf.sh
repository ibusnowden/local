#!/usr/bin/env bash
# Download the GLM-5.2 GGUF (Unsloth dynamic 3-bit, UD-Q3_K_XL) for llama.cpp.
#
# UD-Q3_K_XL is ~343 GB across 9 split parts. It fits fully on one 8x RTX 6000
# Ada node (384 GB VRAM) with ~41 GB headroom for KV cache + buffers.
#
# Usage: /project/inniang/inference/download-glm52-gguf.sh
#   QUANT=UD-Q3_K_M  ./download-glm52-gguf.sh   # smaller 3-bit alternative
#   QUANT=UD-Q2_K_XL ./download-glm52-gguf.sh   # 2-bit fallback (~254 GB)
set -euo pipefail

REPO_ID="${REPO_ID:-unsloth/GLM-5.2-GGUF}"
QUANT="${QUANT:-UD-Q3_K_XL}"
LOCAL_DIR="${LOCAL_DIR:-/project/inniang/inference/models/GLM-5.2-GGUF}"
HF_HOME="${HF_HOME:-/project/inniang/inference/.hf-cache}"
HF_CLI="${HF_CLI:-}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

if [ -z "${HF_CLI}" ]; then
    if [ -x /project/inniang/.venv/bin/hf ]; then
        HF_CLI="/project/inniang/.venv/bin/hf"
    elif command -v hf >/dev/null 2>&1; then
        HF_CLI="$(command -v hf)"
    else
        printf 'Hugging Face CLI not found. Install with: python -m pip install -U huggingface_hub\n' >&2
        exit 2
    fi
fi

mkdir -p "${LOCAL_DIR}" "${HF_HOME}"
export HF_HOME

printf '[%s] Downloading %s [%s] to %s\n' "$(date -Iseconds)" "${REPO_ID}" "${QUANT}" "${LOCAL_DIR}"

"${HF_CLI}" download "${REPO_ID}" \
    --include "${QUANT}/*" \
    --max-workers "${HF_MAX_WORKERS}" \
    --local-dir "${LOCAL_DIR}"

FIRST_SHARD="$(ls "${LOCAL_DIR}/${QUANT}"/*-00001-of-*.gguf 2>/dev/null | head -n1 || true)"
printf '[%s] Download complete.\n' "$(date -Iseconds)"
if [ -n "${FIRST_SHARD}" ]; then
    printf '[%s] First shard (point serve config MODEL_GGUF here): %s\n' "$(date -Iseconds)" "${FIRST_SHARD}"
else
    printf '[%s] WARNING: no -00001-of- shard found under %s/%s\n' "$(date -Iseconds)" "${LOCAL_DIR}" "${QUANT}" >&2
fi
