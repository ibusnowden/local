#!/usr/bin/env bash
# Pull the prebuilt ggml-org llama.cpp CUDA server container for serving on the
# RTX 6000 Ada nodes. This replaces a from-source build: the image ships its own
# CUDA runtime + a llama-server at /app/llama-server, and recent images include
# native support for GLM-5.2's `glm-dsa` architecture.
#
# Usage: /project/inniang/inference/pull-llamacpp.sh   (run on a login node w/ internet)
set -euo pipefail

SCRIPT_DIR="/project/inniang/inference"
IMAGE="${IMAGE:-docker://ghcr.io/ggml-org/llama.cpp:server-cuda}"
SIF="${SIF:-${SCRIPT_DIR}/llama-cpp-server-cuda.sif}"
APPTAINER="${APPTAINER:-$(command -v apptainer || command -v singularity)}"

if [ -z "${APPTAINER}" ]; then
    printf 'apptainer/singularity not found on PATH.\n' >&2
    exit 2
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRIPT_DIR}/.apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}" "${SCRIPT_DIR}/logs"

printf '[%s] Pulling %s -> %s\n' "$(date -Iseconds)" "${IMAGE}" "${SIF}"
"${APPTAINER}" pull --force "${SIF}" "${IMAGE}"
printf '[%s] Pull complete: %s (%s)\n' "$(date -Iseconds)" "${SIF}" "$(du -h "${SIF}" | cut -f1)"
"${APPTAINER}" exec "${SIF}" /app/llama-server --version 2>&1 | head -3 || true
