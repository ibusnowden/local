#!/usr/bin/env bash
set -uo pipefail
cd /project/inniang/glmserve
export LD_LIBRARY_PATH=/project/inniang/entropy/.cudaenv/lib:${LD_LIBRARY_PATH:-}
/project/inniang/.venv/bin/python tests/test_mtp_logits.py --gpu
