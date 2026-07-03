#!/usr/bin/env bash
set -uo pipefail
BIN=/project/inniang/inference/llama.cpp/build-cuda/bin
TINY=/project/inniang/inference/tmp-mtp-test/tiny-glm-dsa.gguf
LOG=/project/inniang/inference/tmp-mtp-test/srv-$SLURM_JOB_ID.log
"$BIN/llama-server" -m "$TINY" --port 8992 -ngl 99 --spec-type draft-mtp --spec-draft-n-max 3 >"$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8992/health >/dev/null 2>&1 && break; sleep 2; done
OUT=$(curl -sf -m 120 http://127.0.0.1:8992/completion -d '{"prompt":"The quick brown fox","n_predict":32,"temperature":0,"seed":1}')
kill $SRV 2>/dev/null; wait 2>/dev/null
echo "RESPONSE: $OUT" | head -c 900; echo
grep -i "draft acceptance\|statistics\|MTP draft context\|CUDA0\|error" "$LOG" | tail -8
