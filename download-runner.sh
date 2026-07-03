#!/usr/bin/env bash
# Resilient resumable download: retries the hf download (resumes from .cache)
# until all 9 UD-Q3_K_XL shards are finalized.
DIR=/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL
for attempt in $(seq 1 40); do
  n=$(ls "$DIR"/GLM-5.2-UD-Q3_K_XL-*-of-00009.gguf 2>/dev/null | wc -l)
  echo "[$(date -Iseconds)] attempt $attempt — $n/9 shards finalized"
  [ "$n" -ge 9 ] && { echo "ALL_SHARDS_DONE"; exit 0; }
  HF_MAX_WORKERS=4 /project/inniang/inference/download-glm52-gguf.sh && { echo "ALL_SHARDS_DONE"; exit 0; }
  echo "[$(date -Iseconds)] attempt $attempt exited non-zero; resuming in 15s"
  sleep 15
done
echo "GAVE_UP after 40 attempts"; exit 1
