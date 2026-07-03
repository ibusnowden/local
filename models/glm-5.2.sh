# GLM-5.2 served as Unsloth 3-bit GGUF (UD-Q3_K_XL) via llama.cpp.
# Consumed by serve-llamacpp.sh.
#
# RTX 6000 Ada is SM89. llama.cpp implements GLM-5.2's MLA + DeepSeek sparse
# attention in portable CUDA that runs on SM89 (unlike the SGLang FlashMLA
# path, which is Hopper/Blackwell-only). UD-Q3_K_XL (~343 GB) fits fully on
# one 8x 48 GB node with ~41 GB headroom for KV cache + buffers.

QUANT="${QUANT:-UD-Q3_K_XL}"
GGUF_DIR="${GLM52_GGUF_DIR:-/project/inniang/inference/models/GLM-5.2-GGUF}"
# llama.cpp auto-loads the remaining split parts from the first shard.
MODEL_GGUF="${GLM52_MODEL_GGUF:-${GGUF_DIR}/${QUANT}/GLM-5.2-${QUANT}-00001-of-00009.gguf}"

SERVED_NAME="glm-5.2"
ALIASES=("glm-5.2")
PORT=8006

# Offload everything to GPU. If long context OOMs, raise N_CPU_MOE to push a
# few MoE expert layers to host RAM (the node has 515 GB), trading speed.
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
# 256k context: MLA KV (q8_0) is only ~26 GB across 8 GPUs (~3.3 GB/GPU), well
# within native 1M-token range so no RoPE scaling. Offload 2 MoE layers to host
# RAM as a load-time VRAM safety margin (GPU 0 carries the extra compute buffer).
N_CPU_MOE="${N_CPU_MOE:-2}"

# Tensor-split across the 8 GPUs is left to llama.cpp's default (even split).
# CONTEXT_LENGTH is the total KV budget, shared across N_PARALLEL slots
# (per-slot context = CONTEXT_LENGTH / N_PARALLEL). Raise either as VRAM allows.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
# 1 slot so a single request gets the FULL 262144 (256k) per-slot context —
# pi and pincode both advertise a 262144 window, so per-slot must equal the
# total budget (per-slot context = CONTEXT_LENGTH / N_PARALLEL). ctx-size is the
# *total* KV, so dropping from 2 slots to 1 is VRAM-neutral; it just stops
# partitioning the 256k budget into 2x 128k. Trade-off: concurrent requests now
# serialize instead of running in parallel. Raise back to 2 only if you also
# halve the advertised client window to 131072.
N_PARALLEL="${N_PARALLEL:-1}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q8_0}"
FLASH_ATTN="${FLASH_ATTN:-on}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

# Speculative decoding (draftless). ngram-mod self-speculates from a shared
# n-gram hash pool — no draft model, no extra VRAM. Decode on this 343 GB MoE
# is expert-read-bandwidth-bound, so batch-verifying a long draft is nearly
# free; wins are biggest on code edits and reasoning re-statements. MoEs want
# long drafts (llama.cpp docs), hence n-min 48 / n-max 64. Set SPEC_TYPE=none
# to disable.
SPEC_TYPE="${SPEC_TYPE:-ngram-mod}"
SPEC_NGRAM_MOD_N_MATCH="${SPEC_NGRAM_MOD_N_MATCH:-24}"
SPEC_NGRAM_MOD_N_MIN="${SPEC_NGRAM_MOD_N_MIN:-48}"
SPEC_NGRAM_MOD_N_MAX="${SPEC_NGRAM_MOD_N_MAX:-64}"
