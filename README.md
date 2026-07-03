# inference — GLM-5.2 serving (llama.cpp, 3-bit GGUF)

This folder serves the local `/project/inniang` coding stack with a single
OpenAI-compatible model: **GLM-5.2**, as the Unsloth dynamic 3-bit GGUF
(`unsloth/GLM-5.2-GGUF`, quant `UD-Q3_K_XL`) running under **llama.cpp** on one
8× RTX 6000 Ada node.

Fabric/Pi keep using the stable alias `glm-5.2` through the LiteLLM router on
`127.0.0.1:4000`. The backend served model name is also `glm-5.2`.

## Why llama.cpp (not SGLang/KTransformers) on this hardware

RTX 6000 Ada is **SM89**. GLM-5.2 uses MLA + DeepSeek sparse attention
(`glm_moe_dsa`, head dims 192/64/256). Every efficient NSA attention kernel in
the KVCache-AI SGLang/KTransformers fork is **Hopper/Blackwell-only**, so it
loads the model and opens the API but dies on the first forward pass:

```text
flashmla_sparse/fa3: sparse_prefill_fwd is only supported on SM90 or SM100
tilelang:            no valid layout found
trtllm:              qk_nope_head_dim must be 128, GLM-5.2 uses 192
flashmla_kv:         FlashMLA path requires SM90 or SM100
```

This is a kernel-availability wall, not a config bug. **llama.cpp** implements
GLM-5.2's sparse attention in portable CUDA that runs on SM89, so it is the
serving path on the Ada nodes. The NVFP4 checkpoint and the SGLang/KTransformers
stack were removed; they belong on the `itiger01` H100 (SM90) node.

## Sizing

GLM-5.2 is 754B params. One node = 8 × 48 GB = **384 GB VRAM**.

| Quant | ~bpw | Size | Fits 384 GB VRAM? |
|-------|------|------|-------------------|
| `UD-Q2_K_XL` | 2.69 | ~254 GB | yes (lots of headroom) |
| **`UD-Q3_K_XL`** (default) | ~3.5 | **~343 GB** | yes (~41 GB headroom) |
| `UD-Q4_K_XL` | ~4.5 | ~425 GB | no — needs CPU offload, or use the H100 node |

## Layout

- `models/glm-5.2.sh` — llama.cpp serving profile (GGUF path, ngl, ctx, KV type)
- `download-glm52-gguf.sh` — downloads `unsloth/GLM-5.2-GGUF` `UD-Q3_K_XL`
- `build-llamacpp.sh` — Slurm build of llama.cpp w/ CUDA for SM89
- `serve-llamacpp.sh` — Slurm launcher for 8× RTX 6000 Ada
- `llama.cpp/` — standalone llama.cpp source + `build/bin/llama-server`
- `router/regen-config.sh` — writes LiteLLM config from the live Slurm endpoint
- `router/router.sh` — starts the LiteLLM proxy
- `test-glm52.sh` — direct OpenAI-compatible smoke test
- `wire-pi-glm52.sh` — wires the Pi coding agent to the live endpoint

## One-Time Setup

Build llama.cpp (CUDA, SM89) on a GPU node — produces
`llama.cpp/build/bin/llama-server`:

```bash
sbatch /project/inniang/inference/build-llamacpp.sh
tail -f /project/inniang/inference/logs/slurm-build-llamacpp-<jobid>.out
```

Download the 3-bit GGUF (~343 GB, 9 split parts) on a login node:

```bash
/project/inniang/inference/download-glm52-gguf.sh
# -> /project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL/
```

Hugging Face cache defaults to `/project/inniang/inference/.hf-cache`.

## Serve

```bash
sbatch /project/inniang/inference/serve-llamacpp.sh glm-5.2
tail -f /project/inniang/inference/logs/slurm-serve-llamacpp-<jobid>.err
```

Defaults (one 8× RTX 6000 Ada node, full GPU offload):

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPU_LAYERS=999      # everything on GPU
N_CPU_MOE=0           # raise to spill MoE expert layers to host RAM if OOM
CONTEXT_LENGTH=32768  # total KV budget (per-slot = CONTEXT_LENGTH / N_PARALLEL)
N_PARALLEL=1
KV_CACHE_TYPE=q8_0
FLASH_ATTN=on
--fit off             # MANDATORY: skips the device-memory auto-fit probe,
                      # which otherwise hangs the load ~30 min on a model this big
```

If a long context OOMs at this quant's tight ~41 GB headroom, either lower
`CONTEXT_LENGTH` or set `N_CPU_MOE` to a few layers (the node has 515 GB RAM),
trading decode speed for the extra room. Both are env-overridable.

## Router

Once the Slurm job is RUNNING:

```bash
/project/inniang/inference/router/regen-config.sh
/project/inniang/inference/router/router.sh
```

Fabric/Pi defaults to:

```bash
LOCAL_MODEL=glm-5.2
LOCAL_BASE_URL=http://127.0.0.1:4000/v1
```

## Test And Wire Clients

Smoke-test the direct llama.cpp endpoint:

```bash
/project/inniang/inference/test-glm52.sh
```

Wire Pi after the smoke test passes:

```bash
/project/inniang/inference/wire-pi-glm52.sh
/project/inniang/inference/wire-pi-glm52.sh --set-default
```
