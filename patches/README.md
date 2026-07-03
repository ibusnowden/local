# Local llama.cpp patches

## glm-dsa-draft-mtp.patch

Local port of a `draft-mtp` speculative-decoding path for the GLM-5.2 `glm-dsa`
architecture (DECODER_MTP graph). Not upstream. Touches:

- `src/llama-model.cpp`
- `src/models/deepseek2.cpp`
- `src/models/glm-dsa.cpp`
- `src/models/models.h`

Base commit: `ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0`
("opencl: flash attention improvement (#25069)")

To reproduce the working tree:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0
git apply ../patches/glm-dsa-draft-mtp.patch
# then build (native sm_89 CUDA -> build-cuda/bin/llama-server):
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --target llama-server -j
```

Validated CPU+GPU on a tiny synthetic glm-dsa model (`tmp-mtp-test/`);
required for `NATIVE=1 SPEC_TYPE=draft-mtp,ngram-mod` serving.
