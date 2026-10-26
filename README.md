# llama2-min inference experiments

Source-only C/CUDA experiments for running Llama 2 inference across a sequence of CPU and GPU implementations. The repository keeps the active version folders and quantization tools; it does not include model or tokenizer data.

## Included versions

- `v0-fp32/`: baseline FP32 C inference
- `v1-int8-vnni/`: CPU INT8/VNNI runner and quantizer
- `v2-cuda/`: CUDA INT8 runner
- `v3-cuda-kvcache/`: CUDA runner with a KV cache
- `v4-cuda-int4/`: CUDA INT4 runner and quantizer
- `v5-cuda-fused/`: fused CUDA INT4 runner and a commented source variant
- `tools/`: canonical INT8 and INT4 quantizers

## Inputs and building

The model weights and tokenizer are deliberately excluded. A runner takes a compatible checkpoint as its first positional argument; `-z` selects a tokenizer file, otherwise the code looks for `tokenizer.bin` in the current directory. The quantizer scripts use a checkpoint under `models/` by default and accept input and output paths as arguments. The local `models/` directory is ignored by Git.

The CPU sources need a C compiler; the VNNI version also requires a CPU and compiler that support its AVX-512 VNNI instructions. CUDA sources need the CUDA toolkit and a compatible NVIDIA GPU. The Python quantizers require NumPy. Each runner prints its usage options when called without a checkpoint; common options include `-t` temperature, `-p` top-p, `-n` generated steps, `-i` prompt, `-m` mode, and `-y` system prompt.

These are separate implementation snapshots, not a single packaged application. Build and run only the variant appropriate for your hardware and checkpoint format. This source-only publication has not been compiled or performance-tested.

No license file was supplied, so this repository does not add a license. Model weights are not included; obtain any required model files from sources and under terms you are authorized to use.
