# SNEPPX-Alg: Secure Neural Architecture (ARIX_Algo)

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![C/C++](https://img.shields.io/badge/language-C%2FC%2B%2B-00599C.svg)]()
[![Python](https://img.shields.io/badge/language-Python-3776AB.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![PRs](https://img.shields.io/badge/PRs-email%20only-yellow.svg)](mailto:algoarix@gmail.com)

> **ARIX_Algo** — Secure, composable, production-grade AI algorithm pipeline with 10 security layers (S0–S9), model zoo, distributed training, quantization, and advanced architectures.

This directory contains the **SNEPPX-Alg** cognitive processing system — a
next-generation AI framework with security built into the foundation.

See the [top-level README](https://github.com/ammar49-cyber/sneppx-alg) and
[`Docs.md`](./Docs.md) for the full overview, build instructions, and the
S0–S9 security model.

## Features

- **5-component algorithm pipeline**: HSS (SSM), SER (MoE), ARC (Adversarial Guard), NPE (Neural VM), FM (Federated Memory)
- **10 security layers (S0–S9)**: Crypto, Secure Memory, Obfuscation, Monitoring, Network, AI Sanitizer, Key Vault, Updates, Formal Verification, Penetration Testing
- **Model Zoo**: `from_pretrained()` API with ModelHub, weight management, model cards, converter presets for LLaMA 2/3, Mistral, Qwen 2, DeepSeek V2
- **Distributed Training**: ZeRO-1/2/3, pipeline/tensor/expert parallelism, elastic training, fault tolerance
- **Quantization**: INT8/INT4/FP8, AWQ, GPTQ
- **Advanced Architectures**: Differential Attention, Mamba-2 SSM, FlexAttention, Mixture of Depth
- **Built-in Profiling & Debugging**: Profiler, logger, NVTX stubs, sanitizer scripts

## What's new in v1.0.0

- **Stable Release** — All 8 development phases complete, API frozen, full regression suite passes
- **Phase 1 — Model Zoo**: `from_pretrained()` API, ModelHub, C/C++/Python model configs, model cards, weight management, integration tests
- **Phase 2 — Distributed Training**: ZeRO-1/2/3, pipeline parallelism, tensor parallelism, expert parallelism, elastic training, fault tolerance
- **Phase 3 — Advanced Architectures**: Differential Attention, Mamba-2 selective SSM, FlexAttention with block-sparse kernels, Mixture of Depth
- **Phase 4 — Security Audit**: S0-S9 hardened (symbol-collision fixes, mapping-leak fixes, behavioral monitoring)
- **Phase 5 — Quantization**: INT8 sym/asym, INT4 packed, FP8 E4M3/E5M2, AWQ, GPTQ
- **Phase 6 — Async Checkpointing**: Double-buffered async save, heartbeat, elastic node join/leave
- **Phase 7 — Profiling & Debugging**: Profiler, logger, NVTX stubs, sanitizer scripts
- **Phase 8 — Weight Converters**: HF integration, safetensors reader, presets for LLaMA 2/3, Mistral, Qwen 2, DeepSeek V2

## Quick build

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

Opt-in backends — real reference implementations, **OFF by default**. Each performs
genuine computation via the shared reference-compute path and reports
`DRIVER_UNSUPPORTED` when its flag is off:

```powershell
cmake -B build -DSNEPPX_BUILD_VULKAN=ON   # Vulkan — real GEMM / elementwise reference compute
cmake -B build -DSNEPPX_BUILD_TPU=ON      # TPU — real GEMM reference + device emulation
cmake -B build -DSNEPPX_BUILD_HTTP=ON     # HTTP — real BSD-socket transport (GET/POST)
cmake -B build -DSNEPPX_BUILD_ZK=ON       # ZK — real Schnorr proof over Curve25519 (p = 2^255 - 19)
cmake -B build -DSNEPPX_BUILD_METAL=ON    # Apple Metal reference backend
cmake -B build -DSNEPPX_BUILD_ONEAPI=ON   # Intel oneAPI/SYCL reference backend
```

Build everything (all opt-in backends + tests) and run the full suite:

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON `
  -DSNEPPX_BUILD_VULKAN=ON -DSNEPPX_BUILD_TPU=ON -DSNEPPX_BUILD_HTTP=ON -DSNEPPX_BUILD_ZK=ON
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

## Layout

| Path | Purpose |
|------|---------|
| `kernel/` | Core tensor/autodiff/optimizer/trainer substrate |
| `algorithms/` | HSS, SER, ARC, NPE, FM, Transformer, ViT, GCN, RNN, GAN, Diffusion, RL |
| `drivers/` | Accelerator backends (CUDA, ROCm, Vulkan, TPU, HTTP, ZK, Metal*, oneAPI*) |
| `security/` | S0–S9 security layer |
| `net/` | Distributed + gRPC coordination |
| `bindings/python/` | Python API |
| `releases/` | Release signing tooling |

`*` Metal and oneAPI are reference-compute backends enabled via
`SNEPPX_BUILD_METAL` / `SNEPPX_BUILD_ONEAPI`. Vulkan/TPU/HTTP/ZK do real
reference computation and are enabled via `SNEPPX_BUILD_VULKAN` /
`SNEPPX_BUILD_TPU` / `SNEPPX_BUILD_HTTP` / `SNEPPX_BUILD_ZK`.

## License

MIT — see [`LICENSE`](./LICENSE). Maintained by **Ammar [SNEPPX]**.
