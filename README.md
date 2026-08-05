# SNEPPX-Alg: Secure Neural Architecture (ARIX_Algo)

[![Version](https://img.shields.io/badge/version-1.1.1-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![C/C++](https://img.shields.io/badge/language-C%2FC%2B%2B-00599C.svg)]()
[![Python](https://img.shields.io/badge/language-Python-3776AB.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![PRs](https://img.shields.io/badge/PRs-email%20only-yellow.svg)](mailto:algoarix@gmail.com)

> **ARIX_Algo** — Secure, composable, production-grade AI algorithm pipeline with 10 security layers (S0–S9), model zoo, distributed training, quantization, and advanced architectures.

This directory contains the **SNEPPX-Alg** cognitive processing system — a
next-generation AI framework with security built into the foundation.

See the [top-level README](https://github.com/ammar49-cyber/sneppx-alg) for the full overview.
For complete documentation, start at [`docs/index.md`](docs/index.md).

## Features

- **5-component algorithm pipeline**: HSS (SSM), SER (MoE), ARC (Adversarial Guard), NPE (Neural VM), FM (Federated Memory)
- **10 security layers (S0–S9)**: Crypto, Secure Memory, Obfuscation, Monitoring, Network, AI Sanitizer, Key Vault, Updates, Formal Verification, Penetration Testing
- **Model Zoo**: `from_pretrained()` API with ModelHub, weight management, model cards, converter presets for LLaMA 2/3, Mistral, Qwen 2, DeepSeek V2
- **Distributed Training**: ZeRO-1/2/3, pipeline/tensor/expert parallelism, elastic training, fault tolerance
- **Quantization**: INT8/INT4/FP8, AWQ, GPTQ
- **Advanced Architectures**: Differential Attention, Mamba-2 SSM, FlexAttention, Mixture of Depth
- **Built-in Profiling & Debugging**: Profiler, logger, NVTX stubs, sanitizer scripts

## Documentation

| Area | Where to start |
|------|----------------|
| Quick start & build | [`docs/index.md`](docs/index.md) |
| Contribution framework | [`docs/CONTRIBUTOR_TIERS.md`](docs/CONTRIBUTOR_TIERS.md) |
| Branching strategy | [`docs/BRANCHING_STRATEGY.md`](docs/BRANCHING_STRATEGY.md) |
| Code review guide | [`docs/CODE_REVIEW_GUIDE.md`](docs/CODE_REVIEW_GUIDE.md) |
| API reference | [`docs/API.md`](docs/API.md) |
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Security layers | [`docs/security_layers.md`](docs/security_layers.md) |
| Development workflow | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) |
| Commenting standard | [`docs/COMMENTING.md`](docs/COMMENTING.md) |
| All docs | [`docs/README.md`](docs/README.md) |

## What's new in v1.1.1

- **C HTTP REST API**: Real serving endpoints — `GET /v1/health`, `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/generate` (see `net/http/http_api.c`, demo at `examples/http_server_demo.c`)
- **Security fixes**: Unsafe C functions replaced with safe alternatives across 8 files (model factory, VIT, numpy/onnx/pth/safetensors formats, HTTP auth, S9 extensions)
- **Python packaging**: `pyproject.toml` wheel metadata + `cp311-cp311-win_amd64` wheel tag, optional serve deps
- **Dev tools chain**: `scripts/dev-tools.ps1/.sh`, `.sneppx-tools.json` manifest, extended pre-commit
- **Quickstart**: `QUICKSTART.md` for build/test/install/serve

## v1.1.0

- **Contribution framework**: Five-tier merit system, branching strategy, code review guide, learning paths
- **CI/CD removed**: All automated pipelines deleted. Manual builds & tests only
- **Docs reorganized**: 4 reference docs converted from `.txt` to `.md`, Doxygen config added

## v1.0.0 — Initial Stable Release

- **Model Zoo**: `from_pretrained()` API, ModelHub, C/C++/Python model configs, model cards, weight management
- **Distributed Training**: ZeRO-1/2/3, pipeline/tensor/expert parallelism, elastic training, fault tolerance
- **Advanced Architectures**: Differential Attention, Mamba-2 SSM, FlexAttention, Mixture of Depth
- **Security Audit**: S0-S9 hardened (symbol-collision fixes, mapping-leak fixes, behavioral monitoring)
- **Quantization**: INT8/INT4/FP8, AWQ, GPTQ
- **Async Checkpointing**: Double-buffered save, heartbeat, elastic node join/leave
- **Profiling & Debugging**: Profiler, logger, NVTX stubs, sanitizer scripts

## Quick build

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

Opt-in backends — real reference implementations, **OFF by default**. Each performs
genuine computation via the shared reference-compute path and reports
`DRIVER_UNSUPPORTED` when its flag is off:

```powershell
cmake -B build -G Ninja -DSNEPPX_BUILD_VULKAN=ON   # Vulkan — real GEMM / elementwise reference compute
cmake -B build -G Ninja -DSNEPPX_BUILD_TPU=ON      # TPU — real GEMM reference + device emulation
cmake -B build -G Ninja -DSNEPPX_BUILD_HTTP=ON     # HTTP — real BSD-socket transport (GET/POST)
cmake -B build -G Ninja -DSNEPPX_BUILD_ZK=ON       # ZK — real Schnorr proof over Curve25519 (p = 2^255 - 19)
cmake -B build -G Ninja -DSNEPPX_BUILD_METAL=ON    # Apple Metal reference backend
cmake -B build -G Ninja -DSNEPPX_BUILD_ONEAPI=ON   # Intel oneAPI/SYCL reference backend
```

Build everything (all opt-in backends + tests) and run the full suite:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON `
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
