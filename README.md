# SNEPPX-Alg: Secure Neural Architecture (ARIX_Algo)

This directory contains the **SNEPPX-Alg** cognitive processing system — a
next-generation AI framework with security built into the foundation.

See the [top-level README](https://github.com/ammar49-cyber/sneppx-alg) and
[`Docs.md`](./Docs.md) for the full overview, build instructions, and the
S0–S9 security model.

## What's new in v0.9.7.890e

- **HSS training backward fixed** — corrected the layer-norm gamma/beta gradient
  pointer dereference in `backward_layer_norm` (`c->gamma->data` was cast to `float*`
  instead of `c->gamma->data->data`) and added the missing `#include <math.h>` in
  `ops.c` (an implicitly-declared `sqrt` was returning a constant, poisoning every
  layer-norm gradient). `test_train_integration` now converges deterministically
  (loss drops >90% over 10 Adam steps).
- **Real format readers** — safetensors, NumPy (`.npy`/`.npz`), PyTorch (`.pth`),
  and ONNX loaders read and write real bytes into the tensor engine.
- **Real kernel** — N-D matrix multiplication and a gradient-descent optimizer on
  the autodiff tape; the kernel no longer shadows real code with stubs.
- **Seven neural architectures** — Transformer, ViT, GCN, RNN, GAN, Diffusion, and
  Reinforcement Learning, plus the ARC / SER / HSS / NPE / FM modules.
- **Hardened security** — fixed a `SNEPPX_secure_free` symbol collision and a
  secure-memory mapping-release leak in the S0–S9 layer.
- **Opt-in backends are now real** — Vulkan, TPU, HTTP, and ZK perform genuine
  reference computation (gated by `SNEPPX_BUILD_*` flags, OFF by default) and are
  exercised by a new `test_backend_full` suite (12/12).

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
