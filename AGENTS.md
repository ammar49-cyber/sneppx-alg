# SNEPPX Algo — Agent Guide

## Project Overview
SNEPPX Algo is a cognitive processing system implementing neural architecture search, hierarchical state spaces, mixture of experts, and a full S0–S9 security layer. Written in C11 + C++20, targeting x86-64 (MSVC 19.44, GCC, Clang).

## Build Commands
```powershell
# Configure (debug)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Configure (release)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build a specific target
cmake --build build --config Release --target neural_security_c

# Build all
cmake --build build --config Release

# Run tests
cd build && ctest -C Release --output-on-failure

# Run a specific test
ctest -C Release -R test_kyber --output-on-failure
```

## Project Conventions
- **Language**: C11 (`.c`), C++20 (`.cpp`), MASM (`.asm`)
- **No VLAs**: MSVC C11 doesn't support them — use `calloc`/`free`
- **No `__int128` on MSVC**: Use `#ifndef NO_UINT128` guards (see `x25519.c` pattern)
- **No K&R-style declarations**: All functions must use modern C prototype syntax
- **Include paths**: Short form from `include/neural_core/security/` — e.g., `#include "kyber.h"`
- **ASM syntax**: MASM (Intel syntax) for x86-64, placed in `security/crypto/asm/x86_64/`
- **CMake**: Uses `file(GLOB_RECURSE)` — new `.c`/`.asm` files are auto-discovered
- **Memory allocation**: `SNEPPX_secure_malloc`/`SNEPPX_secure_free` for sensitive data

## Key Files
| Path | Purpose |
|------|---------|
| `include/neural_core/security/cryptographic_primitives_bundle.h` | Umbrella header for all security modules |
| `security/crypto/c/` | C crypto implementations (Kyber, Dilithium, SPHINCS+, AES-GCM, ChaCha20, etc.) |
| `security/crypto/asm/x86_64/` | MASM-optimized crypto (AES-NI, SHA-NI, AVX2, SSE2) |
| `security/ai/` | RLHF safety, differential privacy, prompt/output filtering |
| `security/network/` | DDoS mitigation, transport security, identity management |
| `security/memory/` | Memory hardening, leak detector |
| `security/monitor/` | Container breakout detection, runtime monitoring |
| `security/ui/` | Key vault |
| `security/updates/` | Container security (SBOM, CVE scanning), signed updates |
| `tests/security/` | Test files for all security modules |
| `kernel/` | Core computational substrate (tensor, autodiff, optimizer, trainer, thread pool) |
| `algorithms/` | ARC, SER, HSS, NPE, FM algorithm implementations |

## Phase 1 Completed (GPU/CUDA Backend)
Core CUDA backend (~9,600 new lines across kernel/cuda/, algorithms/*/cuda/, net/distributed/, tests/):

### kernel/cuda/ — Core CUDA Library (~6,900 lines)
- **common.cuh** (371L): Architecture detection (Hopper/Ampere/Volta/Pascal), tile config, FP16/BF16 conversions, MMA/WGMMA wrappers, cp.async helpers, warp shuffle reductions, Philox RNG, CUBLAS TLS handle management
- **tensor_cuda.h/.cu** (1,238L): Tensor-core GEMM (128x128 block tiling, warp-level MMA), element-wise ops, reduction, layernorm, softmax, CUBLAS fallback, fused bias+activation GEMM
- **attention_cuda.h/.cu** (1,579L): Flash Attention v2 with online softmax + tiling; Flash Attention v3 TMA+WGMMA stub; GQA, paged attention, KV cache management (alloc/free/update), RoPE (forward/inplace/cache precompute), causal/sliding window/block-sparse masks
- **autodiff_cuda.h/.cu** (1,237L): GEMM backward (via cuBLAS), activation backward (ReLU/GELU/SiLU/Tanh/Sigmoid), layernorm/RMSNorm backward, softmax/CE backward, element-wise backward, dropout backward, MSE/BCE backward, convolution backward, gradient clipping/accumulation/scale
- **optim_cuda.h/.cu** (1,033L): Fused AdamW, SGD+momentum, Lion, LAMB, LARS, AdaFactor optimizers; ZeRO-1 partitioned AdamW; in-kernel LR scheduling (cosine/linear/warmup); overflow checking; generic optimizer lifecycle
- **memory_cuda.h/.cu** (713L): Memory pool (pre-allocated blocks), stream pool, event pool, pinned/managed memory, async 2D/batched memcpy, device properties query, kernel auto-block size tuning
- **rng_cuda.h/.cu** (676L): Philox-based uniform/normal/truncated-normal/bernoulli/integer RNG; Xavier/Kaiming initialization; fused dropout forward; batch RNG; permutation (Fisher-Yates)

### algorithms/*/cuda/ — Algorithm CUDA Extensions (~1,700 lines)
- **hss/**: Selective scan (Mamba/S6), S4 forward, HiPPO matrix init, SSM convolution, hierarchical softmax, sparse-dense matmul, top-k, parallel prefix scan (Blelloch)
- **ser/**: Top-k gating (softmax + selection), MoE dispatch/combine, load balancing loss, expert all-to-all, fused MoE forward
- **fm/**: Ring/butterfly all-reduce, gradient quantization (FP16→INT8), Top-K sparsification, federated averaging, memory bank sync
- **npe/**: Neural VM instruction dispatch kernel, differentiable program execution
- **arc/**: PGD/FGSM adversarial attacks, gradient obfuscation, randomized smoothing

### net/distributed/ — NCCL Layer (~530 lines)
- nccl.h/.c: Dynamic NCCL loading (win/linux/mac), all-reduce/all-gather/reduce-scatter/broadcast/send/recv, CPU fallback, process group management

### tests/ — CUDA Test Suite
- cuda_test_suite.cu (487L): Device properties, GEMM vs cuBLAS, element-wise, layernorm, softmax, AdamW, memory pool, RNG, dropout, gradient clipping

## Build Targets
- `neural_core_kernel` — Core tensor/memory/trainer library
- `neural_architecture_layer` — Neural architecture algorithms
- `neural_security_c` — C security library
- `neural_security_cpp` — C++ obfuscation library
- `neural_cuda_kernels` — CUDA kernels (conditional, OFF by default, set -DSNEPPX_BUILD_CUDA=ON)

## Coding Standards
- 4-space indentation, no tabs
- `SNEPPX_` prefix for all public functions and types
- `SNEPPX_` prefix for all macros and constants
- `void` in empty parameter lists: `int foo(void)` not `int foo()`
- Return `int` (0 success, -1 error) for most API functions
- `size_t` for lengths/counts, `uint8_t*` for byte buffers
- Document all public API functions with brief header comments
