# SneppX-ALG Documentation

## Overview

SneppX-ALG is a composable 5-component AI algorithm pipeline wrapped in 4 security layers. This is the first open-source AI algorithm with cryptographic integrity built into its foundation.

## Quickstart

### Build from source

```bash
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### Run a demo

```bash
# After building:
./examples/hss_demo
./examples/ser_demo
./examples/arc_demo
./examples/npe_demo
./examples/fm_demo
```

### Run benchmarks

```bash
cmake .. -DSNEPPX_BUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
./tests/benchmark/bench_tensor
./tests/benchmark/bench_autodiff
```

## Architecture

```
                     ┌──────────────────────────────────────┐
                     │           Security Layer              │
                     │  S0 Crypto · S1 Secure Mem · S2 Obf  │
                     │  S3 Behavioral Monitor (WIP)          │
                     └──────────────────────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │         Algorithm Pipeline            │
                     │  HSS → SER → ARC → NPE → FM          │
                     │  (SSM) (MoE) (Guard) (VM)  (Fed Mem) │
                     └──────────────────────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │         Integrity Layer (Future)      │
                     │  ZK Proofs · Formal Safety · On-Device│
                     └──────────────────────────────────────┘
```

## Component Descriptions

### HSS — Hierarchical State Space

Multi-layer state space model with zero-order hold discretization. Processes sequences in O(n log n) time using a parallel scan over the state dimension. Supports hierarchical decomposition for long-range dependencies.

- `docs/architecture.md` for mathematical details
- `src/arch/src/hss/` for implementation
- `tests/unit/hss/` for tests

### SER — Sparse Expert Routing

Softmax-based routing with top-k selection. Each input token is routed to k experts, and the outputs are combined via weighted sum. Includes load-balancing loss to prevent expert collapse.

- `docs/architecture.md` for routing details
- `src/arch/src/ser/` for implementation
- `tests/unit/ser/` for tests

### ARC — Adversarial Robustness Core

Three-layer defense: input guard (z-score anomaly detection), gradient obfuscation (noise + clamping), output verifier (cosine similarity + temporal smoothing). Includes attack simulation (FGSM, PGD, CW).

- `docs/architecture.md` for threat model
- `src/arch/src/arc/` for implementation
- `tests/unit/arc/` for tests

### NPE — Neural Program Executor

16-register virtual machine with 14 opcodes (MATMUL, ATTENTION, SOFTMAX, LAYERNORM, etc.). Supports attention and MLP program compilation from network configurations. Includes static verifier for program correctness.

- `docs/architecture.md` for instruction set
- `src/arch/src/npe/` for implementation
- `tests/unit/npe/` for tests

### FM — Federated Memory

Per-node memory banks with euclidean similarity search and LRU eviction. Supports trust-weighted all-reduce synchronization with Laplace differential privacy noise. Gradient compression via random sampling.

- `docs/architecture.md` for sync protocols
- `src/arch/src/fm/` for implementation
- `tests/unit/fm/` for tests

## Status Table

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| Tensor Core | ~2,000 | 57+27 edge | ✅ Real |
| Memory | ~800 | 13 | ✅ Real |
| Thread Pool | ~300 | 11 | ⚠️ Stub |
| HSS | ~500 | 2 | ⚠️ Partial |
| SER | ~600 | 5 | ⚠️ Partial |
| ARC | ~600 | 5 | ⚠️ Partial |
| NPE | ~700 | 4 | ⚠️ Partial |
| FM | ~600 | 4 | ⚠️ Partial |
| Autodiff | ~400 | 1 | ❌ Stub |
| Optimizer | ~300 | 1 | ❌ Stub |
| Python API | ~500 | 3 | ❌ Stub |
| S0 Crypto | ~2,000 | 10 | ✅ Real |
| S1 Secure Mem | ~800 | 3 | ✅ Real |
| S2 Obfuscation | ~1,500 | 4 | ⚠️ Partial |
| S3 Monitor | ~100 | 0 | ⚠️ Partial |

## Directory Structure

```
SneppX_ALG/
├── CMakeLists.txt
├── CMakePresets.json
├── src/
│   ├── core/                    # Foundation: tensor, memory, thread, autodiff, optimizer
│   ├── arch/                    # Algorithm pipeline: HSS, SER, ARC, NPE, FM, train
│   ├── security/
│   │   ├── c/                   # S0 — Crypto Core + S1 — Secure Memory (C)
│   │   ├── cpp/                 # S2 — Obfuscation Engine (C++)
│   │   ├── asm/                 # x86_64 assembly helpers
│   │   └── rust/                # Future: Rust security layer
│   └── python/                  # pybind11 bindings
├── tests/
│   ├── unit/                    # Component unit tests
│   ├── integration/             # Multi-component integration tests
│   ├── benchmark/               # Performance benchmarks
│   ├── security/                # S0 + S1 (C) tests
│   └── security/cpp/            # S2 (C++) tests
├── examples/                    # Demos for each component
├── docs/                        # Documentation
├── scripts/                     # Build and release scripts
└── cmake/                       # CMake modules
```

## Key Metrics

- **C/C++ source**: ~15,000 lines across all components
- **Tests**: 50 registered (47 pass, 2 pre-existing security edge cases, 1 slow thread test)
- **Build time**: ~30s on modern hardware (Release, 8 cores)
- **Dependencies**: None for C core. pybind11 for Python bindings (optional)

## Next Steps

- Read [docs/architecture.md](architecture.md) for deep technical details
- Read [docs/roadmap.md](roadmap.md) for the project timeline
- Read [docs/installation.md](installation.md) for platform-specific build guides
- Read [docs/api/c.md](api/c.md) for the C API reference
- Read [CONTRIBUTING.md](../CONTRIBUTING.md) to learn how to contribute
