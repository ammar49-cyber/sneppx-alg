# SneppX-ALG Documentation

## Overview

SneppX-ALG is a composable 5-component AI algorithm pipeline wrapped in 10 security layers. v1.0.0 delivers a production-stable pipeline with Model Zoo, distributed training (ZeRO-1/2/3, pipeline/tensor/expert parallelism), advanced architectures (Differential Attention, Mamba-2, FlexAttention, MoD), quantization (INT8/INT4/FP8/AWQ/GPTQ), async checkpointing, profiling & debugging tools, and weight converters for LLaMA 2/3, Mistral, Qwen 2, DeepSeek V2.

## Quickstart

### Build from source

```powershell
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

### Run a demo

```powershell
build\tests\Release\test_hss.exe
build\tests\Release\test_ser.exe
build\tests\Release\test_arc.exe
build\tests\Release\test_npe.exe
build\tests\Release\test_fm.exe
```

### Python quickstart

```powershell
$env:PYTHONPATH = "bindings/python"
python -c "from SneppX_ALG import *; print('ok')"
```

## Architecture

```
                     ┌──────────────────────────────────────┐
                     │         Security Layer (S0-S9)        │
                     │  Crypto · Secure Mem · Obfuscation    │
                     │  Monitor · Network · AI Sanitizer     │
                     │  Key Vault · Updates · Formal · Pentest│
                     └──────────────────────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │         Algorithm Pipeline            │
                     │  HSS → SER → ARC → NPE → FM          │
                     │  (SSM) (MoE) (Guard) (VM)  (Fed Mem) │
                     └──────────────────────────────────────┘
                                     │
                      ┌───────────────▼──────────────────────┐
                      │         Integrity Layer               │
                      │  ZK Proofs · Formal Safety ·          │
                      │  On-Device Attestation                │
                      └──────────────────────────────────────┘
```

## Component Descriptions

### HSS — Hierarchical State Space

Multi-layer state space model with zero-order hold discretization. Blelloch parallel scan over state dimension enabled by default on CPU. Supports hierarchical decomposition for long-range dependencies. Training through the autodiff tape is supported (see `docs/hss_training.md`).

- `docs/ARCHITECTURE.md` for mathematical details
- `algorithms/hss/core/` for implementation
- `tests/unit/hss/` for tests

### SER — Sparse Expert Routing

Softmax-based routing with top-k selection. Each input token is routed to k experts, and the outputs are combined via weighted sum. Includes learned MLP gating (2-layer MLP per expert) with autodiff subgraph, plus load-balancing loss to prevent expert collapse.

- `docs/ARCHITECTURE.md` for routing details
- `algorithms/ser/core/` for implementation
- `tests/unit/ser/` for tests

### ARC — Adversarial Robustness Core

Three-layer defense: input guard (z-score anomaly detection), gradient obfuscation (noise + clamping), output verifier (cosine similarity + temporal smoothing). Includes attack simulation (FGSM, PGD, CW) and adversarial training graph builder that constructs shared-weight clean+adversarial forward graphs.

- `docs/ARCHITECTURE.md` for threat model
- `algorithms/arc/core/` for implementation
- `tests/unit/arc/` for tests

### NPE — Neural Program Executor

16-register virtual machine with 15+ opcodes (MATMUL, ATTENTION, SOFTMAX, LAYERNORM, fused MATMUL+ADD+RELU, etc.). Includes JIT pipeline (DCE → constant folding → fusion → compilation), auto-JIT mode in VM (hot threshold triggers optimization), and a static verifier for program correctness.

- `docs/ARCHITECTURE.md` for instruction set and JIT passes
- `algorithms/npe/core/` for implementation
- `tests/unit/npe/` for tests

### FM — Federated Memory

Per-node memory banks with euclidean similarity search and LRU eviction. Supports trust-weighted all-reduce synchronization with Laplace differential privacy noise. Includes NCCL sync bridge with callback pattern.

- `docs/ARCHITECTURE.md` for sync protocols
- `algorithms/fm/core/` for implementation
- `tests/unit/fm/` for tests

### Trainer — Training Loop

CPU training loop with MSE loss, tape-based autodiff backward, and optimizer step. Optional CUDA-accelerated optimizer: SGD via pure cuBLAS on GPU, AdamW via cuBLAS + host fallback. Configurable via `use_cuda_optimizer` flag.

- `kernel/train/trainer.c` for CPU loop
- `kernel/train/trainer_cuda.c` for CUDA bridge
- `tests/unit/train/` for tests

## Status Table

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| Tensor Core | ~2,500 | 57+27 edge | ✅ Real |
| Memory | ~800 | 13 | ✅ Real |
| Thread Pool | ~300 | 11 | ✅ Real |
| Autodiff | ~600 | tape backward | ✅ Real (layer-norm fixed) |
| Optimizer (CPU) | ~400 | 1 | ✅ Real (Adam/SGD/RMSprop/Adagrad) |
| Optimizer (CUDA) | ~1,100 | 4 | ✅ Real (cuBLAS SGD, AdamW) |
| HSS | ~700 | 2 + integration | ✅ Real (training, parallel scan) |
| SER | ~800 | 6 | ✅ Real (MLP gater + autodiff) |
| ARC | ~700 | 6 | ✅ Real (FGSM/PGD/CW + adversarial train graph) |
| NPE | ~1,200 | 10 | ✅ Real (JIT pipeline, auto-JIT, fused ops) |
| FM | ~700 | 5 | ✅ Real (NCCL sync bridge) |
| Trainer (CPU) | ~500 | 3 | ✅ Real |
| Trainer (CUDA) | ~400 | 4 | ✅ Real (device detection, async transfers) |
| Python API | ~800 | 11 | ✅ Real (ARC/NPE/FM/Trainer wrappers) |
| Model Zoo | ~1,200 | 49 | ✅ Real (from_pretrained, weight converters) |
| Distributed | ~1,200 | 44 | ✅ Real (ZeRO-1/2/3, TP/PP/EP, elastic) |
| Quantization | ~1,300 | 17 | ✅ Real (INT8/INT4/FP8/AWQ/GPTQ) |
| Advanced Arch | ~500 | 20 | ✅ Real (DifferentialAttn, Mamba-2, FlexAttn, MoD) |
| Async Checkpoint | ~500 | 23 | ✅ Real (fault tolerance, elastic) |
| Profiling | ~300 | 13 | ✅ Real (profiler, logger, sanitizers) |
| CUDA Backend | ~9,600 | 25 | ✅ Real (FlashAttn, fused optim, ZeRO) |
| S0 Crypto | ~2,500 | 10 | ✅ Real |
| S1 Secure Mem | ~800 | 3 | ✅ Real |
| S2 Obfuscation | ~1,500 | 4 | ✅ Complete |
| S3 Monitor | ~100 | 0 | ✅ Complete |
| S4-S9 | ~3,000 | 5 | ✅ Complete |

## Directory Structure

```
SneppX_ALG/
├── CMakeLists.txt
├── CMakePresets.json
├── include/neural_core/        # Public headers
│   ├── kernel/                  # Tensor, memory, autodiff, optimizer, trainer
│   ├── architecture/            # Algorithm configs and API
│   └── security/                # S0-S9 security headers
├── kernel/                      # Core implementations
│   ├── tensor/                  # Tensor ops, IO, format readers
│   ├── autodiff/                # Autodiff tape and gradient ops
│   ├── train/                   # Trainer and CUDA bridge
│   ├── optimizer/               # SGD, Adam/W, RMSprop, Adagrad
│   ├── attention/               # Flash Attention v2/v3, RoPE, KV-cache
│   ├── arch/                    # Advanced architectures
│   ├── distributed/             # DDP, ZeRO, pipeline, tensor/ep parallel
│   ├── quantization/            # INT8/FP8/AWQ/GPTQ quantization
│   └── cuda/                    # CUDA kernels (opt-in)
├── algorithms/                  # Algorithm pipeline implementations
│   ├── hss/core/                # HSS forward, training, CUDA kernels
│   ├── ser/core/                # SER routing, MLP gater, CUDA kernels
│   ├── arc/core/                # ARC defense, attacks, adversarial train graph
│   ├── npe/core/                # NPE VM, compiler, JIT pipeline, CUDA kernels
│   └── fm/core/                 # FM memory bank, NCCL sync, CUDA kernels
├── security/                    # S0-S9 security layer source
│   ├── crypto/                  # S0 — Cryptographic Core
│   ├── memory/                  # S1 — Secure Memory
│   ├── cpp/                     # S2 — Obfuscation Engine (+ S3 monitor)
│   ├── network/                 # S4 — Network Security
│   ├── ai/                      # S5 — AI Sanitizer
│   ├── ui/                      # S6 — Security UI / Key Vault
│   ├── updates/                 # S7 — Secure Updates
│   ├── formal/                  # S8 — Formal Verification
│   ├── pentest/                 # S9 — Penetration Testing
│   ├── fuzzing/                 # Fuzz harnesses and corpora
│   └── asm/                     # x86_64 MASM helpers
├── tests/
│   ├── unit/                    # Per-component unit tests
│   ├── integration/             # Multi-component integration tests
│   ├── benchmark/               # Performance benchmarks
│   ├── security/                # S0-S9 security tests
│   ├── python/                  # Python wrapper tests
│   └── quantization/            # Quantization tests
├── bindings/python/             # Python bindings
│   └── SneppX_ALG/interface_bindings/  # Per-algorithm Python wrappers
├── examples/                    # Demo programs
├── docs/                        # Documentation
├── scripts/                     # Build and release scripts
├── cmake/                       # CMake modules
└── config/                      # Model zoo configs
```

## Key Metrics

- **C/C++ source**: ~25,000 lines across all components
- **Tests**: 100+ registered (all pass except 2 pre-existing S0 edge cases)
- **Build time**: ~30s on modern hardware (Release, 8 cores)
- **Dependencies**: None for C core. Python stdlib-only for Python wrappers

## Model Zoo (v1.0.0)

Load pre-trained models from Hugging Face with one call:

```python
from SneppX_ALG.model_zoo import from_pretrained

model = from_pretrained("meta-llama/Llama-2-7b-hf")
```

Supported architectures: LLaMA 2 (7B/13B/70B), LLaMA 3 (8B/70B), Mistral (7B), Qwen 2 (7B/72B), DeepSeek V2 (Lite/Full). See [`docs/guide/model_zoo.md`](guide/model_zoo.md) for the full guide.

## Context Extension (v1.0.0)

Extend any supported model to 128K context via NTK-aware RoPE scaling:

```python
cfg = LLMConfig.from_name("llama3", "8B")
cfg.extend_context(131072)
```

See [`docs/guide/128k_context_extension.md`](guide/128k_context_extension.md).

## Multi-Head Attention Forward (v1.0.0)

Full MHA forward pass with GQA, Flash Attention, RoPE, and MLA (DeepSeek V2):

```python
output = cfg.forward_mha(hidden_states, attention_mask, position_ids)
```

See [`docs/guide/mha_forward_pass.md`](guide/mha_forward_pass.md).

## Next Steps

- Read [docs/guide/index.md](guide/index.md) for feature guides
- Read [docs/ARCHITECTURE.md](ARCHITECTURE.md) for deep technical details
- Read [docs/hss_training.md](hss_training.md) for an end-to-end HSS training walkthrough
- Read [docs/API.md](API.md) for the full C and Python API reference
- Read [docs/installation.md](installation.md) for platform-specific build guides
- Read [docs/ROADMAP.md](ROADMAP.md) for the project timeline
- Read [docs/contributing.md](contributing.md) to learn how to contribute
