# SNEPPX-ALG

**Next-generation AI architecture with security built into the foundation** — v0.9.5.748

[![Build Status](https://github.com/ammar49-cyber/sneppx-alg/workflows/CI/badge.svg)](https://github.com/ammar49-cyber/sneppx-alg/actions)
[![PyPI Version](https://img.shields.io/pypi/v/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code Size](https://img.shields.io/badge/130K%20lines-630%20files-blue)](https://github.com/ammar49-cyber/sneppx-alg)

---

## Overview

**SNEPPX-ALG** is a next-generation AI architecture that integrates security at every layer — from tensor operations to model deployment. Built for high-performance AI workloads with enterprise-grade security.

### Key Features

- **🚀 High-Performance Tensor Engine**: SIMD-optimized (AVX2/AVX-512), CUDA kernels for NVIDIA GPUs (Ampere/Hopper)
- **🔒 Security-First Design**: S0-S9 security layers (memory hardening, obfuscation, post-quantum crypto, AI safety)
- **🧠 Modern Architectures**: Vision Transformers (ViT, DeiT, Swin), LLMs (LLaMA, Mistral, Qwen2, DeepSeek V2), MAE
- **⚡ Distributed Training**: ZeRO-1/2/3, Pipeline/Tensor/Expert Parallelism, Elastic Training
- **🔧 Model Compression**: Quantization (INT8/FP8/AWQ/GPTQ), Pruning (Magnitude/Movement/Taylor), Distillation
- **📦 Deployment Ready**: ONNX Export, TensorRT Integration, Python/C++ APIs
- **🐍 Complete Python Bindings**: 37 modules across 7 phases with 300+ tests, pure-Python fallback
- **🖥️ CLI Commands**: `sneppx-train`, `sneppx-serve`, `sneppx-experiment` installed with the package

---

## Quick Start

### Python Installation

```bash
pip install sneppx-alg
```

With inference server (uvicorn):
```bash
pip install sneppx-alg[serve]
```

With HuggingFace integration:
```bash
pip install sneppx-alg[hf]
```

### Quick Example

```python
from SneppX_ALG import Tensor, Linear, AdamW, Trainer, TrainConfig
import numpy as np

# Create a model
model = Linear(768, 10)

# Training config
config = TrainConfig(
    learning_rate=3e-4,
    batch_size=32,
    max_steps=1000,
    optimizer="adamw"
)

trainer = Trainer(model, config)

# Synthetic data
x = Tensor.from_numpy(np.random.randn(32, 768).astype(np.float32))
y = Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))

# Train
trainer.fit([(x, y)])
```

### CLI Commands

```bash
# Training with YAML config
sneppx-train --config config.yaml --eval-only --resume checkpoint.sneppx

# Inference server with firewall, auth, TLS
sneppx-serve --port 8000 --model-config model.json --tokenizer tokenizer.json \
  --auth-mode bearer --api-keys "key1,key2" --rate-limit-rpm 60 \
  --firewall-config firewall.yaml --tls-enabled --tls-certfile cert.pem --tls-keyfile key.pem

# Experiment tracking
sneppx-experiment list --experiment my-exp
sneppx-experiment view run_abc123
sneppx-experiment compare run_1 run_2 run_3
sneppx-experiment export --output runs.json
```

Also available via `snepp <train|serve|experiment>` (npm-style dispatcher).

---

## Architecture

```
SNEPPX-ALG (v0.9.5.748)
├── Core Tensor Engine (C/CUDA)
│   ├── SIMD GEMM (AVX2/AVX-512) + Strassen fallback
│   ├── CUDA Kernels (GEMM, Conv, Flash Attention v2/v3, LayerNorm, Softmax, AdamW fused)
│   ├── Autodiff Engine (reverse-mode, 100+ ops)
│   ├── Memory Management (pools, streams, events, pinned/managed)
│   └── RNG (Philox, Xavier/Kaiming init, fused dropout)
├── Python Bindings (Phases 1-7, 37 modules)
│   ├── Phase 1: Foundation — c_loader, c_types, dispatch factory
│   ├── Phase 2: Crypto — 6 families (sign, kem, symmetric, hash, kdf, util)
│   ├── Phase 3: Security S1-S9 — 10 modules (memory, network, monitor, vault, updates, verify, pen-test, obfuscation, AI safety, middleware)
│   ├── Phase 4: Algorithms — ARC, FM, HSS, NPE, SER
│   ├── Phase 5: Kernels — attention, arch (GELU/Mamba2), memory, thread, tensor_expr, simd_gemm, logger
│   ├── Phase 6: Infrastructure — net (topology, RDMA, gRPC, NCCL), drivers (CUDA/ROCm/TPU)
│   └── Phase 7: ASM — CPU features, crypto assembly (AES-NI, SHA-NI, AVX2), MASM build
├── Neural Architectures
│   ├── Vision: ViT (Tiny/Small/Base/Large/Huge), DeiT, Swin, MAE
│   ├── LLMs: LLaMA 2/3 (7B/13B/70B), Mistral 7B, Qwen2 (7B/72B), DeepSeek V2 (Lite/Full)
│   └── Custom: HSS (Hierarchical State Space), SER (Mixture of Experts), ARC (Adversarial), NPE (Neural Program Engine), FM (Feature Map)
├── Training Pipeline
│   ├── Optimizers: AdamW, Lion, LAMB, Sophia, RAdam, AdaFactor, SM3, Demeter, CaProp, SOAP, ScheduleFree, DistributedAdam, OrthoAdam
│   ├── Schedulers: Cosine, OneCycle, Polynomial, LinearWarmupCosineDecay, TriStage, ReduceLROnPlateau, Chained/Sequential
│   ├── AMP/GradScaler, Gradient Checkpointing, Gradient Clipping
│   └── ZeRO-1/2/3, Pipeline/Tensor/Expert Parallelism, Elastic Training + Fault Tolerance
├── Model Compression
│   ├── Quantization: INT8 (sym/asym/channel), INT4 (packed), FP8 (E4M3/E5M2), AWQ, GPTQ
│   ├── Pruning: Magnitude, L1 Channel, Taylor, Global Magnitude, Movement, Soft
│   └── Distillation: KD, Attention Transfer, Feature Matching, CRD, Multi-teacher, Online
└── Security (S0-S9, 21,809+ lines C + 31 Python test suites)
    ├── S0: Build Integrity (SBOM, Reproducible Builds)
    ├── S1: Memory Hardening (Guard Pages, Canaries, ASLR, Locked Memory)
    ├── S2: Code Obfuscation (CFG Flattening, Instruction Substitution, Opaque Predicates, String Encryption, VM Obfuscation)
    ├── S3: Runtime Monitoring (Integrity, Container Breakout, ML Anomaly)
    ├── S4: Network Security (TLS 1.3, Noise, QUIC, mTLS, DDoS, Port Knocking)
    ├── S5: AI Safety (RLHF, Prompt Injection Defense, Output Verification, Watermarking)
    ├── S6: AI Sanitizer (Semantic Injection Detection, Encoded Attack Decoder)
    ├── S7: Supply Chain (TUF Signed Updates, bsdiff Deltas, A/B Partitions, TPM)
    ├── S8: Formal Verification (TLA+ Parser, LTL Model Checking, Lean 4 Export)
    └── S9: Penetration Testing (CVE Scanner, Fuzzer, Red Team, Compliance Auto-checker)
```

---

## Installation

### From PyPI
```bash
pip install sneppx-alg
pip install sneppx-alg[serve]   # for inference server
pip install sneppx-alg[hf]      # for HuggingFace integration
pip install sneppx-alg[dev]     # development dependencies
```

### From Source
```bash
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
pip install -e .
```

### With CUDA Support
```bash
# Requires CUDA 12.0+, cuDNN, MSVC/GCC/Clang
pip install sneppx-alg[cuda]
# Or build from source with: cmake -B build -DSNEPPX_BUILD_CUDA=ON
```

---

## Python API Overview

### Core Modules
```python
from SneppX_ALG import (
    # Tensor & autograd
    Tensor, Dtype, Device, Linear, Embedding, Dropout,
    LayerNorm, RMSNorm, GELU, SiLU, ReLU, Sequential,
    MultiheadAttention, TransformerBlock, Transformer,
    
    # Optimizers
    Optimizer, SGD, AdamW, Lion, LAMB, LARS, AdaFactor,
    RAdam, Sophia, Adan, ScheduleFreeAdamW,
    
    # Schedulers
    CosineAnnealingLR, OneCycleLR, LinearWarmupCosineDecay,
    ReduceLROnPlateau, PolynomialLR, TriStageLR,
    
    # Training
    Trainer, TrainConfig,
    
    # Data
    Dataset, TensorDataset, DataLoader, DistributedSampler,
    
    # Distributed
    init_process_group, all_reduce, barrier, get_rank, get_world_size,
    
    # Model Zoo
    from_pretrained, get_model_config, build_transformer_from_config,
    LlamaConfig, MistralConfig, Qwen2Config, DeepSeekV2Config,
    
    # Quantization
    QuantMode, quantize_int8_sym, quantize_fp8_e4m3, awq_quantize,
    gptq_quantize, QuantizedLinear,
    
    # Checkpointing
    CheckpointWriter, CheckpointReader, CheckpointCoordinator,
    HeartbeatMonitor, ElasticTrainer, FaultToleranceManager,
    
    # Profiling
    Profiler, Timer, MemoryTracker, TrainProfiler, timeit,
    
    # Security (S1-S9)
    SecureAllocator, StackCanary, ASLR, DDoSMitigation,
    IdentityManager, TransportSecurity, IntegrityMonitor,
    KeyVault, SignedUpdate, LTLProperty, NetworkFuzzer,
    ObfuscationPipeline, RLHFSafety, AIPromptFilter, AIOutputVerifier,
    SecurityConfig, SecurityMiddleware, AuthConfig, PromptFilterConfig,
    
    # Algorithms
    AlgoHSSModel, AlgoSERModel, ARCAttackSim, FMController,
    NPECompiler, NPEVM,
    
    # Kernels
    Attention, DifferentialAttention, FlexAttention, ArchOps, Mamba2,
    MemoryPool, WorkStealingPool, TensorExprCompiler, SimdGemm,
    
    # Infrastructure
    Topology, SocketComm, RDMA, CUDADriver, ROCmDriver, TPUDriver,
    
    # ASM
    CPUFeatures, AESNI, SHANI, ChaCha20AVX2, Poly1305ASM,
    Ed25519ASM, ConstantTimeOps, FirewallASM,
)
```

---

## Supported Architectures

| Category | Models |
|----------|--------|
| **Vision** | ViT (Tiny/Small/Base/Large/Huge), DeiT, Swin (Tiny/Small/Base/Large), MAE |
| **LLMs** | LLaMA 2/3 (7B/13B/70B), Mistral 7B, Qwen2 (7B/72B), DeepSeek V2 (Lite/Full) |
| **Custom** | HSS (Hierarchical State Space), SER (Mixture of Experts), ARC (Adversarial), NPE (Neural Program Engine), FM (Feature Map) |

---

## Security

SNEPPX-ALG implements **10 layers of security (S0-S9)** — fully implemented in C with Python bindings:

| Layer | Description | Status |
|-------|-------------|--------|
| S0 | Build Integrity (SBOM, Reproducible Builds, TUF) | ✅ Complete |
| S1 | Memory Hardening (Guard Pages, Canaries, ASLR, Locked Memory, W^X) | ✅ Complete |
| S2 | Code Obfuscation (CFG Flattening, Instruction Substitution, Opaque Predicates, String Encryption, VM Obfuscation) | ✅ Complete |
| S3 | Runtime Monitoring (Integrity, Container Breakout, ML Anomaly, File System) | ✅ Complete |
| S4 | Network Security (TLS 1.3, Noise NK/XX/IK, QUIC, mTLS, DDoS, Port Knocking) | ✅ Complete |
| S5 | AI Safety (RLHF, Differential Privacy, Prompt/Output Filters, Watermarking, Adversarial Smoothing) | ✅ Complete |
| S6 | AI Sanitizer (Semantic Injection, Encoded Attack Decoder, Token Anomaly Scoring, Model Inversion Defense) | ✅ Complete |
| S7 | Supply Chain (TUF Multi-Role Keys, bsdiff Deltas, A/B Partitions, TPM PCR, Canary Rollout) | ✅ Complete |
| S8 | Formal Verification (TLA+ Parser, LTL Model Checking, Symbolic Execution, Lean 4 Export) | ✅ Complete |
| S9 | Penetration Testing (CVE Scanner, Fuzzer, API Security, Supply Chain Audit, Compliance Auto-checker) | ✅ Complete |

**4-Ring Firewall Architecture** (v0.9.4.467+):
- **Transport Ring**: TLS 1.3/mTLS, cert pinning, ALPN
- **Network Ring**: CIDR allow/deny, rate limiting (token bucket), connection tracking, port knocking
- **Application Ring**: Injection filter (SQLi/XSS/command), path traversal, concurrent limiter
- **Security Middleware**: `set_security()` with firewall overrides, `check_firewall()` in pipeline

**MASM x64 Hot-Path Routines**:
- `ip_match.asm` — CIDR matching with SIMD
- `rate_counter.asm` — Token bucket rate limiting
- `conn_track.asm` — Connection state tracking
- `port_knock.asm` — Port knock sequence validation

---

## Testing

All **31 test suites** (300+ tests) pass on pure-Python fallback (no C compiler needed):

```bash
# Run all tests
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/ -v

# Run specific phase
python tests/python/test_crypto_sign.py
python tests/python/test_secure_memory.py
python tests/python/test_algo_hss.py
python tests/python/test_c_attention.py
python tests/python/test_net_bindings.py
python tests/python/test_asm_bridge.py
```

---

## Performance

| Model | Device | Throughput | Latency |
|-------|--------|------------|---------|
| ViT-B/16 | A100 | ~1,200 img/s | 4.2ms |
| LLaMA-7B | A100 | ~2,500 tok/s | 8.5ms |
| Mistral-7B | A100 | ~2,800 tok/s | 7.8ms |

*Benchmarks on NVIDIA A100 80GB, batch size 32*

---

## Documentation

- [Installation Guide](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [GPU Setup](docs/getting-started/gpu-setup.md)
- [API Reference](docs/api/)
- [Advanced Guides](docs/advanced/)
- [Security Architecture](docs/security/)
- [CHANGELOG](CHANGELOG.md)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{sneppx-alg,
  title = {SNEPPX-ALG: Secure Neural Architecture v0.9.5.748},
  author = {Ammar [SNEPPX]},
  year = {2026},
  url = {https://github.com/ammar49-cyber/sneppx-alg}
}
```

---

## Links

- **GitHub**: https://github.com/ammar49-cyber/sneppx-alg
- **PyPI**: https://pypi.org/project/sneppx-alg/
- **Documentation**: https://sneppx-alg.org
- **Arix-Site**: https://github.com/ammar49-cyber/Arix-Site
- **Security**: security@sneppx.org

---

*Release v0.9.5.748 — 130K+ lines, 630+ files, 37 Python binding modules, 31 test suites, 3 CLI commands*