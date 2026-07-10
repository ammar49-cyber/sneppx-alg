# SNEPPX-ALG

**Next-generation AI architecture with security built into the foundation**

[![Build Status](https://github.com/ammar49-cyber/sneppx-alg/workflows/CI/badge.svg)](https://github.com/ammar49-cyber/sneppx-alg/actions)
[![PyPI Version](https://img.shields.io/pypi/v/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

**SNEPPX-ALG** is a next-generation AI architecture that integrates security at every layer — from the tensor operations to the model deployment pipeline. Built for high-performance AI workloads with enterprise-grade security.

### Key Features

- **🚀 High-Performance Tensor Engine**: SIMD-optimized (AVX2/AVX-512), CUDA kernels for NVIDIA GPUs (Ampere/Hopper)
- **🔒 Security-First Design**: S0-S9 security layers (memory hardening, obfuscation, post-quantum crypto, AI safety)
- **🧠 Modern Architectures**: Vision Transformers (ViT, DeiT, Swin), LLMs (LLaMA, Mistral, Qwen2, DeepSeek V2), MAE
- **⚡ Distributed Training**: ZeRO-1/2/3, Pipeline/Tensor/Expert Parallelism, Elastic Training
- **🔧 Model Compression**: Quantization (INT8/FP8/AWQ/GPTQ), Pruning (Magnitude/Movement/Taylor), Distillation
- **📦 Deployment Ready**: ONNX Export, TensorRT Integration, Python/C++ APIs

---

## Quick Start

### Python Installation

```bash
pip install sneppx-alg
```

With CUDA support:
```bash
pip install sneppx-alg[cuda]
```

With HuggingFace integration:
```bash
pip install sneppx-alg[hf]
```

### Quick Example

```python
import sneppx_alg as snx
import numpy as np

# Create a model
model = snx.VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_classes=1000
)

# Run inference
x = snx.Tensor.from_numpy(np.random.randn(1, 3, 224, 224).astype(np.float32))
logits = model(x)
print(f"Output shape: {logits.shape}")  # [1, 1000]
```

---

## Architecture Overview

```
SNEPPX-ALG
├── Core Tensor Engine (C/CUDA)
│   ├── SIMD GEMM (AVX2/AVX-512)
│   ├── CUDA Kernels (GEMM, Conv, Attention)
│   ├── Autodiff Engine
│   └── Memory Management
├── Neural Architectures
│   ├── Vision: ViT, DeiT, Swin, MAE
│   ├── NLP: LLaMA, Mistral, Qwen2, DeepSeek V2
│   └── Custom: HSS, SER, ARC, NPE, FM
├── Training Pipeline
│   ├── Optimizers (AdamW, Lion, LAMB, Sophia...)
│   ├── Schedulers (Cosine, OneCycle, Polynomial...)
│   ├── AMP/GradScaler
│   └── Gradient Checkpointing
├── Distributed Training
│   ├── ZeRO-1/2/3
│   ├── Pipeline/Tensor/Expert Parallelism
│   └── Elastic Training + Fault Tolerance
├── Model Compression
│   ├── Quantization (INT8/FP8/AWQ/GPTQ)
│   ├── Pruning (Magnitude/Taylor/Movement)
│   └── Distillation (KD, Attention Transfer, CRD)
└── Security (S0-S9)
    ├── Memory Hardening
    ├── Code Obfuscation
    ├── Post-Quantum Crypto (Kyber, Dilithium)
    ├── AI Safety (RLHF, Guardrails)
    └── Formal Verification
```

---

## Installation

### From PyPI
```bash
pip install sneppx-alg
```

### From Source
```bash
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
pip install -e .
```

### With CUDA Support
```bash
# Requires CUDA 12.0+ and cuDNN
pip install sneppx-alg[cuda]
```

### Development Install
```bash
pip install -e .[dev,hf]
```

---

## Documentation

- [Installation Guide](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [GPU Setup](docs/getting-started/gpu-setup.md)
- [API Reference](docs/api/)
- [Advanced Guides](docs/advanced/)

---

## Supported Architectures

| Category | Models |
|----------|--------|
| **Vision** | ViT (Tiny/Small/Base/Large/Huge), DeiT, Swin (Tiny/Small/Base/Large), MAE |
| **LLMs** | LLaMA 2/3 (7B/13B/70B), Mistral 7B, Qwen2 (7B/72B), DeepSeek V2 (Lite/Full) |
| **Custom** | HSS, SER, ARC, NPE, FM |

---

## Performance

| Model | Device | Throughput | Latency |
|-------|--------|------------|---------|
| ViT-B/16 | A100 | ~1,200 img/s | 4.2ms |
| LLaMA-7B | A100 | ~2,500 tok/s | 8.5ms |
| Mistral-7B | A100 | ~2,800 tok/s | 7.8ms |

*Benchmarks on NVIDIA A100 80GB, batch size 32*

---

## Security

SNEPPX-ALG implements **9 layers of security (S0-S9)**:

| Layer | Description |
|-------|-------------|
| S0 | Build Integrity (SBOM, Reproducible Builds) |
| S1 | Memory Hardening (Guard Pages, Canaries, ASLR) |
| S2 | Code Obfuscation (Control Flow, String Encryption) |
| S3 | Network Security (DDoS, mTLS, ZTNA) |
| S4 | AI Safety (RLHF, Guardrails, Watermarking) |
| S5 | AI Sanitizer (Prompt Injection Defense) |
| S6 | Runtime Monitoring (Integrity, Container Breakout) |
| S7 | Supply Chain (SBOM, Signed Updates, SBOM) |
| S8 | Formal Verification (Model Checking) |
| S9 | Hardware Security (TEE, HSM, Side-Channel) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{sneppx-alg,
  title = {SNEPPX-ALG: Secure Neural Architecture},
  author = {Ammar [SNEPPX]},
  year = {2026},
  url = {https://github.com/ammar49-cyber/sneppx-alg}
}
```

---

## Links

- **GitHub**: https://github.com/ammar49-cyber/sneppx-alg
- **PyPI**: https://pypi.org/project/sneppx-alg/
- **Documentation**: https://sneppx-alg.readthedocs.io
- **Security**: security@sneppx.org