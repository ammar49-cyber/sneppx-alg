# ARIX-Algo

> Next-generation AI architecture with security built into the foundation.
> Not patched later. Not bolted on. In every instruction.

## Status

| Version | Date | Status |
|---------|------|--------|
| v0.1.0 | 2026-06-24 | Research prototype — core structure implemented, training not yet enabled |

## What This Is

ARIX-Algo is a new class of AI architecture. Not a model. Not a chatbot. A foundation.

Five components:

- **HSS** — Hierarchical State Space. O(n log n) sequence modeling.
- **SER** — Sparse Expert Routing. Dynamic parameter efficiency.
- **ARC** — Adversarial Robustness Core. Security in the weights.
- **NPE** — Neural Program Executor. Guaranteed-correct paths.
- **FM** — Federated Memory. Learn collectively. Forget nothing.

## What Works Now (v0.1.0)

| Component | Status | What Works | What Doesn't |
|-----------|--------|-----------|--------------|
| Tensor Core | ✅ Real | Multi-dim arrays, 5 dtypes, row-major, 50+ ops | GPU (CPU only) |
| Memory | ✅ Real | Aligned allocation, secure zero, guard pages | NUMA optimization |
| Thread Pool | ⚠️ Stub | Single-threaded fallback | Real parallelism |
| HSS | ⚠️ Partial | Forward pass, sequential scan, first-order discretization | Parallel scan, training |
| SER | ⚠️ Partial | Top-k routing, expert forward, load balance loss | Learned gating |
| ARC | ⚠️ Partial | Z-score input guard, gradient noise, output check | Formal proofs |
| NPE | ⚠️ Partial | VM executes, compilers generate programs | JIT, formal verification |
| FM | ⚠️ Partial | Single-node memory banks, sync stubs | Real distributed |
| Autodiff | ❌ Stub | Structure only | Backward pass (does nothing) |
| Optimizer | ❌ Stub | Structure only | Parameter updates |
| Python API | ❌ Stub | Package installs, `hello()` works | Model, Trainer classes |
| Security S0 | ✅ Real | Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3, Argon2id | — |
| Security S1 | ✅ Real | Guard pages, canaries, ASLR, locked memory | — |
| Security S2 | ⚠️ Partial | Control flow flattening, string encryption stubs | Full obfuscation |
| Security S3 | ⚠️ Partial | Behavioral monitor structure | Real anomaly detection |

## What This Means

You can:

- ✅ Build the project from source
- ✅ Run all tests (50 pass)
- ✅ Run demos (HSS, SER, ARC, NPE, FM)
- ✅ Audit the security code (S0-S1 are production-grade)
- ✅ Read the architecture and understand the design
- ✅ Contribute to development

You cannot:

- ❌ Train a model
- ❌ Generate text
- ❌ Benchmark against GPT-2
- ❌ Use as PyTorch replacement
- ❌ Deploy to production

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DARIX_BUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### Quick Start

```c
#include "arix_tensor.h"
#include "arix_hss.h"

int main() {
    size_t shape[] = {4, 8, 16};
    ArixTensor* input = arix_tensor_randn(shape, 3, ARIX_FLOAT32);

    ArixHSSConfig config = arix_hss_config_default();
    ArixHSSModel* model = arix_hss_model_create(&config, 42);

    ArixTensor* output;
    arix_hss_forward(model, input, &output);

    arix_tensor_print(output);
    return 0;
}
```

### Python (Stub)

```python
from arix_algo import hello
print(hello())  # "ARIX-Algo v0.1.0 — Core tensor operations implemented in C"
```

Full Python API in v0.5.0.

## Security

Security is not a feature. It is the architecture.

| Phase | Status | Description |
|-------|--------|-------------|
| S0 | ✅ Complete | Cryptographic primitives |
| S1 | ✅ Complete | Secure memory |
| S2 | ⚠️ In Progress | Obfuscation engine |
| S3 | ⚠️ In Progress | Behavioral monitor |
| S4-S9 | ⏳ Planned | Network, AI sanitizer, UI, updates, formal verification, penetration |

## Roadmap

| Version | Timeline | Milestone |
|---------|----------|-----------|
| v0.5.0 | 6 months | Trainable on CPU, WikiText-2, autodiff real |
| v1.0 | 18 months | 7B parameters, competitive with GPT-2, CUDA |
| v2.0 | 4 years | 70B parameters, 1M LOC, competitive with LLaMA-3 |
| v3.0 | 7 years | 1T parameters, 5M LOC, competitive with GPT-4 |
| v4.0 | 10 years | 10T parameters, 15M LOC, beyond existing systems |

## License

MIT for algorithm. Closed-source for website and distribution.

## Governance

BDFL: Ammar [ARIX]

Patches: patches@arix.dev

Security: security@arix.dev

## Links

- Website: https://aixsite.vercel.app
- GitHub: https://github.com/ARIX-Algo
- Twitter: https://x.com/Arixdrv
- Instagram: https://www.instagram.com/algoarix/
- YouTube: https://www.youtube.com/@ArixAlgo

## Citation

```bibtex
@software{arix_algo_2026,
  author = {Ammar [ARIX]},
  title = {ARIX-Algo: Next-generation AI architecture},
  url = {https://github.com/ARIX-Algo},
  year = {2026}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
