# ARIX-Algo

**Next-generation AI architecture with security built into the foundation.**

ARIX-Algo is a modular, 5-component neural architecture combining state-space models, sparse expert routing, adversarial robustness, a neural program executor, and federated memory — all exposed through C and Python APIs.

## Components

| Component | Description |
|-----------|-------------|
| **HSS** — Hierarchical State Space | Multi-layer SSM with ZOH discretization, sequential scan, and hierarchical scan stub. Projection → layernorm → discretize → scan per layer. |
| **SER** — Sparse Expert Routing | Softmax + top-k (greedy/noisy) routing, ReLU/GELU/Swish experts, load-balance loss, multi-layer model with weighted combine. |
| **ARC** — Adversarial Robustness Core | Input guard (L2 anomaly detection), gradient obfuscator (noise+clamp), output verifier (cosine consistency, history smoothing), multi-metric security scoring, FGSM/PGD/CW attack simulation. |
| **NPE** — Neural Program Executor | 16-register VM with 14 opcodes (MATMUL, ATTENTION, SOFTMAX, LAYERNORM, etc.), 64K memory pool, MLP/attention compiler, static verifier. |
| **FM** — Federated Memory | Per-node memory banks with euclidean similarity, LRU eviction, retention-based forgetting; trust-weighted all-reduce sync with Laplace DP noise, gradient compression, gossip/ring topology. |

## Project Stats

- **91 source files, ~12,300 lines** (C, C++, Python)
- **25 C tests + 5 Python tests = 30 passing tests**
- **5 training demos + 1 federation demo** (NaN/Inf-free verified)

## Build

Depends on CMake 3.20+, a C11/C++17 compiler, and Python 3.11+ with pybind11 for bindings.

```bash
git clone https://github.com/ammar49-cyber/nextgen-arixalgo.git
cd nextgen-arixalgo
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -DBUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure -C Release
```

### Python Bindings

```bash
# Rebuild CMake with Python support
cd build
cmake .. -DARIX_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python) -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build . --config Release --target arix_algo_core
Copy-Item "src/python/arix_algo/Release/arix_algo_core.cp311-win_amd64.pyd" "src/python/arix_algo/arix_algo_core.pyd"
$env:PYTHONPATH="src\python;$env:PYTHONPATH"
pytest tests/python/ -v
```

## Architecture Overview

```
Input → [HSS] → [SER] → [ARC] → [NPE] → [FM] → Output
         SSM     MoE      Guard    VM       Fed
```

Each component feeds into the next in a composable pipeline, all managed by `ArixModel` in `src/arch/`.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Development](docs/DEVELOPMENT.md)
- [API Reference](docs/API.md)

## License

MIT — see [LICENSE](LICENSE)
