# Python API Reference

## Status: ⚠️ Partial (v0.1.0)

Python bindings are built via Pybind11 (71 KB `bindings.cpp`) when `SNEPPX_BUILD_PYTHON=ON`.
Ready for forward-pass usage; backward/training depends on autodiff implementation.

## Installation

```bash
pip install -e src/python
```

Build with Python bindings:
```bash
cmake -B build -DSNEPPX_BUILD_PYTHON=ON
cmake --build build
```

## Quick Start

```python
import SneppX_ALG as ax

# Create a tensor
t = ax.Tensor.randn((4, 8, 16), dtype=ax.float32)
```

## Components (v0.1.0)

| Component | Module | Status |
|-----------|--------|--------|
| Tensor | `ax.Tensor` `ax.tensor.*` | ✅ Creation, ops, reductions, IO, NN |
| Autodiff | `ax.Variable` `ax.Tape` | ⚠️ Forward only, backward is stub |
| Optimizer | `ax.SGD` `ax.Adam` | ⚠️ Forward step only |
| HSS | `ax.HSSModel` | ✅ Forward pass |
| SER | `ax.SERModel` | ✅ Forward pass |
| ARC | `ax.ARCModel` | ✅ Forward pass |
| NPE | `ax.NPEModel` | ✅ Compile + execute |
| FM | `ax.FMModel` | ✅ Read/write/sync |
| Trainer | `ax.Trainer` | ⚠️ Training loop stub |
| Security | `ax.s0` `ax.s1` | ✅ S0 crypto, S1 secure memory |

## Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e src/python
```

The `.pyd` extension module is copied automatically by CMake when `SNEPPX_BUILD_PYTHON=ON`.
