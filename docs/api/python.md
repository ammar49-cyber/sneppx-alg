# Python API Reference

## Status: ❌ Stub

The Python API is a placeholder in v0.1.0. The package installs and `hello()` works, but no classes or methods are implemented.

## Installation

```bash
pip install -e src/python
```

## Quick Start

```python
from arix_algo import hello

print(hello())
# Output: "ARIX-Algo v0.1.0 — Core tensor operations implemented in C"
```

## Planned API (v0.5.0+)

```python
import arix_algo as ax

# Create a tensor
t = ax.Tensor.randn((4, 8, 16), dtype=ax.float32)

# Create a model
model = ax.HSSModel(d_model=64, d_state=16, num_layers=2)

# Forward pass
output = model(t)

# Training
loss = model.train_step(t, target)
loss.backward()
model.optimizer.step()
```

## Components (v0.5.0)

| Component | Planned API |
|-----------|-------------|
| Tensor | `ax.Tensor` — numpy-compatible multi-dimensional array |
| HSS | `ax.HSSModel` — state space sequence model |
| SER | `ax.SERModel` — sparse mixture of experts |
| ARC | `ax.ARCModel` — adversarial robustness guard |
| NPE | `ax.NPEModel` — neural program executor |
| FM | `ax.FMModel` — federated memory bank |
| Autodiff | `ax.Variable`, `ax.Tape`, `ax.no_grad()` |
| Optimizer | `ax.SGD`, `ax.Adam` |
| Trainer | `ax.Trainer` — training loop with callbacks |
| Security | `ax.s0`, `ax.s1` — crypto and secure memory |

## Environment Setup

```powershell
# Recommended: use a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/macOS

# Install with build
pip install -e src/python
```

The `.pyd` extension module must be in the Python path. When building with `ARIX_BUILD_PYTHON=ON`, CMake copies it automatically.
