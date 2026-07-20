# Contributing to SNEPPX-Alg

## Governance

**BDFL**: Ammar [SNEPPX] <algoSNEPPX@gmail.com>

All decisions final. No voting. No committees.

## How to Contribute

### 1. Clone & Branch

```bash
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
git checkout -b feature-name
```

### 2. Development Workflow

```bash
# Build with current settings
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run all tests
cd build && ctest -C Release --output-on-failure
```

### 3. Verify Changes

```bash
# Check formatting
npm run lint  # (if available)

# Type checking
npm run typecheck  # (if available)

# Run specific test category
ctest -C Release -R test_autodiff --output-on-failure
```

### 4. Commit Changes

```bash
# Stage files
git add -A

# Create conventional commit message
git commit -m "feat(module): add new functionality"
```

## Contribution Guidelines

### Pre-Commit Checklist

- [ ] The PR must reference a GitHub issue (if applicable) via `Fixes #123` or `Closes #456`
- [ ] Test files updated and passing
- [ ] Documentation updated where relevant
- [ ] Code reviews complete (if applicable)

### Code Quality Requirements

- [ ] Compiles without warnings (`-Wall -Wextra -Wpedantic`)
- [ ] All tests pass via ctest
- [ ] Follows existing code style (see STYLEGUIDE.md)
- [ ] Consistent with project conventions

### Documentation Standards

- [ ] Include docstrings for all public functions
- [ ] Document APIs in the official documentation
- [ ] Update relevant user-facing documentation

## Testing

### Running Tests

```bash
# Run all tests
cd build && ctest -C Release --output-on-failure

# Individual test categories
ctest -C Release -R test_unicode   # Character encoding tests
ctest -C Release -R test_crypto    # Crypto module tests
ctest -C Release -R test_tensor    # Tensor engine tests
ctest -C Release -R test_security  # Security module tests
```

### Test Coverage

Project maintains comprehensive test coverage across:
- Unit tests for all modules (C/C++/Rust)
- Python test suite (300+ tests in tests/python/)
- Benchmark tests (perf regression detection)
- Security tests (fuzzing, validation)

## Build & Run Local

### Python Bindings

```python
from SneppX_ALG import Tensor, Linear, AdamW

# Basic usage
model = Linear(768, 10)
optimizer = AdamW(model.parameters(), lr=0.001)
```

### C/C++ Integration

```c
#include "neural_core/tensor.h"

// Create and manipulate tensors
SNEPPX_Tensor* t = sneppx_tensor_create(2, 2);
sneppx_tensor_free(t);
```

### Rust Bindings

```rust
use neural_core_algo::Tensor;

fn main() {
    let t = Tensor::new(vec![2, 2]);
    let ones = Tensor::ones(vec![2, 3]);
}
```

## Development Tools

### Required Dependencies

| Tool | Purpose |
|------|---------|
| CMake >= 3.16 | Build system |
| GCC/Clang >= 7 | C/C++ compilation |
| Python 3.9+ | Python bindings |
| Rust >= 1.56 | Rust bindings |
| CUDA >= 11.8 | GPU kernels (optional) |

### Build Options

```bash
# Debug build with sanitizers
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_USE_SANITIZERS=ON

# Release with Python bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_PYTHON=ON

# Release with CUDA support
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_CUDA=ON
```

## Package Publishing

### Python Package

```bash
# Build sdist and wheel
python -m build --wheel --sdist

# Install locally
pip install dist/sneppx_alg-0.9.7.890e-py3-none-any.whl
```

### Rust Crates

```bash
# Package the workspace
cargo package --allow-dirty

# Add dependencies for publishing
rustup target add x86_64-unknown-linux-gnu wasm32-unknown-unknown
```

## Licensing & Attribution

### Contributor License Agreement (CLA)

- No corporate CLAs - individual contributors only
- All contributions under MIT License
- Sign your commits with your GPG or Ed25519 key

### Patent Grants

These code contributions grant a perpetual, worldwide, royalty-free license under all patents held by the maintainer for those contributions.

## Appendix

### Supported Architectures

- x86-64 (AVX2, AVX-512)
- ARMv8-A (NEON)
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)

### Performance Profile

- **Tensor Engine**: 100% SIMD-optimized
- **CPU**: Work-stealing scheduler with thread affinity
- **GPU**: CUDA kernels for accelerated workloads
- **Security**: S0-S9 security layers for defense-in-depth

### Architecture Features

- **Neural Architectures**: HSS, MoE, ARC, NPE, FM
- **Security Layers**: Post-quantum crypto, obfuscation
- **Distributed Training**: ZeRO, NCCL, RDMA
- **Deployment**: ONNX, SafeTensors, PyTorch formats

### API Overview

- **C API**: Core tensor and security interfaces
- **Python API**: High-level neural networks, training loops
- **Rust API**: Memory-safe bindings for systems programming
- **CLI Tools**: `sneppx-train`, `sneppx-serve`, `sneppx-experiment`

### Future Priorities

1. **Hardware**: TPU, NPU integration, more accelerators
2. **Protocols**: QUIC, gRPC, HTTP/3 support
3. **Compliance**: SOC 2, ISO certifications
4. **AI Safety**: Federated learning, differential privacy

## Quick Start (5-min guide)

```bash
# Clone & build
mkdir sneppx-alg && cd sneppx-alg
git clone https://github.com/ammar49-cyber/sneppx-alg.git .
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Test
cd build && ctest -C Release --output-on-failure

# Install Python package
pip install -e bindings/python
```

Ready to contribute? Open an issue or directly make changes and submit a pull request!

---

**Questions?**
Contact: algoSNEPPX@gmail.com

**Documentation**: https://github.com/ammar49-cyber/sneppx-alg

**GitHub**: https://github.com/ammar49-cyber/sneppx-alg
