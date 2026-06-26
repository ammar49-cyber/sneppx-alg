# Contributing Guide

## Overview

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contributor agreement, patch format, coding style, and governance model.

## Development Workflow

### Setup

```bash
git clone https://github.com/ARIX-Algo/arix-algo.git
cd arix-algo
mkdir build && cd build

# Debug build for development
cmake .. -DCMAKE_BUILD_TYPE=Debug -DARIX_BUILD_TESTS=ON
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Code Organization

| Directory | Content |
|-----------|---------|
| `src/core/` | Foundation: tensor, memory, thread, autodiff, optimizer |
| `src/arch/` | Algorithm pipeline: HSS, SER, ARC, NPE, FM |
| `src/security/c/` | S0-S1 security primitives (C) |
| `src/security/cpp/` | S2-S3 security engine (C++) |
| `src/security/asm/` | Assembly helpers |
| `src/python/` | Python bindings |
| `tests/unit/` | Unit tests per component |
| `tests/integration/` | Integration tests |
| `tests/benchmark/` | Performance benchmarks |
| `tests/security/` | Security tests |
| `examples/` | Demos and examples |

### Testing

```bash
# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R test_tensor

# Run benchmarks
./tests/benchmark/bench_tensor
./tests/benchmark/bench_autodiff
```

### Build Options

```bash
cmake .. -DARIX_BUILD_TESTS=ON       # Build tests (default: ON)
cmake .. -DARIX_BUILD_BENCHMARKS=ON  # Build benchmarks (default: ON)
cmake .. -DARIX_BUILD_PYTHON=ON      # Build Python bindings
cmake .. -DARIX_USE_ASAN=ON          # Enable AddressSanitizer
cmake .. -DARIX_USE_UBSAN=ON         # Enable UndefinedBehaviorSanitizer
cmake .. -DARIX_USE_LTO=ON           # Enable Link-Time Optimization
```

### Presets

```bash
# Available presets
cmake --list-presets

# Use a preset
cmake --preset debug
cmake --preset release
cmake --preset relwithdebinfo
cmake --preset ninja-release
cmake --preset asan
```

## Contribution Areas

### Good First Issues

- Add missing tensor operations (see `src/core/tensor.h` for planned ops)
- Improve test coverage for edge cases
- Fix compiler warnings on non-MSVC platforms
- Add documentation for existing functions
- Optimize memory allocation paths

### Medium Difficulty

- Implement autodiff backward pass (C gradients)
- Add Python bindings for tensor operations
- Implement CUDA kernel stubs
- Add GPU detection and fallback logic
- Implement parallel thread pool

### Advanced

- Implement HSS parallel scan
- Implement SER learned gating
- Complete S2 obfuscation engine
- Complete S3 behavioral monitor
- NPE JIT compilation
- FM distributed synchronization

## Communication

- **Patches**: patches@arix.dev
- **Security**: security@arix.dev
- **Conduct**: conduct@arix.dev
- **No**: GitHub issues, Discord, Slack

## Code Review

Patches are reviewed by the BDFL. Expect feedback within 7 days. Address all comments and resend.
