# Development

## Prerequisites

- **CMake** 3.16+
- **C11 compiler** (MSVC 2022, GCC 11+, Clang 14+)
- **C++20** for S2 obfuscation (optional)
- **Python 3.11+** for bindings (optional)
- **Git** with GPG or Ed25519 signing configured

## Quick Start

```bash
git clone https://github.com/ammar49-cyber/SNEPPX_ALG.git
cd SNEPPX_ALG
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_BUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

## Workflow

1. **Branch**: feature branches from `main`
2. **Develop**: write code, add tests, run locally
3. **Format**: `clang-format -i -style=file <files>`
4. **Test**: `ctest --output-on-failure`
5. **Commit**: `git commit -s -m "component: message"`
6. **Patch**: `git format-patch -1 HEAD`
7. **Submit**: email to [algoSNEPPX@gmail.com](mailto:algoSNEPPX@gmail.com)

## Project Layout

```
include/neural_core/     # Public headers (kernel, architecture, security)
kernel/                   # Core implementations (arch, tensor, attention, etc.)
tests/                    # Unit, integration, benchmark, security tests
examples/                 # Demo programs
bindings/                 # Python (pybind11) and Rust bindings
tools/                    # CLI utilities and benchmarks
scripts/                  # Build and development scripts
cmake/                    # CMake modules
docs/                     # Documentation
security/                 # S0-S3 security layer source
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SNEPPX_BUILD_TESTS` | ON | Build test suite |
| `SNEPPX_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `SNEPPX_BUILD_PYTHON` | OFF | Build Python bindings |
| `SNEPPX_BUILD_CUDA` | OFF | Build CUDA kernels |
| `SNEPPX_USE_ASAN` | OFF | AddressSanitizer |
| `SNEPPX_USE_UBSAN` | OFF | UndefinedBehaviorSanitizer |
| `SNEPPX_USE_LTO` | OFF | Link-Time Optimization |

## Testing

- All new features must include tests
- Run full suite before submitting patches
- Pre-existing failures: Argon2id (1 timing edge case), SER training (1 edge case)
- Timeouts: Ed25519 (slow test vectors), thread pool (sleep-based timing)

## Code Review

Patches are reviewed for:
- Correctness: does the code do what it claims?
- Style: does it follow STYLE_GUIDE.md?
- Safety: are all allocations checked? No buffer overflows?
- Tests: are new features adequately tested?
