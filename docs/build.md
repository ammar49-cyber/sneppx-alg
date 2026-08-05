# Build Instructions

## 1. Prerequisites

**Required:**
- CMake 3.16+
- C11/C++20 compiler (MSVC 19.44+, GCC 11+, Clang 14+)
- Python 3.9+ (for bindings)

**Optional:**
- CUDA 12.0+ (GPU optimizer acceleration, `SNEPPX_BUILD_CUDA=ON`)
- Doxygen (API docs)

## 2. Quick Build (All Targets)

**Windows (MSVC):**
```powershell
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build --config Release
```

**Linux/Mac (GCC/Clang):**
```bash
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build -j$(nproc)
```

## 3. Build Individual Targets

```powershell
# Core kernel library
cmake --build build --config Release --target neural_core_kernel

# Architecture algorithms
cmake --build build --config Release --target neural_architecture_layer

# Security C library
cmake --build build --config Release --target neural_security_c

# Security C++ obfuscation
cmake --build build --config Release --target neural_security_cpp

# Python bindings
cmake --build build --config Release --target _SNEPPX_c

# CUDA kernels (requires CUDA SDK)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_CUDA=ON
cmake --build build --config Release --target neural_cuda_kernels
```

## 4. Build Options

| CMake Variable | Default | Description |
|----------------|---------|-------------|
| `SNEPPX_BUILD_CUDA` | OFF | Enable CUDA kernel compilation |
| `SNEPPX_BUILD_PYTHON` | OFF | Build Python bindings |
| `SNEPPX_BUILD_TESTS` | ON | Build test executables |
| `SNEPPX_BUILD_EXAMPLES` | ON | Build example programs |
| `SNEPPX_BUILD_VULKAN` | OFF | Vulkan reference backend |
| `SNEPPX_BUILD_TPU` | OFF | TPU reference backend |
| `SNEPPX_BUILD_HTTP` | OFF | HTTP transport backend |
| `SNEPPX_BUILD_ZK` | OFF | Zero-knowledge backend |
| `SNEPPX_ENABLE_ASAN` | OFF | AddressSanitizer |
| `SNEPPX_ENABLE_UBSAN` | OFF | UndefinedBehaviorSanitizer |

## 5. Debug Build

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

Debug builds disable optimization and enable assertions. Use `-DSNEPPX_S1_SECURE_MEMORY=OFF` for faster debug iteration.

## 6. Python Package Build

```powershell
# Editable install
pip install -e .

# With CUDA support
pip install -e .[cuda]

# Build wheel
pip install build
python -m build --wheel
```

## 7. Running Tests

```powershell
# Python tests (no build required)
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/ -v

# C tests (build required)
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure

# Specific test
ctest -C Release -R test_kyber

# Sanitizer tests (Windows)
powershell -File scripts/run_sanitizers.ps1
```

## 8. Common Issues

| Issue | Fix |
|-------|-----|
| `pybind11_add_module: Unknown CMake command` | Set `pybind11_DIR` or ensure FetchContent can reach GitHub |
| `cuda_runtime.h: No such file or directory` | Build without CUDA: `-DSNEPPX_BUILD_CUDA=OFF` |
| `LNK2001: unresolved external symbol` | Build `neural_core_kernel` and `neural_security_c` before `_SNEPPX_c` |
| MSVC "no VLAs" | Use `calloc`/`free` instead of stack-allocated VLAs |
| `__int128 undeclared` on MSVC | Guard with `#ifndef NO_UINT128` (see `x25519.c` pattern) |

## 9. Build Directories

The project uses multiple build directories:

- `build/` — Standard debug build
- `build_bench/` — Benchmark-optimized release build
- `build_test/` — Test build
- `build_py/` — Python-specific build

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full development workflow.
