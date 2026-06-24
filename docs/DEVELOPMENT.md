# Development

## Prerequisites

- **C11/C++17 compiler** (MSVC 19.44+ on Windows, GCC 11+ or Clang 14+ on Linux)
- **CMake 3.20+**
- **Python 3.11+** with pybind11 3.0+ (for Python bindings)
- **Git**

## Build (C only)

```bash
# Configure
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -DBUILD_TYPE=Release

# Build
cmake --build . --config Release

# Test
ctest --output-on-failure -C Release
```

## Build (with Python bindings)

```bash
cd build
cmake .. -G "Visual Studio 17 2022" -DBUILD_TYPE=Release `
  -DARIX_BUILD_PYTHON=ON `
  -DPython_EXECUTABLE=$(which python) `
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build . --config Release --target arix_algo_core

# Copy .pyd to package dir (Windows)
Copy-Item "src/python/arix_algo/Release/arix_algo_core.cp311-win_amd64.pyd" `
          "src/python/arix_algo/arix_algo_core.pyd"

# Run Python tests
$env:PYTHONPATH="src\python;$env:PYTHONPATH"
pytest tests/python/ -v
```

## Test Suites

| Suite | Count | Command |
|-------|-------|---------|
| C Unit Tests | 21 | `ctest -C Release` |
| C Integration Tests | 4 | `ctest -C Release` |
| Python Tests | 5 | `pytest tests/python/ -v` |

All tests use a simple counter-based runner (no test framework dependency).

## Code Style

- **C**: K&R indentation, `snake_case` for functions/variables, `PascalCase` for types
- **C++**: pybind11 extension only — `camelCase` for methods, same C style for wrapped calls
- **Python**: `snake_case` following PEP 8
- No memory leaks: every `arix_malloc` has a matching `arix_free`, every `arix_tensor_create` has a `arix_tensor_destroy`
- All structs are zero-initialized via `memset`
- No BLAS dependency — all matmul uses nested loops

## Adding a New Component

1. Create header in `src/arch/include/arix_<name>.h`
2. Create implementation in `src/arch/src/<name>/`
3. Create tests in `tests/unit/<name>/`
4. Add CMake glob for sources (they're auto-discovered via `GLOB_RECURSE`)
5. Register in `ArixArchConfig` / `ArixModel` in `src/arch/include/arix_arch.h`
6. Wire into `arix_model_create` and `arix_model_forward` in `src/arch/src/arch/arch.c`

## Key Conventions

- Memory: `arix_malloc(size, alignment)` / `arix_free(ptr, size)` — aligned, secure-zero on free
- Errors: return non-zero on failure, NULL on allocation failure
- Ownership: caller is responsible for destroying any `ArixTensor*` returned from `*_forward` functions
- Thread safety: not guaranteed in v0.1 (single-threaded)
