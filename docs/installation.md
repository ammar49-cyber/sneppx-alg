# Installation Guide

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| CMake | 3.16 | Build system |
| C compiler | C11 | MSVC 2022, GCC 11+, Clang 14+ |
| C++ compiler | C++20 | Only needed for S2 obfuscation engine |
| Python | 3.11+ | Optional (for bindings) |
| pybind11 | 2.10+ | Optional (for Python bindings) |

## Linux

### Ubuntu / Debian

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Build
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON
cmake --build . -j$(nproc)

# Test
ctest --output-on-failure

# Install (optional)
sudo cmake --install .
```

### Fedora / RHEL

```bash
sudo dnf install -y gcc gcc-c++ cmake git
# Same build steps as above
```

### Arch Linux

```bash
sudo pacman -S gcc cmake git
# Same build steps as above
```

## macOS

### Intel (x86_64)

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake gcc

# Build
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON
cmake --build . -j$(sysctl -n hw.ncpu)

# Test
ctest --output-on-failure
```

### Apple Silicon (ARM64)

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake

# Build (using Apple Clang; ARM NEON optimizations auto-detected)
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON
cmake --build . -j$(sysctl -n hw.ncpu)

# Test
ctest --output-on-failure
```

## Windows

### Native (Visual Studio)

```powershell
# Prerequisites: Visual Studio 2022 with "Desktop development with C++" workload
# CMake 3.16+ (included with VS 2022)

# Open "Developer Command Prompt for VS 2022"
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DSNEPPX_BUILD_TESTS=ON
cmake --build . --config Release

# Test
ctest -C Release --output-on-failure
```

### PowerShell Script

```powershell
# From the repository root
.\scripts\build.ps1 -Config Release -Tests
.\scripts\test.ps1 -Config Release
```

### WSL (Windows Subsystem for Linux)

```bash
# Follow Linux (Ubuntu) instructions above
# Performance is ~90% of native Linux
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SNEPPX_BUILD_TESTS` | ON | Build test suite |
| `SNEPPX_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `SNEPPX_BUILD_PYTHON` | OFF | Build Python bindings |
| `SNEPPX_BUILD_CUDA` | OFF | Build CUDA kernels (future) |
| `SNEPPX_USE_ASAN` | OFF | Enable AddressSanitizer |
| `SNEPPX_USE_UBSAN` | OFF | Enable UndefinedBehaviorSanitizer |
| `SNEPPX_USE_LTO` | OFF | Enable Link-Time Optimization |

## Build Configurations

| Config | Flags | Use Case |
|--------|-------|----------|
| Debug | `-g -O0` | Development, debugging |
| Release | `-O3 -DNDEBUG` | Performance testing, benchmarks |
| RelWithDebInfo | `-O2 -g` | Profiling with debug symbols |

## Verify Installation

```bash
# Run full test suite
ctest --output-on-failure

# Run specific test
ctest -R test_tensor

# Run benchmarks
./tests/benchmark/bench_tensor
./tests/benchmark/bench_autodiff
```

## Troubleshooting

### Build fails: "cannot open program database"

This is a known MSVC parallel build issue. The CMakeLists.txt includes `/FS` flag to fix it. If you still see it:

```bash
cmake --build . --config Release -j1
```

### Test fails: "Ed25519 verify" or "Argon2 timing"

These are pre-existing S0 edge case failures. They do not affect security. See [SECURITY.md](../SECURITY.md) for details.

### Python import fails

Ensure the `.pyd` file is in the Python path:

```powershell
# Windows
copy build\Release\SneppX_ALG_core.pyd src\python\SneppX_ALG\
```

### CMake not found

```bash
# Ubuntu
sudo apt-get install cmake

# macOS
brew install cmake

# Windows
# Install Visual Studio 2022 with CMake component
```

### "undefined reference to" errors

Ensure you're linking against the correct libraries. Core tests link `SNEPPX_core;SNEPPX_arch`. Security tests additionally link `SNEPPX_security_c` or `SNEPPX_security_cpp`.
