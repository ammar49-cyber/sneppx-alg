# SNEPPX Algo — Agent Guide

## Project Overview
SNEPPX Algo is a cognitive processing system implementing neural architecture search, hierarchical state spaces, mixture of experts, and a full S0–S9 security layer. Written in C11 + C++20, targeting x86-64 (MSVC 19.44, GCC, Clang).

## Build Commands
```powershell
# Configure (debug)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Configure (release)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build a specific target
cmake --build build --config Release --target neural_security_c

# Build all
cmake --build build --config Release

# Run tests
cd build && ctest -C Release --output-on-failure

# Run a specific test
ctest -C Release -R test_kyber --output-on-failure
```

## Project Conventions
- **Language**: C11 (`.c`), C++20 (`.cpp`), MASM (`.asm`)
- **No VLAs**: MSVC C11 doesn't support them — use `calloc`/`free`
- **No `__int128` on MSVC**: Use `#ifndef NO_UINT128` guards (see `x25519.c` pattern)
- **No K&R-style declarations**: All functions must use modern C prototype syntax
- **Include paths**: Short form from `include/neural_core/security/` — e.g., `#include "kyber.h"`
- **ASM syntax**: MASM (Intel syntax) for x86-64, placed in `security/crypto/asm/x86_64/`
- **CMake**: Uses `file(GLOB_RECURSE)` — new `.c`/`.asm` files are auto-discovered
- **Memory allocation**: `SNEPPX_secure_malloc`/`SNEPPX_secure_free` for sensitive data

## Key Files
| Path | Purpose |
|------|---------|
| `include/neural_core/security/cryptographic_primitives_bundle.h` | Umbrella header for all security modules |
| `security/crypto/c/` | C crypto implementations (Kyber, Dilithium, SPHINCS+, AES-GCM, ChaCha20, etc.) |
| `security/crypto/asm/x86_64/` | MASM-optimized crypto (AES-NI, SHA-NI, AVX2, SSE2) |
| `security/ai/` | RLHF safety, differential privacy, prompt/output filtering |
| `security/network/` | DDoS mitigation, transport security, identity management |
| `security/memory/` | Memory hardening, leak detector |
| `security/monitor/` | Container breakout detection, runtime monitoring |
| `security/ui/` | Key vault |
| `security/updates/` | Container security (SBOM, CVE scanning), signed updates |
| `tests/security/` | Test files for all security modules |
| `kernel/` | Core computational substrate (tensor, autodiff, optimizer, trainer, thread pool) |
| `algorithms/` | ARC, SER, HSS, NPE, FM algorithm implementations |

## Build Targets
- `neural_core_kernel` — Core tensor/memory/trainer library
- `neural_architecture_layer` — Neural architecture algorithms
- `neural_security_c` — C security library
- `neural_security_cpp` — C++ obfuscation library
- `neural_cuda_kernels` — CUDA kernels (conditional, OFF by default)

## Coding Standards
- 4-space indentation, no tabs
- `SNEPPX_` prefix for all public functions and types
- `SNEPPX_` prefix for all macros and constants
- `void` in empty parameter lists: `int foo(void)` not `int foo()`
- Return `int` (0 success, -1 error) for most API functions
- `size_t` for lengths/counts, `uint8_t*` for byte buffers
- Document all public API functions with brief header comments
