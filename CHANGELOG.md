# Changelog

All notable changes to ARIX-Algo.

## [0.1.0] — 2026-06-24

### Added

- Core tensor operations: multi-dimensional arrays, float32/64/int32/64/bool, 50+ ops
- Secure memory allocator: aligned allocation, guard pages, canaries, secure wipe
- Thread pool: single-threaded fallback (parallel in v0.5)
- HSS: forward pass, sequential scan, first-order discretization
- SER: top-k routing, expert forward, load balance loss computation
- ARC: z-score input guard, gradient noise, output consistency check
- NPE: bytecode VM, attention/MLP compilers, static verifier
- FM: single-node memory banks, all-reduce sync stub
- Security S0: Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3, secure random, Argon2id
- Security S1: guard pages, canaries, ASLR, locked memory, constant-time ops
- Security S2: control flow flattening, string encryption stubs
- Security S3: behavioral monitor structure
- Build system: CMake, cross-platform scripts
- Tests: 50 unit tests, 4 integration tests
- Examples: 5 demos (HSS, SER, ARC, NPE, FM)
- Documentation: architecture overview, API reference stubs

### Known Limitations

- Autodiff is stub: backward pass does nothing
- No GPU acceleration: CUDA stubs only
- No distributed training: Rust stubs only
- No Python API: placeholder package only
- Security S2-S3: partial implementations

### Security

- All releases signed with Ed25519
- Checksums: SHA-256, SHA-512
- BDFL governance, commit signing practiced

### Build Requirements

```
CMake 3.16+ | C11/C++20 | Python 3.11+ (optional) | MSVC 2022 / GCC 11+ / Clang 14+
```
