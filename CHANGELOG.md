# Changelog

All notable changes to ARIX-Algo.

## [0.1.1] — 2026-06-30

### Added

- Multi-head attention module with RoPE, causal mask, batched matmul 3D
- Inference engine: autoregressive generation with top-k/top-p/temperature sampling
- Data pipeline: TextDataset with BPE tokenization and batching
- Infrastructure: 20+ config files (CI, linting, workflows, docs)

### Changed

- Model architecture: modular pipeline with per-module enable/disable flags
- Tensor rename: comprehensive project-wide nomenclature restructuring
- arch.c: attention + embedding/lm_head wired into forward/get_params/create/destroy
- README fully updated for v0.1.1 features

### Fixed

- arix_sample_from_logits: rand() == RAND_MAX edge case
- data_pipeline.c: buffer overflow in offsets allocation, line_by_line tokenization
- test_inference.c: duplicate tests_failed macro conflict with test_common.h

### Tests

- 6 new tests: attention forward, RoPE, causal mask, KV-cache, batched matmul
- 6 new tests: data pipeline (dataset create, batch) + inference (argmax, sample, generate)
- 64 registered tests (62 pass, 2 pre-existing crypto edge cases)

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
