# Changelog

All notable changes to SNEPPX-Algo.

## [0.9.5.748] — 2026-07-15

### Added

**Phase 1 — Foundation Layer**
- `c_loader.py`: Unified ctypes library loader with `load_library()` and `find_load()` — auto-detects C extension (`.pyd`/`.so`) with pure-Python fallback
- `c_types.py`: Shared ctypes declarations (int8_t..uint64_t, float_t, double_t, const_char_p, void_p, DataBuffer, make_struct)
- `dispatch/` subpackage: Factory `get_backend(device)` with CPU/CUDA dispatch + domain-specific backends (`get_security_backend`, `get_algo_backend`, `get_kernel_backend`, `get_infra_backend`)

**Phase 2 — Crypto Bindings (6 files, 85+ tests)**
- `crypto_sign.py`: Ed25519, Dilithium2/3/5, SPHINCS+ (SHAKE/SHA2) signatures
- `crypto_kem.py`: Kyber512/768/1024 KEM, X25519 ECDH
- `crypto_symmetric.py`: AES-GCM, ChaCha20-Poly1305, AEAD unified interface
- `crypto_hash.py`: SHA-256/512, SHA3-256/512, SHAKE128/256, BLAKE3, SipHash
- `crypto_kdf.py`: HMAC, HKDF, PBKDF2, Argon2id
- `crypto_util.py`: BigNum (Montgomery), DRBG (CTR/HMAC/Hash), SecureRandom, ConstantTime, EntropyPool

**Phase 3 — Security Bindings S1–S9 (10 files, 100+ tests)**
- `secure_memory.py`: SecureAllocator, StackCanary, ASLR, MemoryHardening, MemoryLeakDetector
- `s4_network.py`: DDoSMitigation, IdentityManager, TransportSecurity (TLS 1.3, Noise, QUIC)
- `runtime_monitor.py`: IntegrityMonitor, ContainerBreakoutDetector, AdvancedMonitor
- `key_vault.py`: KeyVault, AuditLogger
- `secure_updates.py`: SignedUpdate, ContainerSecurity
- `formal_verify.py`: LTLProperty, ModelChecker, StateGraph
- `pen_testing.py`: NetworkFuzzer, SelfAuditor
- `obfuscation_bridge.py`: ObfuscationConfig, CfgFlattening, InstructionSubstitution, OpaquePredicate, StringEncryption, ObfuscationPipeline
- `ai_safety_ext.py`: RLHFSafety, DataPoisoningDefense, PromptFilter, OutputVerifier
- `security_middleware.py`: AuthConfig, RateLimitConfig, PromptFilterConfig, OutputVerifierConfig, SecurityConfig, SecurityMiddleware, Authenticator, PromptFilter, OutputVerifier

**Phase 4 — Algorithm Bindings (5 files, 47+ tests)**
- `algo_arc.py`: ARCAttackSim, ARCFForward, ARCGradientObfuscator, ARCInputGuard, ARCOutputVerifier
- `algo_fm.py`: FMController, FMMemoryBank, FMNode, FMSync
- `algo_hss.py`: HSSDiscretize, HSSScan, HSSStep, HSSModel (as AlgoHSSModel)
- `algo_npe.py`: NPEInstruction, NPEProgram, NPECompiler, NPEVM, NPEVerify
- `algo_ser.py`: SERExpert, SERRoute, SERLayer, SERLoss, SERModel (as AlgoSERModel)

**Phase 5 — Kernel Bindings (7 files, 70+ tests)**
- `c_attention.py`: Attention, DifferentialAttention, FlexAttention
- `c_arch.py`: ArchOps, Mamba2 (GELU, SiLU, layer norm, RMSNorm, softmax)
- `c_memory.py`: MemoryAllocator, MemoryPool
- `c_thread.py`: ThreadPool, WorkStealingPool
- `c_tensor_expr.py`: TensorExpr, TensorExprCompiler
- `c_simd_gemm.py`: SimdGemm, GemmConfig
- `c_logger.py`: Logger

**Phase 6 — Infrastructure Bindings (4 files, 26 tests)**
- `net_bindings.py`: Topology, SocketComm, RDMA, GRPCService, NCCLComm, ProcessGroup
- `driver_bindings.py`: CUDADriver, CUDADeviceProps, ROCmDriver, TPUDriver

**Phase 7 — ASM Integration (5 files, 23 tests)**
- `asm_bridge.py`: CPUFeatures, AsmLibrary, AsmFunction, ConstantTime, SecureMemory
- `crypto_asm.py`: AESNI, SHANI, ChaCha20AVX2, Poly1305ASM, MontgomeryMul, Ed25519ASM, ConstantTimeOps, FirewallASM
- `security/crypto/asm/build.ps1`: MASM x64 build script

**CLI Entry Points (installed with `pip install sneppx-alg`)**
- `sneppx-train` → `SneppX_ALG.interface_bindings.train_cli:main`
- `sneppx-serve` → `SneppX_ALG.interface_bindings.serve_cli:main`
- `sneppx-experiment` → `SneppX_ALG.interface_bindings.experiment_cli:main`

**snepp-cli Dispatcher**
- Created `snepp-cli/index.js` — `snepp <command>` wrapper for train/serve/experiment

### Fixed

- `experiment_cli.py`: Moved `import time` from `if __name__ == "__main__"` guard to top-level (was causing `NameError` in `list`/`view`/`compare` commands)
- `setup.py`: Added `pyyaml>=5.1` to `install_requires` (required by `train_cli.py` and `serve_cli.py` for YAML config loading)
- `setup.py`: Added `uvicorn>=0.22.0` to `extras_require["serve"]` (required by `serve_cli.py`)

### Improved

- `interface_bindings/__init__.py`: Expanded from ~400 to ~1284 lines — all Phase 1-7 exports + complete `__all__` manifest
- `pyproject.toml`: Synced with `setup.py` — added `[project.scripts]` and `serve` extra
- All 31 test suites pass (pure-Python fallback paths verified)

## [0.9.4.467] — 2026-07-14

### Added

**4-Ring Firewall Architecture**
- **Transport Ring**: TLS 1.3/mTLS with cert pinning, ALPN, Noise NK/XX/IK
- **Network Ring**: CIDR allow/deny, rate limiting (token bucket), connection tracking, port knocking
- **Application Ring**: Injection filter (SQLi/XSS/command injection), path traversal detection, concurrent request limiter
- **Security Middleware Integration**: `set_security()` accepts firewall overrides, `check_firewall()` in request pipeline

**MASM x64 Hot-Path Routines** (`security/firewall/asm/`)
- `ip_match.asm`: CIDR IPv4/IPv6 matching with SIMD
- `rate_counter.asm`: Token bucket rate limiting
- `conn_track.asm`: Connection state tracking
- `port_knock.asm`: Port knock sequence verification

**YAML Configuration with Env/CLI Overrides**
- `security/firewall/firewall.yaml`: All rings configurable
- Environment variable overrides (`FIREWALL_*`)
- CLI flags in `serve_cli.py` (`--firewall-config`, `--firewall-enabled`, `--tls-enabled`, etc.)

**Firewall Tests** (24 new tests)
- `test_firewall_network.py`: CIDR, rate limit, port knock
- `test_firewall_application.py`: Injection filter, path traversal, concurrent limiter
- `test_firewall_transport.py`: TLS/mTLS, cert pinning

### Fixed
- Path traversal detection edge cases
- Lazy SSL context building (avoid startup cost)
- kwargs safety in firewall constructors

## [0.9.2.094] — 2026-07-10

### Added
- Complete refactor of `__init__.py` exports (753 insertions, 161 deletions)
- `benchmark.py`, `benchmarking.py`, `benchmarks.py` — unified benchmarking suite
- `distillation.py` — knowledge distillation losses (KD, AT, FM, CRD, multi-teacher)
- `onnx_export.py` — ONNX exporter/importer, TensorRT integration, opset upgrade
- `advanced_ops.py` — conv1d/2d, pooling, RNN/LSTM/GRU cells, MHA, transformer block
- `quantization.py` — INT8/INT4/FP8 (E4M3/E5M2), AWQ, GPTQ, QuantizedLinear

### Fixed
- `amp.py`: Autocast proper context manager, duplicate import removed
- `train.py`: `Trainer.train()` method added, optimizer saves LR
- `benchmarking.py`: Tensor import fixed, matmul shape fixed
- `SimpleTokenizer` shadowing resolved (`.data` over `.tokenizer`)

### Improved
- C source updates: `checkpoint_reader.c`, `dilithium.c`
- Test suite updates across `test_benchmarking.py`, `test_checkpoint.py`, `test_data.py`, `test_distributed.py`, `test_hf_integration.py`, `test_model.py`, `test_model_zoo.py`, `test_nn.py`, `test_optim.py`, `test_profiler.py`, `test_quantization.py`, `test_tensor.py`, `test_train.py`, `test_trainer_v2.py`, `test_ultra_trainer.py`

## [0.9.0] — 2026-06-30

### Added
- Core tensor operations: multi-dimensional arrays, float32/64/int32/64/bool, 50+ ops
- Secure memory allocator: aligned allocation, guard pages, canaries, secure wipe
- Thread pool: single-threaded fallback (parallel in v0.5)
- HSS: forward pass, sequential scan, first-order discretization
- SER: top-k routing, expert forward, load balance loss
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
- No distributed training: stubs only
- No Python API: placeholder package only
- Security S2–S3: partial implementations

### Security
- All releases signed with Ed25519
- Checksums: SHA-256, SHA-512
- BDFL governance, commit signing practiced

### Build Requirements
```
CMake 3.16+ | C11/C++20 | Python 3.11+ (optional) | MSVC 2022 / GCC 11+ / Clang 14+
```

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
- SNEPPX_sample_from_logits: rand() == RAND_MAX edge case
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
- No distributed training: stubs only
- No Python API: placeholder package only
- Security S2–S3: partial implementations

### Security
- All releases signed with Ed25519
- Checksums: SHA-256, SHA-512
- BDFL governance, commit signing practiced

### Build Requirements
```
CMake 3.16+ | C11/C++20 | Python 3.11+ (optional) | MSVC 2022 / GCC 11+ / Clang 14+
```

---

*Generated with automated tooling and manual curation.*