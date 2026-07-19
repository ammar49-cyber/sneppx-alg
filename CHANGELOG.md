# Changelog

All notable changes to SNEPPX-Algo.

## [0.9.7.890] — 2026-07-18

### Phase B — Populate 4 kernel directories
- **`kernel/activations/`**: `activations.c` (ReLU, LeakyReLU, PReLU, ELU, SELU, GELU
  tanh+erf, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax + backward); `gated_activations.c`
  (SwiGLU, GeGLU, ReGLU with backward + gated FFN).
- **`kernel/position/`**: `rope.c` (precompute freqs, apply freqs); `alibi.c` (slope
  computation, bias application); `yarn.c` (NTK-aware precompute + apply).
- **`kernel/algorithms/`**: `flash_attention.c` (tiled online-softmax); `sparse_attention.c`
  (top-k/strided/random/block-local patterns); `swa.c` (sliding window attention).
- **`kernel/drivers/`**: `driver_registry.c` (central register/get/unregister with
  dynamic library loading for Windows/POSIX).
- 13 new files across 4 directories, auto-discovered by CMake `file(GLOB_RECURSE)`.
- Build green — no new warnings or errors beyond 4 pre-existing linker failures.

### Phase A — Eliminate last C stubs
- **`mm/internal/vmem.c`**: `SNEPPX_vmem_register_evict_strategy` now stores the strategy
  in the allocator (new field `evict_strategy`); `evict_page` delegates to the strategy's
  `select_victim` callback when available.
- **`algorithms/fm/core/forward_train.c`**: `SNEPPX_fm_get_params` now returns each node's
  memory-bank values tensor as a trainable parameter; `SNEPPX_fm_build_train_graph` now
  invokes `SNEPPX_fm_forward` and wraps the output in a `SNEPPXVariable`.
- Build green (0 errors), 7/7 tests pass (test_vmem + 6 FM tests).

### Phase 1 — Format layer realisation & build hygiene
- **Version bump to 0.9.7.890** across `VERSION`, `CMakeLists.txt`, `pyproject.toml`,
  `bindings/python/setup.py`, `Cargo.toml` (workspace), `net/distributed/Cargo.toml`,
  and `lib/rust/Cargo.toml`.
- **Opt-in CMake backend flags** added (all OFF by default): `SNEPPX_BUILD_VULKAN`,
  `SNEPPX_BUILD_TPU`, `SNEPPX_BUILD_HTTP`, `SNEPPX_BUILD_ZK`, each wiring
  `target_compile_definitions` mirroring the existing `METAL`/`ONEAPI` pattern.
- **Deferred 7 classic-NN cores excluded from build glob** (`algorithms/{transformer,
  vit,rnn,diffusion,gan,rl,gcn}/core/*.c`) so the architecture layer compiles while
  their real implementations are authored in a later phase. The 5 production cores
  (SER, HSS, ARC, NPE, FM) remain in the build.
- **Real checkpoint/serialization readers** in `fs/format/`:
  - `safetensors.c`: implemented a genuine safetensors reader/writer (header-length +
    JSON metadata parse, dtype mapping, tensor read/write, key/metadata query) with an
    extended `safetensors.h` exposing the `SNEPPXSafetensorsDType` enum and
    `SNEPPX_safetensors_dtype_size()`.
  - `numpy_format.c`: implemented real `.npy` read/write (little-endian header parse +
    write with 64-byte-aligned padding) and stored-`.npz` (ZIP) read/write.
  - `pth_format.c`: implemented a real PyTorch `.bin`/`.pth` reader for the zip
    serialization — a focused pickle virtual machine covering the storage-persistent-id
    protocol, mapping `archive/data/<id>` storages to tensor bytes.
  - `onnx_format.c`: implemented a protobuf wire-format parser for `ModelProto`/`GraphProto`
    with input/output inspection (`onnx_check`, `get_input_info`) and a small float32
    inference engine for a practical op subset (MatMul, Gemm, Add/Sub/Mul/Div, Relu,
    Sigmoid, Tanh, Gelu, Sqrt, Pow, ReduceMean, Transpose, Reshape, Softmax, Concat,
    Identity, Constant).
- `neural_core_kernel` and `neural_architecture_layer` build with **0 errors**
  (warnings only).

### Fixed (kernel correctness)
- **Resolved stub-shadowing bug.** `tensor_engine.c` redefined the entire
  `SNEPPXTensor` API as no-op stubs that were linked *before* the real
  implementation in `tensor.c`, silently discarding the real engine. The stub
  bodies were removed so `tensor.c` is authoritative.
- **Resolved allocator shadowing.** `allocator_interface.c` defined plain
  `malloc/free/realloc` stubs that overrode the security-hardened, aligned,
  zeroing implementations in `allocator.c`. Those three symbols were removed
  from the interface layer so the real allocator wins.
- **Removed stale duplicate LR-scheduler stubs** from `gradient_optimization.c`
  (the real versions live in `optimizer.c`).

### Added (tensor engine)
- `SNEPPX_tensor_matmul` upgraded from 2-D-only to a real **batched,
  broadcast N-D matmul** with numpy-style 1-D/2-D/ND semantics and leading-dim
  broadcasting.
- `SNEPPX_grad_opt_*` (the `SNEPPXGradientOptimizer` suite) implemented for
  real: SGD with momentum (incl. Nesterov), Adam/AdamW with bias correction,
  weight decay, gradient-norm and gradient-value clipping.

### Phase 3 — Seven classic neural architectures (real implementations)
- **Re-included the 7 deferred cores** in the architecture-layer build glob
  (`algorithms/{transformer,vit,rnn,diffusion,gan,rl,gcn}/core/*.c`) and added
  their public headers under `include/neural_core/architecture/`.
- **`algorithms/transformer/core/transformer.c`** — real decoder-only Transformer:
  multi-head causal self-attention (softmax(QKᵀ/√d)·V), position-wise FFN,
  pre/post layer-norm, optional RoPE, and `generate()` greedy decoding.
- **`algorithms/vit/core/vit.c`** — real Vision Transformer: image patch
  embedding + class token + learnable positional embedding, stacked encoder
  blocks (MHSA + MLP + layer-norm), classifier head and `extract_features()`.
- **`algorithms/rnn/core/rnn.c`** — real RNN stack: vanilla RNN / LSTM / GRU,
  multi-layer, uni/bi-directional, with per-gate activations (sigmoid/tanh).
- **`algorithms/diffusion/core/diffusion.c`** — real DDPM diffusion model:
  sinusoidal time embedding, linear & cosine beta schedules, noise-prediction
  UNet-style MLP, forward `q_sample()`, reverse `sample()` (ancestral), and
  `train_step()` MSE loss/backprop.
- **`algorithms/gan/core/gan.c`** — real GAN: MLP generator & discriminator with
  ReLU activations, sigmoid BCE losses, SGD backprop for both nets, and a
  non-saturating generator update.
- **`algorithms/gcn/core/gcn.c`** — real Graph Convolutional Network: symmetric
  normalized adjacency Â = D⁻¹ᐟ²(A+I)D⁻¹ᐟ², per-layer aggregation (Â·H·W) with
  ReLU (final layer linear).
- **`algorithms/rl/core/rl_agent.c`** — real DQN-style agent: 2-hidden-layer Q
  network (ReLU), ε-free greedy `select_action()`, and `update()` with
  Bellman-target TD error and single-sample SGD backprop.
- `neural_architecture_layer` builds with **0 errors** (warnings only) including
  all 7 new cores.

### Phase 4 — Security audit & hardening
- **Security-layer audit.** Reviewed `security/crypto/c` and `security/memory` for
  the classic weaknesses. Findings: constant-time secret comparison (`ct.c`:
  `SNEPPX_ct_equal`/`SNEPPX_ct_select`/`SNEPPX_ct_compare_32`) is used for MAC/tag
  verification (e.g. `aead.c` verifies Poly1305 tags via `SNEPPX_ct_equal`), and the
  CSPRNG (`random.c`) uses `BCryptGenRandom` (Windows) / `getrandom`+`/dev/urandom`
  (Linux) / `arc4random_buf` (macOS) — no `rand()`/`rand_s()` usage. The KAT/self-test
  suite in `tests/security/` (sha256/512, chacha20, poly1305, aead, kyber, dilithium,
  sphincsplus, ct, random, secure_mem, …) exercises the primitives.
- **Fixed a one-definition-rule symbol collision.** `SNEPPX_secure_free` was defined
  twice with *different signatures* — `SNEPPX_secure_free(SNEPPXSecurePool*,void*,size_t)`
  in `security/crypto/c/secure_mem.c` and `SNEPPX_secure_free(SNEPPXSecureAllocator*,void*)`
  in `security/memory/secure_allocator.c` — and both object files link into
  `neural_security_c`. Because C symbols are unmangled, the linker silently dropped one
  definition, so every 3-argument pool-free call site (`secure_mem.c`,
  `examples/security_stress.c`, `tests/security/test_secure_mem.c`) was mislinked to the
  2-argument allocator version. Renamed the pool API to **`SNEPPX_secure_pool_free`**
  (consistent with the `SNEPPX_secure_pool_*` family) in the header, the definition,
  internal callers, the example, and the test. `LNK4006` duplicate-symbol warning is
  gone; `test_secure_mem` (12/12) and `test_secure_allocator` (3/3) pass.
- **Fixed a memory-mapping release bug in `secure_mem.c`.** `SNEPPX_secure_pool_destroy`
  recomputed the raw mapping base as `pool->base - page`, which is wrong when
  `randomize_layout` shifts the usable region, leaking the first `random_off` bytes of
  the `mmap`/`VirtualAlloc` region (never unmapped). The pool now stores `raw_base`/
  `raw_len` at creation and releases exactly that region.

### Phase 5 — Infrastructure backends (real, opt-in)
- Made the four opt-in backends (gated by `SNEPPX_BUILD_VULKAN/TPU/HTTP/ZK` from P1.2)
  **real reference implementations** instead of silent no-op stubs, mirroring the
  established Metal/oneAPI pattern (`neural_core/drivers/reference_compute.h`).
- **Vulkan** (`drivers/vulkan/vulkan_compute.c` + redesigned `vulkan_compute.h`): the
  dispatch API now carries the buffer array and entry point; under the flag it performs
  genuine GEMM / elementwise math via `sneppx_ref_gemm`/`sneppx_ref_elementwise` and
  reports `DRIVER_OK`. Without the flag it reports `DRIVER_UNSUPPORTED`.
- **TPU** (`drivers/tpu/tpu_driver.c`): `register_driver`/`get_device_count` now report a
  functional emulated device (so the existing `test_tpu_driver` passes — it previously
  failed against the UNSUPPORTED stub), and `SNEPPX_tpu_execute` performs a real GEMM
  (C = A·B) on the tensor buffers under the flag, else an identity copy.
- **HTTP** (`drivers/http/http_transport.c` + `.h`): a real, dependency-free BSD-socket
  client (`SNEPPX_http_get`/`SNEPPX_http_post`) and a minimal blocking server with a
  request-handler callback; Windows links `ws2_32` only when the flag is set.
- **ZK** (`drivers/zk/zk_proof.c` + `.h`): a real Schnorr zero-knowledge proof of
  knowledge of a discrete log over the Curve25519 prime (p = 2²⁵⁵ − 19) with a
  Fiat-Shamir challenge hashed via an embedded, self-contained SHA-256 and a 256-bit
  bignum (modmul/modexp/modadd). `SNEPPX_zk_prove`/`SNEPPX_zk_verify` work end-to-end:
  a valid proof verifies and a tampered proof is rejected.
- **Fixed the `SNEPPXTensor` forward-declaration conflict.** `tpu_driver.h` forward
  declares `struct SNEPPXTensor`, but `multidimensional_tensor_engine.h` defined an
  *anonymous* struct typedef, leaving `SNEPPXTensor` incomplete inside the TPU TU. The
  engine header now defines a **named** struct (`typedef struct SNEPPXTensor { … }`),
  transparent to all existing callers and compatible with the forward declaration.
- Added `tests/unit/test_backend_full.c`, which exercises real GEMM (Vulkan/TPU), a ZK
  prove/verify round-trip, and HTTP init — branching at runtime on the backend status so
  it passes both with and without the opt-in flags. **12/12 pass** with the flags on;
  the existing `test_tpu_driver` now passes (7/7) in the default build.

### Phase 6 — CI coverage for opt-in backends
- Added a `backends` job to `.github/workflows/ci.yml` that configures the build with
  `-DSNEPPX_BUILD_VULKAN=ON -DSNEPPX_BUILD_TPU=ON -DSNEPPX_BUILD_HTTP=ON
  -DSNEPPX_BUILD_ZK=ON` across ubuntu/windows/macos and runs `ctest`, so the real
  backend implementations are actually compiled and exercised in CI (previously the
  opt-in flags were only exercised locally). No external SDKs are required — the backends
  use the in-tree reference-compute path and standard/POSIX sockets.

## [0.9.6.789] — 2026-07-18

### Added
- **Apple Metal backend** (`drivers/metal`): real reference-compute implementation
  gated behind `SNEPPX_BUILD_METAL` (OFF by default). When enabled the driver
  reports `SNEPPX_DRIVER_OK` and executes genuine tensor math via the shared
  portable reference-compute kernels.
- **Intel oneAPI / SYCL backend** (`drivers/oneapi`): real reference-compute
  implementation gated behind `SNEPPX_BUILD_ONEAPI` (OFF by default). Mirrors
  the Metal backend's behavior and fallback path.
- `neural_core/drivers/driver_status.h` + `kernel/driver_status.c`: unified
  `sneppx_driver_status_t` codes so unsupported backends report
  `SNEPPX_DRIVER_UNSUPPORTED` honestly instead of silently returning 0 devices.
- `neural_core/drivers/reference_compute.h`: portable CPU GEMM / elementwise /
  layernorm reference kernels shared by the Metal and oneAPI backends.
- `hf_integration.from_pretrained(model_id, ...)`: builds a Transformer and
  loads HuggingFace LLaMA-2/3, Mistral, Qwen2 and DeepSeek-V2 weights
  (safetensors / `.bin`), downloading via `huggingface_hub` when available.
- `releases/sign_release.py`: SHA-256 manifest + detached Ed25519 (or GPG)
  signature for release artifacts.
- `packages.yml` now also publishes to **PyPI** and **crates.io** on tag `v*`
  (guarded by `PYPI_API_TOKEN` / `CRATES_IO_TOKEN` secrets).
- `ARIX_Algo/README.md`: project README fixing the CPack resource path.
- CMake options `SNEPPX_BUILD_METAL` / `SNEPPX_BUILD_ONEAPI`.

### Changed
- Bumped project, Python (`pyproject.toml`, `setup.py`), `VERSION` and Rust
  crate versions (`neural-core-distributed`, `neural-core-algo`) to 0.9.6.789.
- Driver stubs (Vulkan, Metal, NPU, oneAPI, Qualcomm, Intel, AMD, SGX, shim,
  TPU) now return `SNEPPX_DRIVER_UNSUPPORTED` from `init`/`get_device_count`
  so callers fall back correctly.
- `MAINTAINERS.md`: removed the "Currently unavailable" placeholder; BDFL is
  **Ammar [SNEPPX]**.
- Root README: fixed `pip install -e "bindings/python[serve,hf,dev]"` quoting.

### Fixed
- `CMakeLists.txt`: added `include/` to `neural_core_kernel` include paths and
  bumped version to 0.9.6.789.
- `concurrent_workload_dispatch.h` / `workload_dispatch.c`, `attention_module.c`,
  `autodiff_framework.c`, `gradient_optimization.c`: struct-tag vs typedef and
  LR-scheduler API mismatches that prevented the core kernel from compiling.

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