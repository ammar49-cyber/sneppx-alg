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

## Phase 1 Completed (GPU/CUDA Backend)
Core CUDA backend (~9,600 new lines across kernel/cuda/, algorithms/*/cuda/, net/distributed/, tests/):

### kernel/cuda/ — Core CUDA Library (~6,900 lines)
- **common.cuh** (371L): Architecture detection (Hopper/Ampere/Volta/Pascal), tile config, FP16/BF16 conversions, MMA/WGMMA wrappers, cp.async helpers, warp shuffle reductions, Philox RNG, CUBLAS TLS handle management
- **tensor_cuda.h/.cu** (1,238L): Tensor-core GEMM (128x128 block tiling, warp-level MMA), element-wise ops, reduction, layernorm, softmax, CUBLAS fallback, fused bias+activation GEMM
- **attention_cuda.h/.cu** (1,579L): Flash Attention v2 with online softmax + tiling; Flash Attention v3 TMA+WGMMA stub; GQA, paged attention, KV cache management (alloc/free/update), RoPE (forward/inplace/cache precompute), causal/sliding window/block-sparse masks
- **autodiff_cuda.h/.cu** (1,237L): GEMM backward (via cuBLAS), activation backward (ReLU/GELU/SiLU/Tanh/Sigmoid), layernorm/RMSNorm backward, softmax/CE backward, element-wise backward, dropout backward, MSE/BCE backward, convolution backward, gradient clipping/accumulation/scale
- **optim_cuda.h/.cu** (1,033L): Fused AdamW, SGD+momentum, Lion, LAMB, LARS, AdaFactor optimizers; ZeRO-1 partitioned AdamW; in-kernel LR scheduling (cosine/linear/warmup); overflow checking; generic optimizer lifecycle
- **memory_cuda.h/.cu** (713L): Memory pool (pre-allocated blocks), stream pool, event pool, pinned/managed memory, async 2D/batched memcpy, device properties query, kernel auto-block size tuning
- **rng_cuda.h/.cu** (676L): Philox-based uniform/normal/truncated-normal/bernoulli/integer RNG; Xavier/Kaiming initialization; fused dropout forward; batch RNG; permutation (Fisher-Yates)

### algorithms/*/cuda/ — Algorithm CUDA Extensions (~1,700 lines)
- **hss/**: Selective scan (Mamba/S6), S4 forward, HiPPO matrix init, SSM convolution, hierarchical softmax, sparse-dense matmul, top-k, parallel prefix scan (Blelloch)
- **ser/**: Top-k gating (softmax + selection), MoE dispatch/combine, load balancing loss, expert all-to-all, fused MoE forward
- **fm/**: Ring/butterfly all-reduce, gradient quantization (FP16→INT8), Top-K sparsification, federated averaging, memory bank sync
- **npe/**: Neural VM instruction dispatch kernel, differentiable program execution
- **arc/**: PGD/FGSM adversarial attacks, gradient obfuscation, randomized smoothing

### net/distributed/ — NCCL Layer (~530 lines)
- nccl.h/.c: Dynamic NCCL loading (win/linux/mac), all-reduce/all-gather/reduce-scatter/broadcast/send/recv, CPU fallback, process group management

### tests/ — CUDA Test Suite
- cuda_test_suite.cu (487L): Device properties, GEMM vs cuBLAS, element-wise, layernorm, softmax, AdamW, memory pool, RNG, dropout, gradient clipping

## Phase 3 Completed (Advanced Architectures)
Advanced architecture implementations across `kernel/attention/`, `kernel/arch/`, `kernel/activations/`, `kernel/position/`:

- **advanced_arch.h** (140L): Unified header for DifferentialAttn, MLA, FlexAttn, Mamba2, MoD, gated activations, YaRN NTK-RoPE, ALiBi
- **differential_attention.c** (140L): Differential Attention (λ-scaled subtracted QK pairs), Multi-head Latent Attention (DeepSeek MLA with absorbed KV projection + GQA fallback)
- **flex_attention.c** (245L): FlexAttention with block-sparse kernel + mask modulation function, multi-modal cross-attention (text→vision KV), Mixture of Depth (token-level expert routing), SwiGLU/GeGLU/ReGLU gated activations, YaRN NTK-aware RoPE scaling with ramp interpolation, ALiBi position encoding
- **mamba2.c** (116L): Mamba-2 selective SSM with HiPPO A_log initialization, 1D convolution, discretized selective scan with SiLU output gate

## Phase 2 Completed (Distributed Training)
Distributed training infrastructure across `kernel/distributed/`:

- **distributed.h** (171L): Unified config for DP/TP/PP/EP sizes, ZeRO stage, NCCL dtype, pipeline microbatches, expert capacity, master addr/port
- **zero.c** (160L): ZeRO-1/2/3 optimizer state partitioning with AdamW fused kernel, partition-aware start/size helpers
- **pipeline.c** (119L): 1F1B schedule with microbatches, non-blocking inter-stage send/recv
- **tensor_parallel.c** (97L): Row/column split linear with partial GEMM + all-reduce
- **expert_parallel.c** (99L): Expert-to-rank mapping, all-to-all dispatch, FM distributed helpers
- **ddp.c** (94L): Bucket-based gradient all-reduce with compute overlap, event synchronization
- **gradient_comm.c** (210L): Hierarchical all-reduce (intra-node NVLink + inter-node RDMA), Top-K gradient compression with error feedback, CUDA kernel implementation
- **checkpoint.c** (264L): Distributed checkpoint coordinator, async save with stream, coordinated barrier save, fault tolerance with heartbeat/health check, elastic training with max restarts, communication profiler (CUDA events, bandwidth measurement)
- **distributed_sampler.c** (136L): Sharded data loading with epoch-based Fisher-Yates shuffle, gradient accumulation manager (fused accumulation + normalization)

## Phase 5 Completed (Quantization & Compression)
Quantization kernels and Python API across C, CUDA, and Python:

- **include/neural_core/kernel/quantization.h**: `SNEPPXQuantMode` (INT8_SYM/ASYM, INT4_SYM, FP8_E4M3/E5M2, AWQ, GPTQ), granularity modes
- **kernel/quantization/quantize_int8.c**: INT8 sym/asym, per-channel, INT4 packed quant/dequant
- **kernel/quantization/quantize_fp8.c**: FP8 E4M3/E5M2 bit-level float↔uint8 encode/decode with special value handling
- **kernel/quantization/awq.c**: AWQ optimal scale grid search, weight scaling, quantize
- **kernel/quantization/gptq.c**: GPTQ Hessian computation, Cholesky inverse, column-by-column quant
- **kernel/quantization/quantize_cuda.cu**: CUDA INT8/FP8 kernels (shared mem reduction, Hopper native FP8)
- **bindings/python/.../quantization.py**: Pure NumPy QuantMode, quant/dequant, AWQ, GPTQ, QuantizedLinear, error metrics
- **tests/python/test_quantization.py**: 17 tests (all pass)
- **tests/quantization/test_quantize.c**: C host-side test suite

## Phase 6 Completed (Async Checkpointing & Fault Tolerance)
Checkpoint format, heartbeat, elastic training, and fault tolerance:

- **include/neural_core/kernel/checkpoint.h**: `SNEPPX_CheckpointCoord` with async state, `SNEPPX_FaultTolerance`
- **include/neural_core/kernel/heartbeat.h**: UDP heartbeat messages, peer status tracking, timeout detection
- **include/neural_core/kernel/elastic.h**: Elastic training with join/leave/failure/reconfigure lifecycle
- **kernel/distributed/checkpoint.c**: Double-buffered async save (D2D→D2H overlap), background I/O thread, checkpoint_reader format integration
- **kernel/distributed/heartbeat.c**: UDP-based heartbeat send/recv, non-blocking I/O, SUSPECT→DEAD state machine
- **kernel/distributed/elastic.c**: Node join/leave, failure handling, topology reconfiguration, versioned checkpoints
- **fs/format/checkpoint_reader.c** (real I/O now): Binary format with header, tensor records, metadata
- **kernel/distributed/checkpoint.c**: `SNEPPX_FaultTolerance` wrapping heartbeat + elastic for comprehensive fault management
- **bindings/python/.../checkpoint.py**: `CheckpointWriter/Reader`, `CheckpointCoordinator`, `HeartbeatMonitor`, `ElasticTrainer`, `FaultToleranceManager`
- **tests/python/test_checkpoint.py**: 23 tests (all pass)

## Phase 7 Completed (Profiling & Debugging)
Profiling infrastructure, NVTX markers, structured logging, and sanitizer CI:

- **include/neural_core/kernel/profiler.h**: `SNEPPX_Profiler` with named entry aggregation, `SNEPPX_KernelTimer`, `SNEPPX_PROFILE_KERNEL` macro, range push/pop
- **kernel/profiler.c**: Profiler implementation with JSON export, range stack, kernel timing
- **include/neural_core/kernel/logger.h**: `SNEPPX_Logger` with 6 severity levels, JSON/color stdout, per-rank
- **kernel/logger.c**: Logger implementation with ANSI colors, JSON file append, timestamps
- **kernel/cuda/tensor_cuda.cu**: NVTX stubs replaced — real `SNEPPX_USE_NVTX` path + software fallback
- **bindings/python/.../profiler.py**: `Profiler`, `Timer` (context/start-stop), `@timeit` decorator, `MemoryTracker`, `TrainProfiler`
- **scripts/run_sanitizers.sh / .ps1**: ASan/UBSan builds + compute-sanitizer (memcheck/racecheck/initcheck) + ctest
- **tests/python/test_profiler.py**: 13 tests (all pass)

## Phase 8 Completed (Model Zoo)
Model configs, weight converters, and `from_pretrained()` for LLaMA 2/3, Mistral, Qwen 2, DeepSeek V2:

- **include/neural_core/kernel/model_zoo.h**: `SNEPPXModelFamily` enum, LLM config structs (LLaMA/Mistral/Qwen2/DeepSeekV2), JSON serialize/parse, weight name mapping helpers
- **kernel/model_zoo/model_zoo.c**: Presets for llama2 7B/13B/70B, llama3 8B/70B, mistral 7B, qwen2 7B/72B, deepseek_v2 lite/full; `SNEPPX_llm_config_from_name`, `SNEPPX_llm_config_to_json`, `SNEPPX_llm_config_from_json`
- **config/model_zoo/*.json**: Reference config files for each model size
- **bindings/python/.../model_zoo.py**: Config dataclasses, `get_model_config`, `from_pretrained`, `convert_hf_to_sneppx`, `build_model_from_config`, `build_transformer_from_config`, `HF_WEIGHT_MAP`, safetensors reader
- **tests/python/test_model_zoo.py**: 49 tests (all pass)

## Python Test Commands
```powershell
$env:PYTHONPATH = "bindings/python"
python tests/python/test_tensor.py
python tests/python/test_nn.py
python tests/python/test_optim.py
python tests/python/test_data.py
python tests/python/test_distributed.py
python tests/python/test_hf_integration.py
python tests/python/test_train.py
python tests/python/test_quantization.py
python tests/python/test_checkpoint.py
python tests/python/test_profiler.py
python tests/python/test_model_zoo.py
```

## Build Targets
- `neural_core_kernel` — Core tensor/memory/trainer library
- `neural_architecture_layer` — Neural architecture algorithms
- `neural_security_c` — C security library
- `neural_security_cpp` — C++ obfuscation library
- `neural_cuda_kernels` — CUDA kernels (conditional, OFF by default, set -DSNEPPX_BUILD_CUDA=ON)

## Coding Standards
- 4-space indentation, no tabs
- `SNEPPX_` prefix for all public functions and types
- `SNEPPX_` prefix for all macros and constants
- `void` in empty parameter lists: `int foo(void)` not `int foo()`
- Return `int` (0 success, -1 error) for most API functions
- `size_t` for lengths/counts, `uint8_t*` for byte buffers
- Document all public API functions with brief header comments
