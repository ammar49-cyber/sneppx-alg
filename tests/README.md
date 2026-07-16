# Tests — Quality Assurance

The `tests/` directory contains **31+ test suites** with **300+ tests** covering C, Python,
CUDA, security, integration, fuzzing, chaos engineering, and benchmarks.

## Test Suites

### Python Tests (`tests/python/`)
Run with no C compiler needed (pure-Python fallback):

```
test_tensor.py       — Tensor creation, ops, reductions, NN, loss
test_nn.py           — Neural network layers (Linear, Embedding, Norm, Attention)
test_optim.py        — Optimizers (AdamW, SGD, Lion, LAMB, etc.)
test_data.py         — Data loading (Dataset, DataLoader)
test_distributed.py  — Distributed primitives (all_reduce, barrier)
test_hf_integration.py — HuggingFace model loading
test_train.py        — Training loop
test_quantization.py — INT8/FP8/AWQ/GPTQ (17 tests)
test_checkpoint.py   — Checkpoint writer/reader, heartbeat, elastic training (23 tests)
test_profiler.py     — Profiler, Timer, MemoryTracker (13 tests)
test_model_zoo.py    — Model configs, from_pretrained() (49 tests)
test_crypto_sign.py  — Digital signature tests
test_secure_memory.py — Memory hardening tests
test_algo_hss.py     — HSS model tests
test_c_attention.py  — Attention kernel tests
test_net_bindings.py — Networking bindings tests
test_asm_bridge.py   — ASM bridge tests
```

Run all:
```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/ -v
```

### C Tests (`tests/unit/`, `tests/quantization/`, `tests/security/`)
Require CMake build. Run via ctest:
```powershell
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

### Security Tests (`tests/security/`)
Tests for all S0-S9 security modules:
```
test_kyber          — Kyber KEM (CCA-secure)
test_dilithium       — Dilithium signatures
test_sphincsplus     — SPHINCS+ signatures
test_aes_gcm         — AES-GCM encrypt/decrypt
test_chacha20        — ChaCha20 stream cipher
test_ed25519         — Ed25519 signatures
test_secure_mem      — Secure memory allocation/zeroing
test_aslr            — ASLR functionality
test_canary          — Stack canary detection
test_lock            — Memory locking
test_power           — Power analysis resistance
test_obf_pipeline    — Obfuscation pipeline
```

## Excluded Tests

The following C tests are excluded from the default build (`DISABLED` in CMakeLists.txt)
pending implementation of their source files:

| Test | Reason |
|------|--------|
| `test_activation` | C-level activation tests need source |
| `test_arch` | Advanced arch tests need source |
| `test_aslr` | ASLR C test needs implementation |
| `test_canary` | Canary C test needs implementation |
| `test_data_pipeline` | Data pipeline C test needs implementation |
| `test_hss_discretize` | HSS discretize C test needs implementation |
| `test_hss_hierarchical` | HSS hierarchical C test needs implementation |
| `test_inference_engine` | Inference engine C test needs implementation |
| `test_lock` | Memory lock C test needs implementation |
| `test_obf_pipeline` | Obfuscation pipeline C test needs implementation |
| `test_power` | Power analysis C test needs implementation |
| `test_rocm_driver` | ROCm driver C test needs ROCm SDK |
| `test_ser_loss` | SER loss C test needs implementation |

### Additional Tests

| Suite | Location | Description |
|-------|----------|-------------|
| Fuzzing | `tests/fuzz/` | Fuzz testing for security modules |
| Chaos | `tests/chaos/` | Chaos engineering experiments |
| Integration | `tests/integration/` | End-to-end integration tests |
| Benchmark | `tests/benchmark/` | Performance benchmarks |
| Compliance | `tests/compliance/` | Regulatory compliance checks |
| Threat Intel | `tests/threat_intel/` | Threat intelligence tests |
| Incident Response | `tests/incident_response/` | Incident response tests |

## CUDA Tests

`tests/cuda_test_suite.cu` — Device properties, GEMM vs cuBLAS, element-wise,
layernorm, softmax, AdamW, memory pool, RNG, dropout, gradient clipping.
Requires `-DSNEPPX_BUILD_CUDA=ON`.
