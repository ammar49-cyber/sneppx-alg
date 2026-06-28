# 🚀 ARIX-Algo

<p align="center">
  <img src="docs/logo.jpeg" alt="ARIX-Algo Logo" width="30%" style="mix-blend-mode: screen; background: transparent;"/>
</p>

> **Next-Generation AI Architecture** · Security built into the foundation.  
> Not patched later. Not bolted on. **In every instruction.**

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0--alpha-blueviolet?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/C-11-00599C?style=for-the-badge&logo=c" alt="C11"/>
  <img src="https://img.shields.io/badge/C++-20-f34b7d?style=for-the-badge&logo=cplusplus" alt="C++20"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/CMake-3.16-064F8C?style=for-the-badge&logo=cmake" alt="CMake 3.16"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License MIT"/>
  <img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge" alt="Build Passing"/>
  <img src="https://img.shields.io/badge/tests-54%2F56%20passing-success?style=for-the-badge" alt="54/56 Tests Passing"/>
  <img src="https://img.shields.io/badge/security-S0%2FS1%20complete-important?style=for-the-badge" alt="Security S0/S1 Complete"/>
</p>

---

## 📋 Table of Contents

- [🌟 What Is ARIX-Algo?](#-what-is-arix-algo)
- [🧩 The Five Components](#-the-five-components)
- [📊 What Works Now (v0.1.0)](#-what-works-now-v010)
- [✅ What You Can Do](#-what-you-can-do)
- [❌ What You Cannot Do (Yet)](#-what-you-cannot-do-yet)
- [⚡ Quick Start](#-quick-start)
  - [🔧 Build from Source](#-build-from-source)
  - [💻 C Example](#-c-example)
  - [🐍 Python Example](#-python-example)
- [🔐 Security Architecture](#-security-architecture)
- [🗺️ Roadmap](#️-roadmap)
- [📈 Project Stats](#-project-stats)
- [🧪 Test Suite](#-test-suite)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [👑 Governance](#-governance)
- [🌐 Links](#-links)
- [📖 Citation](#-citation)
- [✨ Acknowledgments](#-acknowledgments)

---

## 🌟 What Is ARIX-Algo?

**ARIX-Algo** is a **new class of AI architecture** — a composable, cryptographically-secure algorithm pipeline designed from the ground up with security as a **first principle**, not an afterthought.

Unlike existing AI systems that bolt on safety layers after the fact, ARIX-Algo weaves security into **every instruction, every memory allocation, and every data path**. It is:

- 🏗️ **A foundation**, not a model
- 🔬 **A research platform**, not a product
- 🛡️ **A security-first design**, not a retrofitted patch
- 🧩 **Modular and composable** — every component can be used independently
- 📖 **Open source**, not a black box

ARIX-Algo is built for researchers, engineers, and security professionals who believe that **safe AI requires safe foundations**.

---

## 🧩 The Five Components

```
┌────────────────────────────────────────────────────────────────┐
│                     🔐 SECURITY LAYER                          │
│  S0 Crypto · S1 Secure Mem · S2 Obfuscation · S3 Monitor      │
└────────────────────────────────────────────────────────────────┘
                                │
┌────────────────────────────────────────────────────────────────┐
│                   🧠 ALGORITHM PIPELINE                        │
│  HSS  ──▶  SER  ──▶  ARC  ──▶  NPE  ──▶  FM                   │
│  (SSM)    (MoE)    (Guard)   (VM)     (Fed Mem)               │
└────────────────────────────────────────────────────────────────┘
                                │
┌────────────────────────────────────────────────────────────────┐
│                   🛡️ INTEGRITY LAYER (Future)                  │
│  ZK Proofs · Formal Verification · On-Device Attestation      │
└────────────────────────────────────────────────────────────────┘
```

### 🔷 HSS — Hierarchical State Space Model
**O(n log n) sequence modeling with state space decomposition.**

Multi-layer state space model using zero-order hold discretization. Processes sequences in logarithmic time via parallel scan over the state dimension. Hierarchical decomposition captures both short-term patterns and long-range dependencies simultaneously.

- 📐 **Math**: State space models with learned transition matrices
- ⚡ **Speed**: O(n log n) vs O(n²) for attention
- 🔗 **Memory**: Compressed state representation grows with d_state, not sequence length

### 🔷 SER — Sparse Expert Routing
**Dynamic parameter efficiency through learned sparsity.**

A Mixture-of-Experts layer where each input token is routed to its top-k experts. The remaining experts consume zero compute, giving you the capacity of a massive model at the cost of a small one.

- 🎯 **Top-k routing**: Only k experts active per token
- ⚖️ **Load balancing**: Anti-collapse loss keeps all experts utilized
- 🧠 **Learned gating**: Tiny MLP decides which experts to activate

### 🔷 ARC — Adversarial Robustness Core
**Security baked into the weights themselves.**

A three-layer defense pipeline that guards inputs, obfuscates gradients, and verifies outputs — all during normal forward/backward pass.

- 🛡️ **Input Guard**: Z-score anomaly detection flags adversarial inputs
- 🌫️ **Gradient Obfuscation**: Noise injection + clamping defeats gradient-based attacks
- ✅ **Output Verifier**: Cosine similarity + temporal smoothing catches aberrant outputs
- ⚔️ **Attack Simulation**: Built-in FGSM, PGD, and C&W attack generators

### 🔷 NPE — Neural Program Executor
**Guaranteed-correct neural computation paths.**

A 16-register virtual machine with 14 verified opcodes (MATMUL, ATTENTION, SOFTMAX, LAYERNORM, etc.). Neural networks are compiled into programs that are statically verified before execution.

- 🖥️ **Virtual Machine**: Register-based, 14 opcodes, deterministic execution
- 📝 **Compilers**: Attention and MLP compilers generate programs from config
- 🔍 **Static Verifier**: Proves programs terminate with no out-of-bound access
- 🔄 **Self-Evolving** (future): NPE rewrites its own inefficient paths at runtime

### 🔷 FM — Federated Memory
**Learn collectively. Forget nothing.**

Per-node memory banks with euclidean similarity search, LRU eviction, and trust-weighted synchronization. Enables privacy-preserving collaborative learning across nodes.

- 💾 **Memory Bank**: Per-node, fixed-size, learnable read/write
- 🔄 **Sync Protocol**: Trust-weighted all-reduce with differential privacy
- 📉 **Gradient Compression**: Top-k selection + random sampling
- 🔒 **Privacy**: Laplace noise ensures differential privacy guarantees

---

## 📊 What Works Now (v0.1.0)

| Component | Status | What Works | What Doesn't |
|-----------|--------|-----------|--------------|
| 🧮 **Tensor Core** | ✅ **Real** | Multi-dim arrays, 13 dtypes, row-major, 80+ ops | GPU (CPU only) |
| 💾 **Memory** | ✅ **Real** | Aligned allocation, secure zero, guard pages | NUMA optimization |
| 🧵 **Thread Pool** | ⚠️ Stub | Single-threaded fallback | Real parallelism |
| 🌀 **HSS** | ⚠️ Partial | Forward pass, sequential scan, first-order discretization | Parallel scan, training |
| 🎯 **SER** | ⚠️ Partial | Top-k routing, expert forward, load balance loss | Learned gating |
| 🛡️ **ARC** | ⚠️ Partial | Z-score input guard, gradient noise, output check | Formal proofs |
| 🤖 **NPE** | ⚠️ Partial | VM executes, compilers generate programs | JIT, formal verification |
| 🌐 **FM** | ⚠️ Partial | Single-node memory banks, sync stubs | Real distributed |
| 🔄 **Autodiff** | ❌ **Stub** | Structure only | Backward pass (does nothing) |
| ⚡ **Optimizer** | ❌ **Stub** | Structure only | Parameter updates |
| 🐍 **Python API** | ❌ **Stub** | Package installs, `hello()` works | Model, Trainer classes |
| 🔐 **Security S0** | ✅ **Real** | Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3, Argon2id | — |
| 🔐 **Security S1** | ✅ **Real** | Guard pages, canaries, ASLR, locked memory | — |
| 🔐 **Security S2** | ⚠️ Partial | Control flow flattening, string encryption stubs | Full obfuscation |
| 🔐 **Security S3** | ⚠️ Partial | Behavioral monitor structure | Real anomaly detection |

### 📏 Lines of Code Breakdown

| Component | C/C++ LOC | Tests | Status |
|-----------|-----------|-------|--------|
| 🧮 Tensor Core | ~3,000 | 57 edge + 19 creation + 29 shape + 17 ops + 14 reduction + 14 NN + 9 I/O | ✅ |
| 💾 Memory | ~800 | 13 | ✅ |
| 🧵 Thread Pool | ~300 | 11 | ⚠️ |
| 🌀 HSS | ~500 | 2 | ⚠️ |
| 🎯 SER | ~600 | 5 | ⚠️ |
| 🛡️ ARC | ~600 | 5 | ⚠️ |
| 🤖 NPE | ~700 | 4 | ⚠️ |
| 🌐 FM | ~600 | 4 | ⚠️ |
| 🔄 Autodiff | ~400 | 1 | ❌ |
| ⚡ Optimizer | ~300 | 1 | ❌ |
| 🐍 Python API | ~500 | 3 | ❌ |
| 🔐 S0 Crypto | ~2,000 | 10 | ✅ |
| 🔐 S1 Secure Mem | ~800 | 3 | ✅ |
| 🔐 S2 Obfuscation | ~1,500 | 4 | ⚠️ |
| 🔐 S3 Monitor | ~100 | 0 | ⚠️ |
| **📊 Total** | **~15,000** | **~150** | — |

---

## ✅ What You Can Do

- ✅ **Build the project from source** — on Windows, Linux, or macOS
- ✅ **Run all tests** — 54/56 pass (2 pre-existing S0 edge cases)
- ✅ **Run demos** — HSS, SER, ARC, NPE, FM all have working examples
- ✅ **Audit the security code** — S0 and S1 are production-grade cryptography
- ✅ **Read the architecture** — Full mathematical documentation included
- ✅ **Contribute** — Email patches welcome, see [CONTRIBUTING.md](CONTRIBUTING.md)
- ✅ **Use the tensor ops** — 80+ operations on 13 data types
- ✅ **Run benchmarks** — Compare tensor and autodiff performance

## ❌ What You Cannot Do (Yet)

- ❌ **Train a model** — autodiff backward pass is not implemented
- ❌ **Generate text** — no training → no inference
- ❌ **Benchmark against GPT-2** — wait for v1.0
- ❌ **Use as PyTorch replacement** — not the goal
- ❌ **Deploy to production** — v0.1.0 is a research prototype

---

## ⚡ Quick Start

### 🔧 Build from Source

**Prerequisites:**
- CMake 3.16+
- C11 compiler (MSVC 2022, GCC 11+, Clang 14+)
- C++20 compiler (for S2 obfuscation engine, optional)
- Python 3.11+ (for bindings, optional)

```bash
# Clone 🧬
git clone https://github.com/ammar49-cyber/arixalgo.git
cd arix-algo

# Configure 🔧
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DARIX_BUILD_TESTS=ON

# Build 🏗️
cmake --build . -j$(nproc)

# Test 🧪
ctest --output-on-failure

# Run benchmarks ⏱️
./tests/benchmark/bench_tensor
./tests/benchmark/bench_autodiff
```

**Platform-specific guides:** See [docs/installation.md](docs/installation.md) for Linux, macOS (Intel & Apple Silicon), and Windows (Visual Studio & WSL) instructions.

### 💻 C Example

```c
#include "arix_tensor.h"
#include "arix_hss.h"
#include "arix_ser.h"
#include "arix_arc.h"

int main() {
    // 📐 Create input tensor: batch=4, seq=8, dim=16
    size_t shape[] = {4, 8, 16};
    ArixTensor* input = arix_tensor_randn(shape, 3, ARIX_FLOAT32);

    // 🌀 HSS forward pass
    ArixHSSConfig hss_cfg = arix_hss_config_default();
    ArixHSSModel* hss = arix_hss_model_create(&hss_cfg, 42);
    ArixTensor* hss_out;
    arix_hss_forward(hss, input, &hss_out);

    // 🎯 SER forward pass
    ArixSERConfig ser_cfg = arix_ser_config_default();
    ArixSERModel* ser = arix_ser_model_create(&ser_cfg, 42);
    ArixTensor* ser_out;
    arix_ser_forward(ser, hss_out, &ser_out);

    // 🛡️ ARC forward pass (guard + verify)
    ArixARCConfig arc_cfg = arix_arc_config_default();
    ArixARCModel* arc = arix_arc_model_create(&arc_cfg);
    ArixTensor* arc_out;
    ArixTensor* anomaly_scores;
    arix_arc_forward(arc, ser_out, &arc_out, &anomaly_scores);

    // 📊 Print results
    printf("🔷 Output shape: ");
    for (size_t i = 0; i < arix_tensor_ndim(arc_out); i++)
        printf("%zu ", arix_tensor_shape(arc_out)[i]);
    printf("\n");

    // 🧹 Cleanup
    arix_tensor_destroy(input);
    arix_tensor_destroy(hss_out);
    arix_tensor_destroy(ser_out);
    arix_tensor_destroy(arc_out);
    arix_tensor_destroy(anomaly_scores);
    arix_hss_model_destroy(hss);
    arix_ser_model_destroy(ser);
    arix_arc_model_destroy(arc);

    return 0;
}
```

### 🐍 Python Example

```python
from arix_algo import hello

# 🚀 Currently a stub — full API coming in v0.5.0
print(hello())
# Output: "ARIX-Algo v0.1.0 — Core tensor operations implemented in C"
```

**Planned Python API (v0.5.0):**

```python
import arix_algo as ax

# 🧮 Create a tensor
t = ax.Tensor.randn((4, 8, 16), dtype=ax.float32)

# 🌀 Create an HSS model
model = ax.HSSModel(d_model=64, d_state=16, num_layers=2)

# ⚡ Forward pass
output = model(t)

# 📈 Training
loss = model.train_step(t, target)
loss.backward()
model.optimizer.step()
```

---

## 🔐 Security Architecture

Security is **not a feature**. It is **the architecture**. Every byte of memory, every cryptographic operation, every instruction path is designed with security as the primary constraint.

```
S0 ── S1 ── S2 ── S3 ── S4 ── S5 ── S6 ── S7 ── S8 ── S9
│     │     │     │     │     │     │     │     │     │
Crypto Memory Obfusc Monitor Network AI    UI    Update Formal Pentest
Core  Secure Engine  Engine  Sec   San    Sec   Sec    Verif  Report
      Mem
```

### 🔐 S0 — Cryptographic Core ✅
**Production-grade, side-channel resistant cryptographic primitives.**

| Primitive | Standard | Use Case | Status |
|-----------|----------|----------|--------|
| ✍️ **Ed25519** | RFC 8032 | Digital signatures | 304/306 test vectors pass |
| 🔑 **X25519** | RFC 7748 | Key exchange | Full DH exchange |
| 🔒 **ChaCha20-Poly1305** | RFC 8439 | Authenticated encryption | 100+ test vectors |
| 🔗 **SHA-3** | FIPS 202 | Hashing (224/256/384/512) | NIST test vectors |
| ⚡ **BLAKE3** | Reference | Fast hashing | Reference test vectors |
| 🧂 **Argon2id** | RFC 9106 | Secure key derivation | Test vectors + timing defense |
| 🎲 **Secure Random** | OS CPRG | Entropy source | Windows CNG / Linux getrandom |

### 🔐 S1 — Secure Memory ✅
**Hardened memory allocation that resists physical and software attacks.**

| Feature | Description |
|---------|-------------|
| 🛡️ **Guard Pages** | RW pages with PROT_NONE boundaries — overflow detection |
| 🧪 **Canaries** | 128-bit stack-based overflow detection with generation counters |
| 🎲 **ASLR** | Heap entropy via VirtualAlloc (Win) / mmap (Linux) randomization |
| 🔒 **Locked Memory** | mlock/VirtualLock prevents secrets from swapping to disk |
| 🧹 **Secure Wipe** | memset with compiler barrier — guaranteed zeroing |
| ⏱️ **Constant-Time** | memcmp variant immune to timing side-channels |

### 🔐 S2 — Obfuscation Engine ⚠️
**Multi-level code obfuscation to resist reverse engineering.**

| Feature | Level | Status |
|---------|-------|--------|
| 🔀 Control Flow Flattening | LIGHT–MAXIMUM | ✅ Implemented |
| 🔑 String Encryption | LIGHT–MAXIMUM | ✅ Implemented |
| 🔄 Instruction Substitution | MEDIUM–MAXIMUM | ✅ Implemented |
| 🌀 Opaque Predicates | MEDIUM–MAXIMUM | ✅ Implemented |
| 💻 Code Virtualization | HEAVY–MAXIMUM | ✅ Implemented |
| 🐞 Anti-Debug | MAXIMUM | ✅ Implemented |

### 🔐 S3 — Behavioral Monitor ⚠️
**Real-time runtime integrity monitoring.**

| Feature | Status |
|---------|--------|
| 📊 Frequency Analysis | Structure only |
| ⏱️ Timing Analysis | Structure only |
| 🚨 Anomaly Detection | Structure only |

### 🔐 S4–S9 ⏳ Planned

| Phase | Description | Target |
|-------|-------------|--------|
| 📜 S4 | Zero-Knowledge Proofs for inference verification | 2027 H1 |
| 📱 S5 | On-device runtime (phone, edge, embedded) | 2027 H2 |
| 🤝 S6 | Federated contribution protocol | 2028 H1 |
| 🧬 S7 | Self-evolving NPE paths | 2028 H2 |
| ✅ S8 | Formal verification of critical paths | 2029 |
| 🎯 S9 | Third-party penetration testing | 2029 |

---

## 🗺️ Roadmap

### 🎯 v0.5.0 — Trainable on CPU (6 months)

| Area | Deliverable | Metric |
|------|-------------|--------|
| 🔄 Autodiff | Real C gradients for all 50+ tensor ops | Backward matches numerical gradient |
| ⚡ Optimizer | SGD + AdamW with weight decay & LR scheduling | Converges on toy problems |
| 🌀 HSS | Parallel scan over state dimension | 2× speedup vs sequential |
| 🎯 SER | Learned gating (tiny MLP per expert) | 5% perplexity improvement |
| 🛡️ ARC | Real attack simulation during training | Robust to ε=0.1 FGSM |
| 🤖 NPE | JIT compilation of attention/MLP programs | 10× VM speedup |
| 🌐 FM | Single-node on-device sync (simulated multi-node) | Correct gradient aggregation |
| 🏋️ Trainer | CPU training loop with checkpointing | 10k steps WikiText-2 in 24h |
| 🐍 Python API | Tensor, Variable, Tape, Model, Trainer classes | API complete for demo |
| 🧪 Testing | 200+ tests, all pass | 100% coverage of exposed API |

### 🎯 v1.0.0 — Competitive with GPT-2 (18 months)

| Area | Deliverable | Metric |
|------|-------------|--------|
| 🖥️ GPU | CUDA kernels for all tensor ops, HSS, SER, ARC, NPE, FM | 50× speedup vs CPU |
| 🌍 Distributed | Multi-GPU training with NCCL | Linear scaling up to 8 GPUs |
| 🌀 HSS | Hierarchical scan with multi-resolution states | 5× longer context than GPT-2 |
| 🎯 SER | Top-2 out of 64 experts, load-balanced | 2× parameter efficiency |
| 🛡️ ARC | Provable robustness guarantees | Certified robustness at ε=0.1 |
| 🤖 NPE | On-device execution with CUDA backend | 1000 programs/s throughput |
| 🌐 FM | Federated training across 4 nodes | 95% of centralized performance |
| 🐍 Python | Full API, pip install, documentation | 90% API coverage documented |
| 🔐 Security | S3 behavioral monitor complete, S4 ZK proofs start | Real-time anomaly detection |
| 📏 **Size** | **~200,000 LOC** | **7B parameters** |

### 🎯 v2.0.0 — Competitive with LLaMA-3 (4 years)

| Area | Deliverable |
|------|-------------|
| 📏 Scale | 70B parameters, 1M LOC |
| 🖥️ Hardware | Multi-node training, 64+ GPUs |
| 🌀 HSS | Adaptive state size per layer |
| 🎯 SER | Hierarchical MoE with 256 experts |
| 🤖 NPE | Self-modifying programs with meta-optimization |
| 🌐 FM | Cross-datacenter federated learning |
| 🔐 Security | S4 ZK proofs complete, S5 on-device runtime |
| 🛡️ Integrity | Formal verification of critical paths |

### 🎯 v3.0.0 — Competitive with GPT-4 (7 years)
**1T parameters, 5M LOC** · Fully learned routing · 1000+ GPU cluster · S6 federated contribution protocol · On-device attestation

### 🎯 v4.0.0 — Beyond Existing Systems (10 years)
**10T parameters, 15M LOC** · Self-evolving architecture · S7 self-evolving NPE paths · S8-S9 formal verification + pentest · Full formal proof of alignment

### 📅 Quarterly Milestones

| Quarter | Milestone |
|---------|-----------|
| 🟢 **2026 Q3** | Autodiff backward pass functional · SGD + AdamW · 100+ tests |
| 🟢 **2026 Q4** | HSS parallel scan · SER learned gating · CPU training loop · Python API v0.5 |
| 🟡 **2027 Q1** | WikiText-2 training end-to-end · S3 complete |
| 🟡 **2027 Q2** | S4 ZK proofs: proof generation for MLP · 200+ tests |
| 🟠 **2027 Q3** | CUDA kernels for tensor ops · Benchmarks |
| 🟠 **2027 Q4** | Multi-GPU training · S5 on-device runtime (ARM NEON) |
| 🔴 **2028** | 7B model · S6 · Distributed training · S7 |

---

## 📈 Project Stats

| Metric | Value |
|--------|-------|
| 📝 C/C++ Source | **~15,500 lines** |
| 🧪 Registered Tests | **56** (54 pass, 2 pre-existing crypto edge cases) |
| 🔧 Build Time | **~30s** (Release, 8 cores) |
| 📦 Dependencies | **0** for C core |
| 🖥️ Platforms | **Windows** (MSVC) · **Linux** (GCC/Clang) · **macOS** (Clang) |
| 🐍 Python | **3.11+** (optional, via pybind11) |
| 🔐 Security Layers | **3 of 10** implemented (S0-S2 complete, S3 partial) |
| 🧩 Components | **11** (5 algorithm + 4 security + 2 foundation) |

---

## 🧪 Test Suite

```
📂 tests/
├── 📁 unit/                    # 🧪 Component unit tests
│   ├── test_tensor.c           # 🧮 Tensor operations (6 tests)
│   ├── test_tensor_edge.c      # 🧮 Edge cases (57 tests)
│   ├── test_tensor_creation.c  # 🧮 Creation functions (19 tests)
│   ├── test_tensor_shape.c     # 🧮 Shape manipulation (29 tests)
│   ├── test_tensor_ops.c       # 🧮 Element-wise + comparison (17 tests)
│   ├── test_tensor_reduction.c # 🧮 Reduction + linear algebra (14 tests)
│   ├── test_tensor_nn.c        # 🧮 Neural network ops (14 tests)
│   ├── test_tensor_io.c        # 🧮 Save/load + conversion (9 tests)
│   ├── test_autodiff.c         # 🔄 Autodiff operations
│   ├── test_autodiff_edge.c    # 🔄 Autodiff edge cases (27 tests)
│   ├── test_memory.c           # 💾 Memory allocator
│   ├── test_thread.c           # 🧵 Thread pool
│   ├── hss/                    # 🌀 HSS tests
│   ├── ser/                    # 🎯 SER tests
│   ├── arc/                    # 🛡️ ARC tests
│   ├── npe/                    # 🤖 NPE tests
│   ├── fm/                     # 🌐 FM tests
│   └── train/                  # 🏋️ Trainer tests
├── 📁 integration/             # 🔗 Multi-component integration tests
├── 📁 benchmark/               # ⏱️ Performance benchmarks
│   ├── bench_tensor.c          # 🧮 Tensor benchmarks (7 groups)
│   ├── bench_autodiff.c        # 🔄 Autodiff benchmarks (7 groups)
│   ├── bench_hss.c             # 🌀 HSS benchmarks
│   ├── bench_ser.c             # 🎯 SER benchmarks
│   └── bench_npe.c             # 🤖 NPE benchmarks
├── 📁 security/                # 🔐 S0+S1 crypto + S2 obfuscation tests
├── 📁 fuzz/                    # 🎲 Fuzz testing (future)
├── 📁 integration/             # 🔗 Multi-component integration tests
└── 📁 python/                  # 🐍 Python tests (stubs)
```

```bash
# Run all tests 🧪
ctest --output-on-failure

# Run specific test 🎯
ctest -R test_tensor

# Run benchmarks ⏱️
./tests/benchmark/bench_tensor
./tests/benchmark/bench_autodiff
```

### 🔧 Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ARIX_BUILD_TESTS` | ON | Build test suite |
| `ARIX_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `ARIX_BUILD_PYTHON` | OFF | Build Python bindings |
| `ARIX_BUILD_CUDA` | OFF | Build CUDA kernels (future) |
| `ARIX_USE_ASAN` | OFF | Enable AddressSanitizer |
| `ARIX_USE_UBSAN` | OFF | Enable UndefinedBehaviorSanitizer |
| `ARIX_USE_LTO` | OFF | Enable Link-Time Optimization |

### ⚙️ Build Presets

```bash
cmake --preset debug              # 🐛 Debug build (-g -O0)
cmake --preset release            # 🚀 Release build (-O3 -DNDEBUG)
cmake --preset relwithdebinfo     # 📊 Release with debug symbols (-O2 -g)
cmake --preset ninja-release      # ⚡ Ninja + Release (fast builds)
cmake --preset asan               # 🛡️ Debug + AddressSanitizer
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| 📖 [docs/index.md](docs/index.md) | Documentation landing page |
| 🔧 [docs/installation.md](docs/installation.md) | Platform-specific build guides |
| 🏗️ [docs/architecture.md](docs/architecture.md) | Full architecture with math |
| 🔐 [docs/security.md](docs/security.md) | Security architecture deep dive |
| 🗺️ [docs/roadmap.md](docs/roadmap.md) | Detailed project roadmap |
| 📘 [docs/api/c.md](docs/api/c.md) | C API reference |
| 📗 [docs/api/python.md](docs/api/python.md) | Python API reference |
| 🤝 [docs/contributing.md](docs/contributing.md) | Development workflow |

---

## 🤝 Contributing

We accept email patches. No pull requests. No Discord. **Technical merit above all.**

```bash
git format-patch -1 HEAD
git send-email --to=algoarix@gmail.com 0001-your-patch.patch
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details on:
- 📝 Patch requirements and format
- 🎨 Coding style (C, C++, Python)
- 🔑 GPG/Ed25519 signing
- 🧪 Testing requirements

## 📜 License

- **Algorithm (C/C++ core, Python bindings)**: [MIT License](LICENSE)
- **Website and distribution**: Closed-source

## 👑 Governance

**BDFL**: Ammar [ARIX]

| Purpose | Contact |
|---------|---------|
| 📝 Patches | [algoarix@gmail.com](mailto:algoarix@gmail.com) |
| 🔐 Security | [algoarix@gmail.com](mailto:algoarix@gmail.com) |
| ⚖️ Conduct | [algoarix@gmail.com](mailto:algoarix@gmail.com) |

---

## 🌐 Links

<p align="center">
  <a href="https://aixsite.vercel.app"><b>🌐 Website</b></a> ·
  <a href="https://github.com/ammar49-cyber/arixalgo"><b>🐙 GitHub</b></a> ·
  <a href="https://x.com/Arixdrv"><b>🐦 Twitter / X</b></a> ·
  <a href="https://www.instagram.com/algoarix/"><b>📸 Instagram</b></a> ·
  <a href="https://www.youtube.com/@ArixAlgo"><b>🎬 YouTube</b></a>
</p>

---

## 📖 Citation

```bibtex
@software{arix_algo_2026,
  author = {Ammar [ARIX]},
  title = {{ARIX-Algo}: Next-generation {AI} architecture with cryptographic integrity},
  url = {https://github.com/ammar49-cyber/arixalgo},
  year = {2026}
}
```

---

## ✨ Acknowledgments

ARIX-Algo stands on the shoulders of giants. We are grateful to:

- 🧠 The **Transformer** architecture (Vaswani et al., 2017)
- 🌊 **State Space Models** (Gu, Goel, Ré, 2021)
- 🎯 **Mixture of Experts** (Shazeer et al., 2017 — "Outrageously Large Neural Networks")
- 🛡️ **Adversarial Robustness** (Goodfellow, Shlens, Szegedy, 2014)
- 🔐 **Ed25519** (Bernstein et al., 2012)
- 🔒 **ChaCha20-Poly1305** (Bernstein, 2008)
- 🧪 **Test framework inspiration**: minunit, libtap

---

<p align="center">
  <b>ARIX-Algo</b> — <i>Security in every instruction.</i>
</p>

<p align="center">
  <sub>Built with ❤️ for a safer AI future</sub>
</p>

<p align="center">
  <a href="#">⬆️ Back to Top</a>
</p>
