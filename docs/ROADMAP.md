# Roadmap

## Version Milestones

### v0.1.0 — Research Prototype (Current)

**Release**: 2026-06-24

**Algorithm Pipeline**:
- HSS: forward pass, sequential scan, first-order discretization
- SER: softmax top-k routing, load balancing loss
- ARC: z-score input guard, gradient obfuscation, output verifier
- NPE: 16-register VM, 14 opcodes, attention/MLP compilers, static verifier
- FM: per-node memory banks, euclidean similarity, LRU eviction, trust-weighted sync

**Foundation**:
- Tensor: 5 dtypes, 50+ ops, row-major layout
- Memory: aligned alloc, guard pages, canaries, secure wipe
- Thread pool: single-threaded fallback
- Autodiff: stub (structure only)
- Optimizer: stub (SGD structure only)

**Security**:
- S0: Ed25519, X25519, ChaCha20-Poly1305, SHA-3, BLAKE3, Argon2id, secure random
- S1: Guard pages, canaries, ASLR, locked memory, secure wipe, constant-time ops
- S2: Control-flow flattening, string encryption, instruction substitution, opaque predicates, code virtualization, anti-debug (LIGHT–MAXIMUM levels)
- S3: Behavioral monitor structure (frequency, timing, anomaly tracking)

**Testing**: 49 CTest tests, 47 pass (2 pre-existing S0 edge cases). Benchmarks for tensor and autodiff.

**Limitations**: Autodiff backward does nothing. No GPU. No distributed training. No Python API.

**Size**: ~15,000 C/C++ LOC

---

### v0.5.0 — Trainable on CPU (6 months)

**Target**: End-to-end trainable on CPU for WikiText-2 perplexity evaluation.

| Area | Deliverable | Metric |
|------|-------------|--------|
| Autodiff | Real C gradients for all 50+ tensor ops | Backward pass matches numerical gradient |
| Optimizer | SGD + AdamW with weight decay, LR scheduling | Converges on toy problems |
| HSS | Parallel scan over state dimension | 2× speedup vs sequential |
| SER | Learned gating (tiny MLP per expert) | 5% perplexity improvement |
| ARC | Real attack simulation during training | Robust to ε=0.1 FGSM |
| NPE | JIT compilation of attention/MLP programs | 10× VM speedup |
| FM | Single-node on-device sync (simulated multi-node) | Correct gradient aggregation |
| Trainer | CPU training loop with checkpointing | 10k steps WikiText-2 in 24h |
| Python API | Tensor, Variable, Tape, Model, Trainer classes | API complete for demo |
| Testing | 200+ tests, all pass | 100% coverage of exposed API |
| Benchmarks | Throughput (tokens/s), convergence curves | Measured vs PyTorch reference |

---

### v1.0.0 — Competitive with GPT-2 (18 months)

**Target**: 7B parameter model, CUDA acceleration, competitive with GPT-2 1.5B on standard benchmarks.

| Area | Deliverable | Metric |
|------|-------------|--------|
| GPU | CUDA kernels for all tensor ops, HSS, SER, ARC, NPE, FM | 50× speedup vs CPU |
| Distributed | Multi-GPU training with NCCL | Linear scaling up to 8 GPUs |
| HSS | Hierarchical scan with multi-resolution states | 5× longer context than GPT-2 |
| SER | Top-2 out of 64 experts, load-balanced | 2× parameter efficiency |
| ARC | Provable robustness guarantees | Certified robustness at ε=0.1 |
| NPE | On-device execution with CUDA backend | 1000 programs/s throughput |
| FM | Federated training across 4 nodes | 95% of centralized performance |
| Python API | Full API, pip install, documentation | 90% API coverage documented |
| Security | S3 behavioral monitor complete, S4 ZK proofs start | Real-time anomaly detection |

**Size**: ~200,000 LOC

**Benchmark target**: WikiText-2 perplexity ≤ GPT-2 small; LAMBADA accuracy ≥ GPT-2 small.

---

### v2.0.0 — Competitive with LLaMA-3 (4 years)

**Target**: 70B parameter model, 1M LOC, competitive with LLaMA-3 8B.

| Area | Deliverable |
|------|-------------|
| Scale | 70B parameters, 1M LOC |
| Hardware | Multi-node training, 64+ GPUs |
| HSS | Adaptive state size per layer |
| SER | Hierarchical MoE with 256 experts |
| NPE | Self-modifying programs with meta-optimization |
| FM | Cross-datacenter federated learning |
| Security | S4 ZK proofs complete, S5 on-device runtime |
| Integrity | Formal verification of critical paths |

**Size**: ~1,000,000 LOC

---

### v3.0.0 — Competitive with GPT-4 (7 years)

**Target**: 1T parameter model, 5M LOC, competitive with GPT-4.

| Area | Deliverable |
|------|-------------|
| Scale | 1T parameters, 5M LOC |
| Hardware | 1000+ GPU cluster |
| Architecture | Fully learned routing, no fixed experts |
| Security | S6 federated contribution protocol |
| Integrity | On-device attestation for all nodes |

---

### v4.0.0 — Beyond Existing Systems (10 years)

**Target**: 10T parameter model, 15M LOC, beyond existing architectures.

| Area | Deliverable |
|------|-------------|
| Scale | 10T parameters, 15M LOC |
| Architecture | Self-evolving, meta-learned architecture |
| Security | S7 self-evolving NPE paths, S8-S9 formal verification and pentest |
| Integrity | Full formal proof of alignment |

---

## Security Roadmap

| Phase | Status | Description | Target |
|-------|--------|-------------|--------|
| S0 | ✅ Complete | Cryptographic primitives | 2026 Q2 |
| S1 | ✅ Complete | Secure memory | 2026 Q2 |
| S2 | ✅ Complete | Obfuscation engine | 2026 Q2 |
| S3 | ⚠️ In Progress | Behavioral monitor | 2026 Q4 |
| S4 | ⏳ Planned | ZK proofs for verifiable inference | 2027 H1 |
| S5 | ⏳ Planned | On-device runtime (phone/edge) | 2027 H2 |
| S6 | ⏳ Planned | Federated contribution protocol | 2028 H1 |
| S7 | ⏳ Planned | Self-evolving NPE paths | 2028 H2 |
| S8 | ⏳ Planned | Formal verification | 2029 |
| S9 | ⏳ Planned | Penetration testing | 2029 |

### S4 — Verifiable Inference (ZK Proofs)
*Differentiator: no other open-source AI project has this.*

- Zero-knowledge proofs for neural network inference
- Prove correctness without revealing weights
- Public verifier code
- Integration with NPE VM for per-instruction proofs

### S5 — On-Device Runtime
*Target: phones, edge devices, embedded systems.*

- Quantization-aware execution paths (int8, fp16)
- Sparse compute integration with SER (only route to necessary experts)
- Memory-efficient secure allocator profiling
- ARM NEON / WebAssembly backend for NPE VM
- No cloud dependency

### S6 — Federated Contribution Protocol
*Target: open-source community growth engine.*

- Ed25519-signed gradient contributions
- Trust-weighted all-reduce aggregation (leveraging FM)
- Reputation scoring based on contribution verification
- Public dashboard of contributors and their verified contributions
- Incentive model documentation

### S7 — Self-Evolving NPE Paths
*Target: truly next-gen algorithmic capability.*

- NPE rewrites its own inefficient execution paths at runtime
- Meta-optimization without retraining
- Self-healing: detect degraded paths and replace them
- Formal verification of new paths before deployment

## Intermediate Milestones

### 2026 H2

| Quarter | Milestone |
|---------|-----------|
| Q3 | Autodiff backward pass functional (C, all tensor ops) |
| Q3 | SGD + AdamW optimizers with weight decay |
| Q3 | 100+ tests, all pass |
| Q4 | HSS parallel scan implemented |
| Q4 | SER learned gating (MLP-based) |
| Q4 | CPU training loop working, trainer test passes |
| Q4 | Python API v0.5 complete (Tensor, Variable, Tape, Model, Trainer) |

### 2027 H1

| Quarter | Milestone |
|---------|-----------|
| Q1 | WikiText-2 training pipeline end-to-end |
| Q1 | S3 behavioral monitor complete (frequency, timing, anomaly detection) |
| Q2 | S4 ZK proofs: proof generation for MLP inference |
| Q2 | 200+ tests, 100% coverage of exposed API |

### 2027 H2

| Quarter | Milestone |
|---------|-----------|
| Q3 | CUDA kernels for tensor ops (matmul, element-wise, reductions) |
| Q3 | Benchmarks: throughput (tokens/s), convergence curves |
| Q4 | Multi-GPU training with NCCL |
| Q4 | S5 on-device runtime: ARM NEON backend for tensor ops |

### 2028

| Quarter | Milestone |
|---------|-----------|
| Q1 | 7B model training on 8 GPUs |
| Q2 | S6 federated contribution protocol |
| Q3 | Distributed training across nodes |
| Q4 | S7 self-evolving NPE paths |

### 2029+

| Year | Milestone |
|------|-----------|
| 2029 | S8 formal verification, S9 penetration testing |
| 2030 | 70B model, 1M LOC |
| 2032 | 1T model, 5M LOC |
| 2035 | 10T model, 15M LOC |

## Timeline (High-Level)

```
Now ───────────────────────────────────────────────────────────────►
 │                                                                   │
 v0.1.0──v0.5.0────────────────v1.0.0────────────────v2.0.0────────►
  │       │                    │                     │
  S0-S2    S3───►              S4──────►             S5──────►
           S0-S2                       S6────────────►
                                               S7──────────────────►
                                                                     │
  2026     2026 H2            2027 H2           2028 H2   2030+
```

## Current Stats

| Metric | Value |
|--------|-------|
| Tests passing | 47/49 (2 pre-existing S0 edge cases) |
| C/C++ source lines | ~15,000 |
| Components | 11 (5 algo + 4 security + 2 core) |
| Security layers | 3 of 10 implemented (S0-S2 complete, S3 partial) |
| Build time (Release, 8 cores) | ~30s |
| Dependencies | 0 for C core |
| Platform support | Windows (MSVC), Linux (GCC/Clang), macOS (Clang) |

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) and [docs/contributing.md](contributing.md). All contributions are cryptographically signed and verified.

Pick a milestone from v0.5.0 or S3-S7 above. Submit patches via email to patches@arix.dev.
