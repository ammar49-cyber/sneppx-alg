# Roadmap

## Completed

### S0 — Crypto Core
- SHA-3 (FIPS 202), ChaCha20, Poly1305, AEAD, Ed25519, BLAKE3, Argon2
- Constant-time utilities, cryptographic random
- All primitives are side-channel resistant

### S1 — Secure Memory
- Guard-page protected allocations with mlock/VirtualLock
- 128-bit canary system with generation counters
- ASLR for heap allocations
- Side-channel resistant operations (select, equal, lt, is_zero, cond_copy)
- Cache flushing, prefetch, timing-safe equal, power analysis dummy ops

### S2 — Obfuscation Engine
- Control-flow flattening (switch-dispatch state machine)
- String encryption (compile-time XOR, runtime ChaCha20-derived)
- Instruction substitution (LEA for ADD, NAND for AND/OR/XOR)
- Opaque predicates (math invariants, pointer self-comparison)
- Code virtualization (stack-based VM with 256 registers, encrypted handler table)
- Anti-debug (ptrace, IsDebuggerPresent, RDTSC timing, INT3 scan, CPUID hypervisor)
- Configurable levels: LIGHT, MEDIUM, HEAVY, MAXIMUM

## In Progress

### S3 — Behavioral Monitor
Runtime integrity monitoring, anomaly detection, hook detection, behavioral profiling.

## Planned

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

## Timeline (High-Level)

```
Now ───────────────────────────────────────────────────────────────►
 │                                                                   │
 S3────────────►                                                     │
    S4─────────────────────►                                         │
       S5──────────────────────────►                                 │
          S6───────────────────────────────►                         │
             S7──────────────────────────────────────────►           │
                                                                     │
 2026 H2    2027 H1    2027 H2    2028 H1    2028 H2    2029        │
```

## Current Stats

| Metric | Value |
|--------|-------|
| Tests passing | 40/42 |
| C/C++ source lines | ~3,000 |
| Components | 11 (5 algo + 4 security + 2 core) |
| Security layers | 3 of 7 complete |

## Contributing

Each phase has its own `/docs/{PHASE}.md` with detailed specs. Pick an open issue, read the phase doc, and submit a PR. All contributions are cryptographically signed and verified (starting S6).
