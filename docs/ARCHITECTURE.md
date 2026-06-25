# Architecture

ARIX-Algo is a neuro-symbolic AI system with cryptographic integrity — a composable 5-component algorithm pipeline wrapped in 4 security layers.

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│         Python Bindings · CLI · REST API · Demos                │
├─────────────────────────────────────────────────────────────────┤
│                    Algorithm Pipeline                            │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐              │
│  │ HSS  │→│ SER  │→│ ARC  │→│ NPE  │→│  FM  │              │
│  │ SSM  │  │ MoE  │  │Guard │  │ VM   │  │FedMem│              │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘              │
├─────────────────────────────────────────────────────────────────┤
│               Integrity Layer (Future)                           │
│    ZK Proofs · Formal Safety · On-Device Runtime                 │
├─────────────────────────────────────────────────────────────────┤
│                    Security Layer                                │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐            │
│  │ S0     │  │ S1     │  │ S2       │  │ S3 (WIP)│            │
│  │ Crypto │  │Secure  │  │Obfuscate │  │Behavior │            │
│  │ Core   │  │Memory  │  │Engine    │  │Monitor  │            │
│  └────────┘  └────────┘  └──────────┘  └──────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                    Foundation Layer                              │
│        Tensor · Memory · Thread · Autodiff · Optimizer          │
└─────────────────────────────────────────────────────────────────┘
```

## Algorithm Pipeline

```
Input ──► [HSS] ──► [SER] ──► [ARC] ──► [NPE] ──► [FM] ──► Output
            │         │         │         │         │
            ▼         ▼         ▼         ▼         ▼
         State    Route     Guard     Execute    Sync
         Space    Experts  I/O+Grad   Program   Memory
```

## Security Layer Detail

### S0 — Crypto Core (`src/security/c/`)
| Module | Purpose |
|--------|---------|
| SHA-3 | FIPS 202, all 4 variants (224/256/384/512) |
| ChaCha20 | Stream cipher, IETF variant (96-bit nonce) |
| Poly1305 | MAC, RFC 8439, constant-time |
| AEAD | ChaCha20-Poly1305 combined mode |
| Ed25519 | Sign/verify, RFC 8032 (304/306 pass) |
| BLAKE3 | Parallel hash, keyed hashing, KDF |
| Argon2 | Password hash, id (3/4 pass) |
| CT utils | Constant-time select, equal, compare |

### S1 — Secure Memory (`src/security/c/`)
| Module | Purpose |
|--------|---------|
| Secure pool | Guard pages, canaries, mlock, ASLR |
| Canary | 128-bit random + generation counter |
| ASLR | Page-aligned random offset |
| Locked mem | mlock/VirtualLock with EPERM warning |
| SC ops | Branchless select, equal, lt, is_zero |
| Timing | RDTSC/QPC, random delay, timing-safe equal |
| Cache | clflush, prefetch, mfence |
| Power | Dummy ops, balance regions |
| ASM | x86_64 CMOV-based constant-time (124 lines) |

### S2 — Obfuscation Engine (`src/security/cpp/`)
| Module | Purpose |
|--------|---------|
| CFG flatten | Switch-dispatch state machine, junk states |
| String encrypt | Compile-time XOR, runtime decrypt, secure wipe |
| Inst subst | LEA for ADD, NAND for AND/OR/XOR |
| Opaque pred | Math invariants, pointer self-compare |
| Code VM | Stack-based 256-register VM, encrypted handlers |
| Anti-debug | ptrace, IsDebuggerPresent, RDTSC, INT3 scan, CPUID |
| Pipeline | Levels LIGHT→MAXIMUM, semantic verify |

### S3 — Behavioral Monitor (Planned)
Runtime integrity, anomaly detection, hook detection, execution profiling.

## Directory Structure

```
ARIX_Algo/
├── CMakeLists.txt
├── src/
│   ├── core/                    # Foundation: tensor, memory, thread, autodiff, optimizer
│   ├── arch/                    # Algorithm pipeline: HSS, SER, ARC, NPE, FM, train
│   ├── security/
│   │   ├── c/                   # S0 — Crypto Core + S1 — Secure Memory (C)
│   │   ├── cpp/                 # S2 — Obfuscation Engine (C++)
│   │   ├── asm/                 # x86_64 assembly helpers
│   │   └── rust/                # Future: Rust security layer
│   └── python/                  # pybind11 bindings
├── tests/
│   ├── unit/                    # Component unit tests
│   ├── integration/             # Multi-component integration tests
│   ├── security/                # S0 + S1 (C) tests
│   └── security/cpp/            # S2 (C++) tests
├── examples/                    # Demos for each component
└── docs/                        # Documentation
```

## Data Flow

```
User Input
    │
    ▼
┌─────────────┐
│  S2 Anti-   │  Debugger check, timing anomaly detection
│  Debug      │  If detected → wipe sensitive data → exit
└──────┬──────┘
       │
┌──────▼──────┐
│  S1 Secure  │  Allocate with guard pages, canaries
│  Memory     │  Decrypt strings from S2 pool
└──────┬──────┘
       │
┌──────▼──────┐
│  Pipeline   │  HSS → SER → ARC → NPE → FM
│  Inference  │  Each step uses S0 crypto for RNG,
└──────┬──────┘  S1 allocators, S2 obfuscation
       │
┌──────▼──────┐
│  S0 Crypto  │  Sign output with Ed25519
│  Attest     │  (Future: generate ZK proof)
└──────┬──────┘
       │
       ▼
   Verified Output
```

## Key Properties

- **Constant-time**: All crypto operations are branchless
- **Memory-safe**: All allocations have guard pages + canaries
- **Obfuscatable**: Control flow, strings, instructions all transformable
- **Verifiable**: (Future) Every inference produces a cryptographic proof
- **On-device**: No cloud dependency for inference
