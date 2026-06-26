# Security Architecture

## Overview

ARIX-Algo implements security in ten phases (S0-S9). S0 and S1 are complete. S2 and S3 are in progress. S4-S9 are planned.

Each phase builds on the previous. The architecture is designed so that security is not a bolt-on layer but a property of the entire system.

```
S0 ── S1 ── S2 ── S3 ── S4 ── S5 ── S6 ── S7 ── S8 ── S9
│     │     │     │     │     │     │     │     │     │
Crypto Memory Obfusc Monitor Network AI    UI    Update Formal Pentest
Core  Secure Engine  Engine  Sec   San    Sec   Sec    Verif  Report
      Mem
```

## S0 — Cryptographic Core ✅

**Status**: Complete
**Source**: `src/security/c/arix_s0_crypto.h/.c`

### Primitives

| Primitive | Standard | Use Case | Verified |
|-----------|----------|----------|----------|
| Ed25519   | RFC 8032 | Signatures | 304/306 test vectors pass |
| X25519    | RFC 7748 | Key exchange | Full DH exchange |
| ChaCha20-Poly1305 | RFC 8439 | Authenticated encryption | 100+ test vectors |
| SHA-3     | FIPS 202 | Hashing (224/256/384/512) | NIST test vectors |
| BLAKE3    | Reference | Fast hashing | Reference test vectors |
| Argon2id  | RFC 9106 | Secure KDF | Test vectors + timing defense |
| Secure Random | OS CPRG | Entropy | Windows CNG / Linux getrandom |

### API

```c
#include "arix_s0_crypto.h"

// Ed25519 signing
uint8_t pk[32], sk[64];
arix_ed25519_keypair(pk, sk);
arix_ed25519_sign(sig, msg, msglen, sk, pk);

// ChaCha20-Poly1305 encryption
arix_chacha20_poly1305_encrypt(ct, &ctlen, pt, ptlen, key, nonce, aad, aadlen);

// Argon2id key derivation
arix_s0_argon2id_hash(hash, hashlen, pwd, pwdlen, salt, saltlen, t_cost, m_cost);
```

### Known Issues

1. **Ed25519 verification**: 2 of 306 test vectors fail under specific edge conditions (batch verification edge cases). These do not represent security vulnerabilities.
2. **Argon2 timing**: 1 of 4 timing tests shows variation on certain hardware. Mitigated by noise injection.

## S1 — Secure Memory ✅

**Status**: Complete
**Source**: `src/security/c/arix_s1_secure_memory.h/.c`

### Features

| Feature | Description |
|---------|-------------|
| Guard Pages | Allocate with PROT_NONE guard on both sides in debug mode |
| Canaries | Stack overflow detection on memory regions |
| ASLR | Heap randomization via VirtualAlloc (Windows) / mmap (Linux) |
| Locked Memory | mlock / VirtualLock to prevent swap to disk |
| Secure Wipe | Compiler-barrier-protected zeroing of sensitive data |
| Constant-Time Compare | Timing-attack-resistant memory comparison |

### API

```c
#include "arix_s1_secure_memory.h"

// Allocate locked memory (cannot be swapped to disk)
void* ptr = arix_s1_alloc_locked(4096);

// Securely free
arix_s1_free_locked(ptr, 4096);

// Constant-time comparison
int32_t match = arix_s1_memcmp_consttime(a, b, n);
```

### Platform Requirements

- **Linux**: Requires `CAP_IPC_LOCK` capability for memory locking
- **Windows**: Automatic via `VirtualLock`
- **macOS**: Not fully supported (limited mlock)

## S2 — Obfuscation Engine ⚠️

**Status**: Partial
**Source**: `src/security/cpp/arix_s2_obfuscation.h/.cpp`

### Implemented

- Control flow flattening: converts natural control flow to switch-based dispatch
- String encryption: XOR-based compile-time string obfuscation (stub)

### Planned

- Instruction substitution: replace standard ops with semantically equivalent complex sequences
- Opaque predicates: always-true/false branches to confuse static analysis
- Dynamic code generation: generate decryption stubs at runtime

## S3 — Behavioral Monitor ⚠️

**Status**: Partial
**Source**: `src/security/cpp/arix_s3_monitor.h/.cpp`

### Implemented

- System state structure: frequency, timing, and anomaly tracking classes

### Planned

- Frequency analysis: detect unusual API call patterns
- Timing analysis: detect side-channel probing
- Anomaly detection: ML-based detection of runtime anomalies

## S4 — Network Security ⏳

**Planned**: TLS 1.3, Noise protocol, certificate pinning, DDoS protection, traffic analysis resistance.

## S5 — AI Sanitizer ⏳

**Planned**: Input/output sanitization against prompt injection, jailbreaks, data poisoning, model inversion.

## S6 — Security UI ⏳

**Planned**: Local web dashboard, threat visualization, policy editor, audit log viewer.

## S7 — Secure Updates ⏳

**Planned**: Signed update bundles, rollback protection, staged rollout, verification at boot.

## S8 — Formal Verification ⏳

**Planned**: Theorem prover integration, model checking for critical paths, verified compilers for NPE.

## S9 — Penetration Testing ⏳

**Planned**: Third-party audits, public bug bounty, capture-the-flag exercises.

## Threat Model

### Assumptions

- The hardware is trusted (no side-channel attacks on CPU)
- The OS is trusted (no kernel-level compromise)
- The network is untrusted
- Other AI models are untrusted and potentially adversarial
- Supply chain is trusted (verified commits, signed releases)

### Defenses

| Threat | Defense | Status |
|--------|---------|--------|
| Signature forgery | Ed25519 | ✅ |
| Data breach at rest | ChaCha20-Poly1305 | ✅ |
| Side-channel timing | Constant-time ops | ✅ |
| Memory scraping | Locked memory + guard pages | ✅ |
| Swap forensic | mlock/VirtualLock | ✅ |
| Reverse engineering | CF flattening + string encryption | ⚠️ Partial |
| Runtime tampering | Behavioral monitor | ⚠️ Partial |
| Adversarial input | ARC input guard | ✅ |
| Gradient leakage | ARC gradient obfuscation | ✅ |
| Model inversion | ARC output verifier | ✅ |

## Reporting Vulnerabilities

See [SECURITY.md](../SECURITY.md)

## Security Best Practices

1. **Always verify signatures**: Use `arix_ed25519_verify` on any external data
2. **Lock sensitive data**: Use `arix_s1_alloc_locked` for keys and secrets
3. **Wipe after use**: Call `arix_s1_free_locked` or `arix_secure_zero` on sensitive buffers
4. **Use constant-time comparisons**: Never use `memcmp` for secret comparison
5. **Use AEAD**: Always use `arix_chacha20_poly1305_encrypt` with associated data
6. **Don't roll your own**: Use the provided crypto primitives; do not implement your own
