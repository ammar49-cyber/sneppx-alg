# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Security fixes only |
| 0.5.x   | Security + bug fixes |
| 1.0.x   | Full support |

## Reporting a Vulnerability

Report security vulnerabilities privately. Do NOT open public issues.

Email: algoSNEPPX@gmail.com

GPG key: Available from S0 key infrastructure (Ed25519).

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions and components
- Any proof-of-concept code
- Your contact information for follow-up

### Response Timeline

| Phase | Timeframe |
|-------|-----------|
| Acknowledgment | 48 hours |
| Initial assessment | 7 days |
| Critical fix release | 30 days |
| High fix release | 90 days |

## Security Status

| Phase | Status | Components |
|-------|--------|------------|
| S0 | Complete | Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3, secure random, Argon2id |
| S1 | Complete | Guard pages, canaries, ASLR, locked memory, constant-time operations |
| S2 | Partial | Control flow flattening, string encryption, instruction substitution, opaque predicates, VM obfuscation, anti-debug |
| S3 | Partial | Behavioral monitor structure, anomaly detection, API hooking, syscall filtering stubs |
| S4 | Planned v0.5.0 | Network security: TLS 1.3, QUIC, certificate pinning |
| S5 | Planned v0.5.0 | AI sanitizer: prompt injection filter, output verification |
| S6 | Planned v1.0 | Security UI: key management vault, audit viewer |
| S7 | Planned v1.0 | Secure updates: signed delta updates, rollback protection |
| S8 | Planned v1.0 | Formal verification: Lean 4 proofs of crypto correctness |
| S9 | Planned v1.0 | Penetration testing: third-party audit, bug bounty |

## Known Issues

No reported vulnerabilities at this time.

### Test Suite Edge Cases

- Ed25519: 2/306 verification tests fail under specific edge conditions
- Argon2: 1/4 timing test fails on certain hardware
- These are pre-existing S0 limitations and do not represent security vulnerabilities

## Hall of Fame

No security researchers have been credited yet. Be the first.

## Disclosure Policy

- BDFL decides disclosure timeline
- Coordinated disclosure preferred
- No exploit bounty at this time
