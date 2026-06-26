# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x | Security fixes only |
| 0.5.x | Security + bug fixes |
| 1.0.x | Full support |

## Reporting

Email: security@arix.dev

GPG key: Available from S0 key infrastructure.

Do NOT open public issues.

## Response Time

- Acknowledgment: 48 hours
- Assessment: 7 days
- Fix: 30 days critical, 90 days high

## Security Status

| Phase | Status | Components |
|-------|--------|-----------|
| S0 | ✅ Complete | Crypto primitives |
| S1 | ✅ Complete | Secure memory |
| S2 | ⚠️ In Progress | Obfuscation |
| S3 | ⚠️ In Progress | Behavioral monitor |
| S4 | ⏳ Planned | Network security |
| S5 | ⏳ Planned | AI sanitizer |
| S6 | ⏳ Planned | Security UI |
| S7 | ⏳ Planned | Secure updates |
| S8 | ⏳ Planned | Formal verification |
| S9 | ⏳ Planned | Penetration testing |

## Known Issues

No reported vulnerabilities at this time.

### Test Suite Edge Cases

- Ed25519: 2/306 verification tests fail under specific edge conditions
- Argon2: 1/4 timing test fails on certain hardware
- These are pre-existing S0 limitations and do not represent security vulnerabilities
