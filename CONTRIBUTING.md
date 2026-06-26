# Contributing to ARIX-Algo

## Governance

BDFL: Ammar [ARIX]

All decisions final. No voting. No committees.

## How to Contribute

### 1. Email Patch

```bash
git format-patch -1 HEAD
git send-email --to=patches@arix.dev 0001-your-patch.patch
```

### 2. Wait for Review

Response time: 7 days typical, 30 days maximum.

### 3. Address Feedback

Amend commit. Resend.

## Patch Requirements

- [ ] Signed-off-by line
- [ ] GPG/Ed25519 signature
- [ ] Passes all tests
- [ ] No compiler warnings
- [ ] Follows coding style
- [ ] Includes tests for new code
- [ ] Updates documentation

## Coding Style

### C

- 4 spaces, no tabs
- snake_case for functions
- PascalCase for structs
- SCREAMING_SNAKE_CASE for macros
- 80 column limit
- Braces on same line

### C++

- Google style with exceptions
- snake_case for functions
- PascalCase for classes
- Smart pointers mandatory

### Python

- PEP 8
- Black formatter
- Type hints required

## Prohibited

- GitHub pull requests
- Discord/Slack discussions for technical decisions
- Corporate CLAs
- AI-generated code without human review

## License

By contributing, you agree to license your work under MIT.
