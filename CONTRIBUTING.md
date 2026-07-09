# Contributing to SNEPPX-Algo

## Governance

BDFL: Ammar [SNEPPX] <algoSNEPPX@gmail.com>

All decisions final. No voting. No committees.

## How to Contribute

### 1. Prepare Your Patch

```bash
git format-patch -1 HEAD
```

### 2. Email Your Patch

Send to algoSNEPPX@gmail.com with subject prefix `[PATCH]`.

### 3. Wait for Review

Response time: 7 days typical, 30 days maximum.

### 4. Address Feedback

Amend your commit. Resend the updated patch.

## Patch Requirements

- [ ] Signed-off-by line present
- [ ] GPG or Ed25519 signature on commit
- [ ] All tests pass: `ctest --output-on-failure`
- [ ] No compiler warnings (use `-Wall -Wextra -Wpedantic`)
- [ ] Follows coding style (see STYLEGUIDE.md)
- [ ] Includes tests for new functionality
- [ ] Updates documentation where relevant

## Coding Style

### C

- 4 spaces, no tabs
- snake_case for functions
- PascalCase for structs
- SCREAMING_SNAKE_CASE for macros and enums
- 80 column limit
- Braces on same line (Attach style)
- Pointer asterisk on the right: `int* p`

### C++

- Google style with exceptions
- snake_case for functions and variables
- PascalCase for classes
- Smart pointers mandatory (unique_ptr, shared_ptr)
- No exceptions in hot paths
- RAII for resource management

### Python

- PEP 8
- Black formatter, 88 columns
- Type hints required for all function signatures
- Docstrings for public APIs

## Prohibited

- GitHub pull requests (email patches only)
- Discord or Slack discussions for technical decisions
- Corporate contributor license agreements (CLAs)
- AI-generated code without explicit human understanding and review

## License

By contributing, you agree to license your work under the MIT License.
