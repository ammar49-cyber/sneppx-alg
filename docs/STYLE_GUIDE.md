# Style Guide

ARIX-Algo follows strict coding conventions for consistency across the codebase.

## C Style

### Naming
- **Functions**: `snake_case` with `arix_` prefix: `arix_tensor_create`, `arix_hss_forward`
- **Types**: `PascalCase` with `Arix` prefix: `ArixTensor`, `ArixHSSConfig`
- **Macros**: `UPPER_SNAKE_CASE`: `ARIX_FLOAT32`, `ARIX_MAX_DIMS`
- **Constants**: `UPPER_SNAKE_CASE`: `MAX_TOKEN_LEN`
- **File-scope statics**: `s_` prefix: `s_global_pool`

### Formatting
- **Indentation**: 4 spaces, no tabs
- **Braces**: K&R style (opening brace on same line)
- **Line length**: 100 characters max
- **Spacing**:
  - One space after `if`, `for`, `while`, `do`, `switch`
  - No space between function name and `(`
  - Space around binary operators
  - No space around unary operators
- **Comments**: `//` for single line, `/* */` for block. No trailing comments on code lines.

### Headers
- Include guards: `#ifndef ARIX_MODULE_NAME_H` / `#define ARIX_MODULE_NAME_H`
- Standard headers first, then project headers, alphabetically
- Forward declarations preferred over includes when possible

### Error Handling
- Return `int` (0 = success, nonzero = error) for operations
- Return `NULL` (nullptr) for failed allocations
- Check all allocation returns
- Never assert in production code paths

### Memory
- Always use `arix_malloc` / `arix_free` with size tracking
- Parameter order: output params before input params where possible
- Document ownership transfer

## C++ Style

Used only for S2 obfuscation engine and pybind11 bindings.

- **RAII** everywhere
- No exceptions in hot paths
- `const` correctness
- `auto` for iterators and complex types only
- No RTTI
- No `dynamic_cast`

## Python Style

- PEP 8 compliant
- Type hints on all public functions
- Docstrings: Google style
- Tests use pytest

## Commit Messages

```
Component: brief description

Optional body explaining motivation and approach.
```

- First line: max 72 chars
- Use imperative mood: "Fix", not "Fixed"
- Reference issues when applicable

## Signing

All commits must be signed with GPG or Ed25519.
