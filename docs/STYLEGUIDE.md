# SNEPPX-Algo Style Guide

## C Style

- Indentation: 4 spaces, no tabs
- Line length: 80 columns maximum
- Braces: Attach style (same line)
- Pointer alignment: Right (`int* p`)
- Function names: snake_case
- Struct names: PascalCase
- Enum names: PascalCase
- Macro names: SCREAMING_SNAKE_CASE
- File names: snake_case.c / .h
- Include guards: `#ifndef SNEPPX_COMPONENT_H`, `#define SNEPPX_COMPONENT_H`

### Example

```c
#ifndef SNEPPX_TENSOR_H
#define SNEPPX_TENSOR_H

#define SNEPPX_MAX_NDIM 8

typedef struct {
    float* data;
    size_t shape[SNEPPX_MAX_NDIM];
    size_t ndim;
    size_t size;
} SNEPPXTensor;

SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
void SNEPPX_tensor_destroy(SNEPPXTensor* t);

#endif
```

## C++ Style

- Based on Google C++ Style with exceptions
- Namespace: `SNEPPX`
- Class names: PascalCase
- Method names: snake_case
- Variable names: snake_case
- Member variables: trailing underscore (`member_`)
- Smart pointers: `std::unique_ptr`, `std::shared_ptr`
- RAII for all resource management
- No exceptions in performance-critical hot paths
- `const` correctness: mark all immutable methods and parameters

### Example

```cpp
namespace SNEPPX {

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape);
    ~Tensor();

    size_t size() const { return size_; }
    float* data() { return data_.get(); }

private:
    std::unique_ptr<float[]> data_;
    std::vector<size_t> shape_;
    size_t size_;
};

}  // namespace SNEPPX
```

## Python Style

- PEP 8 compliance
- Line length: 88 columns (Black default)
- Formatter: Black (required)
- Type hints: required for all function signatures
- Docstrings: Google style for public APIs
- File names: snake_case.py
- Class names: PascalCase
- Function names: snake_case

### Example

```python
from typing import Optional


class Tensor:
    """Multi-dimensional array."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data.astype(np.float32)

    @property
    def shape(self) -> tuple:
        return tuple(self._data.shape)

    def numpy(self) -> np.ndarray:
        return self._data.copy()
```

## Commit Messages

```
component: Brief description in present tense

Longer description explaining motivation, approach, and any trade-offs.
Wrap at 72 characters.

Signed-off-by: Name <email>
```

- Prefix with component: `tensor:`, `hss:`, `ser:`, `arc:`, `npe:`, `fm:`,
  `security:`, `build:`, `docs:`, `tests:`, `python:`
- Signed-off-by line required
- GPG or Ed25519 commit signing required
- No merge commits (rebase workflow)

## Documentation Style

- Markdown for all documentation files
- Doxygen (`/** ... */`) for C/C++ API documentation comments
- Python docstrings for Python API
- Code examples in fenced code blocks with language tag
- Tables for structured data
- Links use relative paths within the repository

## Naming Conventions

| Entity | Convention | Example |
|--------|------------|---------|
| C files | snake_case | `tensor_ops.c` |
| C headers | snake_case | `SNEPPX_tensor.h` |
| C functions | snake_case | `SNEPPX_tensor_create` |
| C structs | PascalCase | `SNEPPXTensor` |
| C enums | PascalCase | `SNEPPXDtype` |
| C macros | SCREAMING_SNAKE_CASE | `SNEPPX_MAX_NDIM` |
| C++ files | snake_case | `bindings.cpp` |
| C++ classes | PascalCase | `Tensor` |
| C++ methods | snake_case | `tensor_create` |
| Python files | snake_case | `tensor.py` |
| Python classes | PascalCase | `Tensor` |
| Python functions | snake_case | `from_numpy` |
| Directories | snake_case | `src/core/` |
