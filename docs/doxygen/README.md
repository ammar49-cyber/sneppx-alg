# Doxygen API Documentation

Generate HTML documentation from C/C++ code comments:

```powershell
# From repo root
doxygen docs/doxygen/Doxyfile

# Output: docs/doxygen/output/html/index.html
```

Requirements: [Doxygen](https://www.doxygen.nl/download.html) 1.9+ installed and on PATH.

The configuration extracts docs from `include/neural_core/` (public headers). Private implementation details in `kernel/` are not included.
