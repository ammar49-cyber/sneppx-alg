# Profiles Directory

This directory contains CMake preset files and build profile configurations.

| File | Purpose |
|------|---------|
| `release-optimized.cmake` | Aggressive optimization for release builds |
| `debug-full.cmake` | Full debug symbols + sanitizers |

## Usage

```bash
cmake -B build -C profiles/release-optimized.cmake
cmake -B build-debug -C profiles/debug-full.cmake
```
