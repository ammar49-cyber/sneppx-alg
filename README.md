# SNEPPX-Alg: Secure Neural Architecture (ARIX_Algo)

This directory contains the **SNEPPX-Alg** cognitive processing system — a
next-generation AI framework with security built into the foundation.

See the [top-level README](https://github.com/ammar49-cyber/sneppx-alg) and
[`Docs.md`](./Docs.md) for the full overview, build instructions, and the
S0–S9 security model.

## Quick build

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

Opt-in accelerator backends (compile as `UNSUPPORTED` stubs by default):

```powershell
cmake -B build -DSNEPPX_BUILD_METAL=ON     # Apple Metal reference backend
cmake -B build -DSNEPPX_BUILD_ONEAPI=ON    # Intel oneAPI/SYCL reference backend
```

## Layout

| Path | Purpose |
|------|---------|
| `kernel/` | Core tensor/autodiff/optimizer/trainer substrate |
| `algorithms/` | HSS, SER, ARC, NPE, FM algorithm implementations |
| `drivers/` | Accelerator backends (CUDA, ROCm, Metal*, oneAPI*, …) |
| `security/` | S0–S9 security layer |
| `net/` | Distributed + gRPC coordination |
| `bindings/python/` | Python API |
| `releases/` | Release signing tooling |

`*` Metal and oneAPI are reference-compute backends enabled via
`SNEPPX_BUILD_METAL` / `SNEPPX_BUILD_ONEAPI`.

## License

MIT — see [`LICENSE`](./LICENSE). Maintained by **Ammar [SNEPPX]**.
