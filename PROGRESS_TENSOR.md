# Tensor Core Progress

Target: 6,000 LOC (existing 2,484 + 3,000 new)
Timeline: 6 phases, 6 days

| Phase | Day | Focus | LOC Target | LOC Actual | Status |
|-------|-----|-------|------------|------------|--------|
| T0 | 1 | Audit & Foundation | 500 | 25 | Completed |
| T1 | 2 | Creation & Shape | 500 | | Pending |
| T2 | 3 | Element-wise Ops | 500 | | Pending |
| T3 | 4 | Reduction & LA | 500 | | Pending |
| T4 | 5 | NN Ops | 500 | | Pending |
| T5 | 6 | I/O & Tests | 500 | | Pending |

Total: 6,000 LOC

## T0 Deliverables
- [x] Audit existing tensor files (2,484 LOC across header, source, tests, benchmarks)
- [x] Add 4 dtype helper macros: `ARIX_DTYPE_SIZE`, `ARIX_DTYPE_IS_FLOAT`, `ARIX_DTYPE_IS_INT`, `ARIX_DTYPE_IS_COMPLEX`
- [x] Extend `ArixDevice`: +`ARIX_DEVICE_TPU`, +`ARIX_DEVICE_NPU`
- [x] Extend `ArixLayout`: +`ARIX_LAYOUT_TILED`
- [x] Declare `arix_tensor_save` / `arix_tensor_load`
- [x] Add stub implementations for save/load in tensor.c
- [x] Build passes with zero errors
- [x] All existing tests pass
