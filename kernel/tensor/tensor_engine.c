/* This translation unit intentionally defines NO public symbols.
 *
 * The full real tensor engine implementation lives in `tensor.c`
 * (SNEPPXTensor storage, views, elementwise/broadcast ops, reductions,
 * matmul, linalg, serialization, etc.). Both files are discovered by the
 * CMake GLOB and compiled into `neural_core_kernel`; keeping duplicate
 * (stubbed) definitions here previously caused the linker to drop the real
 * implementation in `tensor.c`. All real symbols are provided by `tensor.c`.
 */
#include "multidimensional_tensor_engine.h"
