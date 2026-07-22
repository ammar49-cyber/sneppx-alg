#ifndef SNEPPX_TRAINER_CUDA_H
#define SNEPPX_TRAINER_CUDA_H

#include "multidimensional_tensor_engine.h"
#include "gradient_optimization_suite.h"
#include <stddef.h>

int SNEPPX_trainer_cuda_available(void);
int SNEPPX_trainer_cuda_init(SNEPPXTensor** params, size_t n);
void SNEPPX_trainer_cuda_shutdown(void);
int SNEPPX_trainer_cuda_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n);
int SNEPPX_trainer_cuda_transfer_to_device(SNEPPXTensor** params, size_t n);
int SNEPPX_trainer_cuda_transfer_to_host(SNEPPXTensor** params, size_t n);

#endif /* SNEPPX_TRAINER_CUDA_H */
