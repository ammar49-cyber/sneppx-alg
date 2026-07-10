#include "../../include/neural_core/architecture/distributed.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Tensor Parallelism (Row/Column Split Linear)
// ============================================================================

struct SNEPPX_TensorParallel {
    int tp_size;
    int tp_rank;
    int use_nccl;
};

// Forward declarations for NCCL
extern int sneppx_nccl_all_reduce(
    const void* sendbuf, void* recvbuf, size_t count,
    int datatype, int op, void* comm, cudaStream_t stream);

int sneppx_tp_init(SNEPPX_TensorParallel** tp,
                   const SNEPPX_DistributedConfig* config) {
    if (!tp || !config) return -1;
    
    SNEPPX_TensorParallel* t = (SNEPPX_TensorParallel*)calloc(1, sizeof(SNEPPX_TensorParallel));
    if (!t) return -1;
    
    t->tp_size = config->tensor_parallel_size;
    t->tp_rank = config->rank % config->tensor_parallel_size;
    t->use_nccl = 1;
    
    *tp = t;
    return 0;
}

// Column-parallel linear: Y = X * W^T (split along K dimension)
// Input: [M, K], Weight: [N, K] split into [N/tp, K] per rank
// Output: partial [M, N/tp], all-reduce to get full [M, N]
int sneppx_tp_linear_forward(SNEPPX_TensorParallel* tp,
                              const float* input, const float* weight,
                              float* output, int m, int n, int k,
                              cudaStream_t stream) {
    if (!tp || !input || !weight || !output) return -1;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    // Each rank computes partial output
    int n_local = n / tp->tp_size;
    int rank_offset = tp->tp_rank * n_local;
    
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n_local, m, k,
                &alpha,
                weight, k,           // [n_local, k] partial weight
                input, k,            // [m, k] input
                &beta,
                output, n_local);    // [m, n_local] partial output
    
    // All-reduce to get full output
    // (in a real implementation, this would use NCCL)
    // For now, copy to contiguous buffer
    float* full_output;
    cudaMalloc(&full_output, m * n * sizeof(float));
    cudaMemcpy(full_output + rank_offset * m, output,
               n_local * m * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Simulate all-reduce by copying full output back
    cudaMemcpy(output, full_output, m * n * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaFree(full_output);
    
    cublasDestroy(handle);
    return 0;
}

// All-reduce wrapper for tensor parallelism
int sneppx_tp_all_reduce(SNEPPX_TensorParallel* tp,
                          float* data, int size,
                          cudaStream_t stream) {
    if (!tp || !data) return -1;
    
    // Use NCCL all-reduce if available
    // (placeholder - actual NCCL call would go here)
    
    return 0;
}

int sneppx_tp_destroy(SNEPPX_TensorParallel* tp) {
    if (!tp) return -1;
    free(tp);
    return 0;
}