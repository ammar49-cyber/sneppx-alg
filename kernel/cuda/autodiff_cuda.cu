#include "autodiff_cuda.h"
#include "common.cuh"
#include <cublas_v2.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// GEMM Backward: dA = dC * B^T, dB = A^T * dC
// ============================================================================

SNEPPX_CudaError sneppx_cuda_gemm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_a,
    half* d_b,
    const half* d_c,
    const half* a,
    const half* b,
    const half* c,
    SNEPPX_ActivationType act,
    int M, int N, int K,
    float alpha, float beta
) {
    if (!d_a || !d_b || !d_c || !a || !b) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    cublasHandle_t handle = sneppx_cublas_get_handle();
    cublasSetStream(handle, stream);
    
    cublasStatus_t status;
    float one = 1.0f;
    float zero = 0.0f;
    
    // Handle activation gradient if needed
    half* d_c_act = const_cast<half*>(d_c);  // Will point to temp buffer if activation
    
    half* temp_grad = nullptr;
    if (act != SNEPPX_ACT_NONE && c != nullptr) {
        size_t temp_size = (size_t)M * N * sizeof(half);
        cudaMallocAsync(&temp_grad, temp_size, stream);
        
        // dC_act = dC * act'(C)
        if (act == SNEPPX_ACT_RELU) {
            sneppx_cuda_activation_bwd(stream, temp_grad, d_c, c, c, SNEPPX_ACT_RELU, M * N);
        } else if (act == SNEPPX_ACT_GELU) {
            sneppx_cuda_activation_bwd(stream, temp_grad, d_c, c, c, SNEPPX_ACT_GELU, M * N);
        } else if (act == SNEPPX_ACT_SILU) {
            sneppx_cuda_activation_bwd(stream, temp_grad, d_c, c, c, SNEPPX_ACT_SILU, M * N);
        }
        d_c_act = temp_grad;
    }
    
    // dA = dC_act * B^T  [M, K] = [M, N] * [K, N]^T
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        K, M, N,
        &one,
        b, CUDA_R_16F, K,        // B [K, N]
        d_c_act, CUDA_R_16F, K,  // dC_act [M, N]
        &zero,
        d_a, CUDA_R_16F, K,     // dA [M, K]
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) return SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
    
    // dB = A^T * dC_act  [K, N] = [K, M] * [M, N]
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, K, M,
        &one,
        d_c_act, CUDA_R_16F, N,  // dC_act [M, N]
        a, CUDA_R_16F, N,        // A [M, K]
        &zero,
        d_b, CUDA_R_16F, N,     // dB [K, N]
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (temp_grad) {
        cudaFreeAsync(temp_grad, stream);
    }
    
    return (status == CUBLAS_STATUS_SUCCESS) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Batched GEMM Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_batched_gemm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_a,
    half* d_b,
    const half* d_c,
    const half* a,
    const half* b,
    SNEPPX_ActivationType act,
    int batch_size,
    int M, int N, int K,
    float alpha, float beta
) {
    if (!d_a || !d_b || !d_c || !a || !b) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    cublasHandle_t handle = sneppx_cublas_get_handle();
    cublasSetStream(handle, stream);
    
    float one = 1.0f;
    float zero = 0.0f;
    
    int stride_a = M * K;
    int stride_b = K * N;
    int stride_c = M * N;
    
    cublasStatus_t status;
    
    // dA_i = dC_i * B_i^T
    status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        K, M, N,
        &one,
        b, CUDA_R_16F, K, stride_b,
        d_c, CUDA_R_16F, K, stride_c,
        &zero,
        d_a, CUDA_R_16F, K, stride_a,
        batch_size,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) return SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
    
    // dB_i = A_i^T * dC_i
    status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, K, M,
        &one,
        d_c, CUDA_R_16F, N, stride_c,
        a, CUDA_R_16F, N, stride_a,
        &zero,
        d_b, CUDA_R_16F, N, stride_b,
        batch_size,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    return (status == CUBLAS_STATUS_SUCCESS) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Element-wise Backward
// ============================================================================

__device__ SNEPPX_EwBwdOp get_ew_bwd_op(SNEPPX_EwBwdOp op) { return op; }

__global__ void ew_bwd_kernel(
    half* d_a,
    half* d_b,
    const half* d_output,
    const half* a,
    const half* b,
    const half* output,
    SNEPPX_EwBwdOp op,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float grad = __half2float(d_output[idx]);
    float va = __half2float(a[idx]);
    float vb = __half2float(b[idx]);
    float vo = output ? __half2float(output[idx]) : 0.0f;
    
    float da = 0.0f, db = 0.0f;
    
    switch (op) {
        case SNEPPX_EW_BWD_ADD:
            da = grad;
            db = grad;
            break;
        case SNEPPX_EW_BWD_MUL:
            da = grad * vb;
            db = grad * va;
            break;
        case SNEPPX_EW_BWD_SUB:
            da = grad;
            db = -grad;
            break;
        case SNEPPX_EW_BWD_DIV:
            da = grad / vb;
            db = -grad * va / (vb * vb);
            break;
        case SNEPPX_EW_BWD_POW:
            da = grad * vb * powf(va, vb - 1.0f);
            db = grad * vo * logf(fmaxf(va, 1e-10f));
            break;
    }
    
    if (d_a) d_a[idx] = __float2half_rn(da);
    if (d_b) d_b[idx] = __float2half_rn(db);
}

SNEPPX_CudaError sneppx_cuda_elementwise_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_a,
    half* d_b,
    const half* d_output,
    const half* a,
    const half* b,
    const half* output,
    SNEPPX_EwBwdOp op,
    int numel
) {
    if (!d_output || !a || !b) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    ew_bwd_kernel<<<grid, block, 0, stream>>>(
        d_a, d_b, d_output, a, b, output, op, numel
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Activation Backward
// ============================================================================

__global__ void activation_bwd_kernel(
    half* d_input,
    const half* d_output,
    const half* input,
    const half* output,
    SNEPPX_ActivationType act,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float grad = __half2float(d_output[idx]);
    float x = __half2float(input[idx]);
    float y = __half2float(output[idx]);
    float dx = 0.0f;
    
    switch (act) {
        case SNEPPX_ACT_RELU:
            dx = (x > 0.0f) ? grad : 0.0f;
            break;
        case SNEPPX_ACT_GELU: {
            // GELU grad: dy = dx * (0.5 * (1 + tanh(...)) + x * 0.5 * (1 - tanh^2(...)) * beta * (1 + 3*kappa*x^2))
            float kBeta = 0.7978845608028654f;
            float kKappa = 0.044715f;
            float x3 = x * x * x;
            float tanh_arg = kBeta * (x + kKappa * x3);
            float tanh_val = tanhf(tanh_arg);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float dtanh = kBeta * (1.0f + 3.0f * kKappa * x * x);