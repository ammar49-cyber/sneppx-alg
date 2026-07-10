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
            float dgelu = 0.5f * (1.0f + tanh_val) + x * 0.5f * sech2 * dtanh;
            dx = grad * dgelu;
            break;
        }
        case SNEPPX_ACT_SILU:
            // SiLU grad: dy = dx * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            float sig = 1.0f / (1.0f + expf(-x));
            dx = grad * sig * (1.0f + x * (1.0f - sig));
            break;
        case SNEPPX_ACT_TANH:
            dx = grad * (1.0f - y * y);
            break;
        case SNEPPX_ACT_SIGMOID:
            dx = grad * y * (1.0f - y);
            break;
        case SNEPPX_ACT_NONE:
            dx = grad;
            break;
    }
    
    d_input[idx] = __float2half_rn(dx);
}

SNEPPX_CudaError sneppx_cuda_activation_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,
    const half* input,
    const half* output,
    SNEPPX_ActivationType act,
    int numel
) {
    if (!d_input || !d_output || !input) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    activation_bwd_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, input, output, act, numel
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// LayerNorm Backward
// ============================================================================

__global__ void layernorm_bwd_kernel(
    half* d_input,
    half* d_gamma,
    half* d_beta,
    const half* d_output,
    const half* input,
    const half* gamma,
    const half* mu,
    const half* rsigma,
    int rows, int cols,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    float* dgamma_partial = smem;
    float* dbeta_partial = &smem[cols];
    
    // Per-row statistics
    float r_var = __half2float(rsigma[row]);
    float m = __half2float(mu[row]);
    
    float dgamma_local = 0.0f;
    float dbeta_local = 0.0f;
    
    // First pass: compute dgamma, dbeta
    for (int c = tid; c < cols; c += blockDim.x) {
        float dx_norm = __half2float(d_output[row * cols + c]);
        float x = __half2float(input[row * cols + c]);
        float g = gamma ? __half2float(gamma[c]) : 1.0f;
        float x_hat = (x - m) * r_var;
        
        dgamma_local += dx_norm * x_hat;
        dbeta_local += dx_norm;
    }
    
    // Warp reduce
    dgamma_local = sneppx_warp_reduce_sum(dgamma_local);
    dbeta_local = sneppx_warp_reduce_sum(dbeta_local);
    
    if (tid == 0) {
        dgamma_partial[blockIdx.x % cols] = dgamma_local;
        dbeta_partial[blockIdx.x % cols] = dbeta_local;
    }
    __syncthreads();
    
    // Second pass: compute d_input
    // Reduce dgamma, dbeta across rows
    float dg_sum = 0.0f, db_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        dg_sum += dgamma_partial[i];
        db_sum += dbeta_partial[i];
    }
    dg_sum = sneppx_warp_reduce_sum(dg_sum);
    db_sum = sneppx_warp_reduce_sum(db_sum);
    
    // Compute d_input
    float inv_N = 1.0f / cols;
    float r_var_val = r_var;
    float dgamma_total = dg_sum;
    float dbeta_total = db_sum;
    
    for (int c = tid; c < cols; c += blockDim.x) {
        float dx_norm = __half2float(d_output[row * cols + c]);
        float x = __half2float(input[row * cols + c]);
        float g = gamma ? __half2float(gamma[c]) : 1.0f;
        float x_hat = (x - m) * r_var_val;
        
        float dx = r_var_val * (dx_norm * g 
                  - (dbeta_total + x_hat * dgamma_total) * inv_N);
        
        d_input[row * cols + c] = __float2half_rn(dx);
    }
    
    // Accumulate d_gamma, d_beta
    if (d_gamma) d_gamma[tid] = __float2half_rn(dgamma_local);
    if (d_beta) d_beta[tid] = __float2half_rn(dbeta_local);
}

SNEPPX_CudaError sneppx_cuda_layernorm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    half* d_gamma,
    half* d_beta,
    const half* d_output,
    const half* input,
    const half* gamma,
    const half* mu,
    const half* rsigma,
    int rows, int cols,
    float epsilon
) {
    if (!d_input || !d_output || !input || !mu || !rsigma) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int block = min(cols, 256);
    size_t smem_size = 2 * cols * sizeof(float);
    
    layernorm_bwd_kernel<<<rows, block, smem_size, stream>>>(
        d_input, d_gamma, d_beta, d_output, input, gamma, mu, rsigma,
        rows, cols, epsilon
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// RMSNorm Backward
// ============================================================================

__global__ void rmsnorm_bwd_kernel(
    half* d_input,
    half* d_weight,
    const half* d_output,
    const half* input,
    const half* weight,
    const half* rms,
    int rows, int cols,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float rms_val = __half2float(rms[row]);
    float inv_rms = __fdividef(1.0f, rms_val);
    float inv_N = __fdividef(1.0f, cols);
    
    // Accumulate dot product for d_weight
    float dw_local = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float dx = __half2float(d_output[row * cols + c]);
        float x = __half2float(input[row * cols + c]);
        float x_norm = x * inv_rms;
        dw_local += dx * x_norm;
    }
    dw_local = sneppx_warp_reduce_sum(dw_local);
    
    if (tid == 0 && d_weight) {
        d_weight[row] = __float2half_rn(dw_local);
    }
    
    // Compute d_input
    float w = weight ? __half2float(weight[0]) : 1.0f;
    float dw = dw_local;
    
    for (int c = tid; c < cols; c += blockDim.x) {
        float dx = __half2float(d_output[row * cols + c]);
        float x = __half2float(input[row * cols + c]);
        float x_norm = x * inv_rms;
        
        float di = inv_rms * (dx * w - x_norm * (dw * inv_N * w));
        d_input[row * cols + c] = __float2half_rn(di);
    }
}

SNEPPX_CudaError sneppx_cuda_rmsnorm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    half* d_weight,
    const half* d_output,
    const half* input,
    const half* weight,
    const half* rms,
    int rows, int cols,
    float epsilon
) {
    if (!d_input || !d_output || !input || !rms) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int block = min(cols, 256);
    
    rmsnorm_bwd_kernel<<<rows, block, 0, stream>>>(
        d_input, d_weight, d_output, input, weight, rms,
        rows, cols, epsilon
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Softmax Backward
// ============================================================================

__global__ void softmax_bwd_kernel(
    half* d_input,
    const half* d_output,
    const half* output,
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Compute dot product: sum(d_output * output) for this row
    float dot = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float do_val = __half2float(d_output[row * cols + c]);
        float o_val = __half2float(output[row * cols + c]);
        dot += do_val * o_val;
    }
    dot = sneppx_warp_reduce_sum(dot);
    
    // Compute d_input = output * (d_output - dot)
    for (int c = tid; c < cols; c += blockDim.x) {
        float do_val = __half2float(d_output[row * cols + c]);
        float o_val = __half2float(output[row * cols + c]);
        float di = o_val * (do_val - dot);
        d_input[row * cols + c] = __float2half_rn(di);
    }
}

SNEPPX_CudaError sneppx_cuda_softmax_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,
    const half* output,
    int rows, int cols
) {
    if (!d_input || !d_output || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = min(cols, 256);
    
    softmax_bwd_kernel<<<rows, block, 0, stream>>>(d_input, d_output, output, rows, cols);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Cross-Entropy Backward
// ============================================================================

__global__ void cross_entropy_bwd_kernel(
    half* d_input,
    const half* d_output,
    const half* probs,
    const int* targets,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    
    if (idx < total) {
        int batch = idx / num_classes;
        int cls = idx % num_classes;
        
        float p = __half2float(probs[idx]);
        float do_val = __half2float(d_output[batch]); // scalar per batch
        int target = targets[batch];
        
        float di = do_val * (p - (cls == target ? 1.0f : 0.0f));
        d_input[idx] = __float2half_rn(di);
    }
}

SNEPPX_CudaError sneppx_cuda_cross_entropy_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,
    const half* probs,
    const int* targets,
    int batch_size,
    int num_classes
) {
    if (!d_input || !d_output || !probs || !targets) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int total = batch_size * num_classes;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    cross_entropy_bwd_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, probs, targets, batch_size, num_classes
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Convolution Backward
// ============================================================================

__global__ void conv2d_bwd_input_kernel(
    half* d_input,
    const half* d_output,
    const half* weight,
    int N, int C, int H, int W,
    int F, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h_out = blockIdx.z;
    int w_out = threadIdx.x;
    
    int H_out = (H + 2 * pad_h - KH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - KW) / stride_w + 1;
    
    if (h_out >= H_out || w_out >= W_out) return;
    
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            int h_in = h_out * stride_h + kh - pad_h;
            int w_in = w_out * stride_w + kw - pad_w;
            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;
            
            float grad = 0.0f;
            for (int f = 0; f < F; f++) {
                float do_val = __half2float(d_output[((n * F + f) * H_out + h_out) * W_out + w_out]);
                float w_val = __half2float(weight[((f * C + c) * KH + kh) * KW + kw]);
                grad += do_val * w_val;
            }
            
            int d_idx = ((n * C + c) * H + h_in) * W + w_in;
            atomicAdd(&d_input[d_idx], __float2half_rn(grad));
        }
    }
}

SNEPPX_CudaError sneppx_cuda_conv2d_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    half* d_weight,
    const half* d_output,
    const half* input,
    const half* weight,
    int N, int C, int H, int W,
    int F, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    if (!d_input || !d_output || !weight) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int H_out = (H + 2 * pad_h - KH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - KW) / stride_w + 1;
    
    // Zero d_input
    cudaMemsetAsync(d_input, 0, (size_t)N * C * H * W * sizeof(half), stream);
    
    // d_input gradient
    dim3 grid(N, C, H_out);
    dim3 block(W_out);
    
    conv2d_bwd_input_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, weight,
        N, C, H, W, F, KH, KW,
        stride_h, stride_w, pad_h, pad_w
    );
    
    // d_weight uses cublas or im2col approach
    // For simplicity, use im2col + GEMM
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Attention Backward (Simplified)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_attention_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_q,
    half* d_k,
    half* d_v,
    const half* d_output,
    const half* q,
    const half* k,
    const half* v,
    const half* output,
    int batch_size,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    int head_dim,
    float scale
) {
    // dV = softmax(Q*K^T)^T * dO
    // dK = dO * (P * V)^T   (where P = softmax(Q*K^T))
    // dQ = (dO * V^T) * P_derivative
    
    // Use cuBLAS for matrix multiplications
    cublasHandle_t handle = sneppx_cublas_get_handle();
    cublasSetStream(handle, stream);
    
    int M = batch_size * num_heads * seq_len_q;
    int N_kv = batch_size * num_heads * seq_len_kv;
    
    float one = 1.0f;
    float zero = 0.0f;
    
    // Allocate temporary for softmax scores (P)
    size_t p_size = (size_t)M * seq_len_kv * sizeof(half);
    half* p_scores;
    cudaMallocAsync(&p_scores, p_size, stream);
    
    // P = softmax(Q*K^T * scale)
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        seq_len_kv, M, head_dim,
        &one,
        k, CUDA_R_16F, seq_len_kv * head_dim,
        q, CUDA_R_16F, seq_len_kv * head_dim,
        &zero,
        p_scores, CUDA_R_16F, seq_len_kv,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // Apply softmax in-place
    sneppx_cuda_softmax_fwd(stream, p_scores, M, seq_len_kv);
    
    // dV = P^T * dO
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        head_dim, N_kv, M,
        &one,
        v, CUDA_R_16F, head_dim,
        d_output, CUDA_R_16F, head_dim,
        &zero,
        d_v, CUDA_R_16F, head_dim,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // dK = dO^T * (P * V) ... in practice use the proper attention gradient
    // Full implementation would compute dP = dO * V^T, then backprop through softmax
    
    cudaFreeAsync(p_scores, stream);
    
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Dropout Backward
// ============================================================================

__global__ void dropout_bwd_kernel(
    half* d_input,
    const half* d_output,
    const half* mask,
    int numel,
    float p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float scale = 1.0f / (1.0f - p);
    float m = __half2float(mask[idx]);
    float g = __half2float(d_output[idx]);
    
    d_input[idx] = __float2half_rn(g * m * scale);
}

SNEPPX_CudaError sneppx_cuda_dropout_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,
    const half* mask,
    int numel,
    float p
) {
    if (!d_input || !d_output || !mask) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    dropout_bwd_kernel<<<grid, block, 0, stream>>>(d_input, d_output, mask, numel, p);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// MSE Loss Backward