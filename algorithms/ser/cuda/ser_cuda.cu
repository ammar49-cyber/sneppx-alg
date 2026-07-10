#include "ser_cuda.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Top-k gating (softmax + top-k selection)
__global__ void topk_gating_kernel(
    const half* router_logits, int* expert_indices,
    half* gating_weights, int batch_size, int num_experts, int top_k
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each warp handles one batch item
    int warp_id = tid / SNEPPX_WARP_SIZE;
    int lane_id = tid % SNEPPX_WARP_SIZE;
    
    if (warp_id >= 1) return;
    
    // Load logits for this batch
    __shared__ float logits[64];  // max experts per shared
    float vals[2];
    
    if (lane_id < num_experts) {
        logits[lane_id] = __half2float(router_logits[batch * num_experts + lane_id]);
    }
    __syncthreads();
    
    // Softmax
    float max_val = -INFINITY;
    for (int i = lane_id; i < num_experts; i += SNEPPX_WARP_SIZE) {
        max_val = fmaxf(max_val, logits[i]);
    }
    max_val = sneppx_warp_reduce_max(max_val);
    
    float sum = 0.0f;
    for (int i = lane_id; i < num_experts; i += SNEPPX_WARP_SIZE) {
        float e = expf(logits[i] - max_val);
        logits[i] = e;
        sum += e;
    }
    sum = sneppx_warp_reduce_sum(sum);
    
    if (sum > 0.0f) {
        for (int i = lane_id; i < num_experts; i += SNEPPX_WARP_SIZE) {
            logits[i] /= sum;
        }
    }
    __syncthreads();
    
    // Top-k selection (simple selection sort for small k)
    for (int k = 0; k < top_k; k++) {
        float best_val = -1.0f;
        int best_idx = -1;
        
        for (int i = 0; i < num_experts; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        
        if (lane_id == 0 && best_idx >= 0) {
            expert_indices[batch * top_k + k] = best_idx;
            gating_weights[batch * top_k + k] = __float2half_rn(best_val);
            logits[best_idx] = -1.0f;  // Remove from selection
        }
        __syncthreads();
    }
}

SNEPPX_CudaError sneppx_cuda_topk_gating(
    SNEPPX_CudaStream_t stream,
    const half* router_logits, int* expert_indices,
    half* gating_weights, int batch_size, int num_experts, int top_k
) {
    if (!router_logits || !expert_indices || !gating_weights) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    topk_gating_kernel<<<batch_size, 32, 0, stream>>>(
        router_logits, expert_indices, gating_weights, batch_size, num_experts, top_k
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Dispatch tokens to experts (scatter)
__global__ void dispatch_to_experts_kernel(
    const half* tokens, half* expert_inputs,
    const int* expert_indices, const int* expert_counts,
    int total_tokens, int num_experts, int expert_capacity, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / dim;
    int d = idx % dim;
    
    if (token_idx >= total_tokens) return;
    
    for (int k = 0; k < 1; k++) {  // Simplified: only top-1 dispatch
        int expert = expert_indices[token_idx];
        if (expert < 0 || expert >= num_experts) continue;
        
        // Use prefix sum count to get position
        int pos = expert_counts[expert];
        if (pos < expert_capacity) {
            expert_inputs[(expert * expert_capacity + pos) * dim + d] = tokens[token_idx * dim + d];
        }
    }
}

SNEPPX_CudaError sneppx_cuda_dispatch_to_experts(
    SNEPPX_CudaStream_t stream,
    const half* tokens, half* expert_inputs,
    const int* expert_indices, const int* expert_counts,
    int total_tokens, int num_experts, int expert_capacity, int dim
) {
    if (!tokens || !expert_inputs || !expert_indices) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int total = total_tokens * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    dispatch_to_experts_kernel<<<grid, block, 0, stream>>>(
        tokens, expert_inputs, expert_indices, expert_counts,
        total_tokens, num_experts, expert_capacity, dim
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Combine from experts (gather + weighted sum)
__global__ void combine_from_experts_kernel(
    half* output, const half* expert_outputs,
    const int* expert_indices, const half* gating_weights,
    const int* expert_counts,
    int total_tokens, int num_experts, int dim, int top_k
) {
    int token_idx = blockIdx.x;
    int d = threadIdx.x;
    if (d >= dim) return;
    
    float result = 0.0f;
    for (int k = 0; k < top_k; k++) {