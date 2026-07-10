#include "../../include/neural_core/architecture/distributed.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// ZeRO Stage 1: Optimizer State Partitioning
// ============================================================================

typedef struct {
    int param_idx;
    int owning_rank;
    size_t numel;
    float* param_ptr;
    float* grad_ptr;
} ZeRO_ParamPartition;

struct SNEPPX_ZeroOptimizer {
    ZeRO_ParamPartition* partitions;
    int num_partitions;
    int world_size;
    int rank;
    int stage;
    float lr, beta1, beta2, epsilon, weight_decay;
    int step;
    float* exp_avg;
    float* exp_avg_sq;
    size_t local_numel;
};

static size_t sneppx_zero_partition_size(int total, int world_size, int rank) {
    size_t base = total / world_size;
    size_t rem = total % world_size;
    return base + (rank < (int)rem ? 1 : 0);
}

static int sneppx_zero_partition_start(int total, int world_size, int rank) {
    size_t base = total / world_size;
    size_t rem = total % world_size;
    if (rank < (int)rem) {
        return rank * (base + 1);
    }
    return rem * (base + 1) + (rank - rem) * base;
}

int sneppx_zero_init(SNEPPX_ZeroOptimizer** opt, int num_params,
                     int world_size, int rank, int stage,
                     const SNEPPX_DistributedConfig* config) {
    if (!opt || num_params <= 0) return -1;
    
    SNEPPX_ZeroOptimizer* o = (SNEPPX_ZeroOptimizer*)calloc(1, sizeof(SNEPPX_ZeroOptimizer));
    if (!o) return -1;
    
    o->world_size = world_size;
    o->rank = rank;
    o->stage = stage;
    o->lr = config ? config->loss_scale : 0.001f;
    o->beta1 = 0.9f;
    o->beta2 = 0.999f;
    o->epsilon = 1e-8f;
    o->weight_decay = 0.01f;
    o->step = 0;
    
    size_t local_n = sneppx_zero_partition_size(num_params, world_size, rank);
    o->local_numel = local_n;
    
    if (stage >= 1) {
        cudaMalloc(&o->exp_avg, local_n * sizeof(float));
        cudaMalloc(&o->exp_avg_sq, local_n * sizeof(float));
        cudaMemset(o->exp_avg, 0, local_n * sizeof(float));
        cudaMemset(o->exp_avg_sq, 0, local_n * sizeof(float));
    }
    
    if (stage >= 2) {
        cudaMalloc(&o->grad_buffer, local_n * sizeof(float));
        cudaMemset(o->grad_buffer, 0, local_n * sizeof(float));
    }
    
    o->num_partitions = 1;
    o->partitions = (ZeRO_ParamPartition*)calloc(1, sizeof(ZeRO_ParamPartition));
    o->partitions[0].param_idx = 0;
    o->partitions[0].owning_rank = rank;
    o->partitions[0].numel = local_n;
    
    *opt = o;
    return 0;
}

static __global__ void zero_adamw_kernel(
    float* params, const float* grads,
    float* exp_avg, float* exp_avg_sq,
    int step, float lr, float beta1, float beta2,
    float epsilon, float weight_decay, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float p = params[idx];
    float g = grads[idx];
    float m = exp_avg[idx];
    float v = exp_avg_sq[idx];
    
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
    
    float m_hat = m / (1.0f - powf(beta1, (float)step));
    float v_hat = v / (1.0f - powf(beta2, (float)step));
    
    p -= lr * weight_decay * p;
    p -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    params[idx] = p;
}

int sneppx_zero_step(SNEPPX_ZeroOptimizer* opt, float* grads,
                     cudaStream_t stream) {
    if (!opt || !grads) return -1;
    
    opt->step++;
    int block = 256;
    int grid = (opt->local_numel + block - 1) / block;
    
    zero_adamw_kernel<<<grid, block, 0, stream>>>(
        opt->params_fp32, grads,
        opt->exp_avg, opt->exp_avg_sq,
        opt->step, opt->lr, opt->beta1, opt->beta2,
        opt->epsilon, opt->weight_decay, opt->local_numel
    );
    
    // Stage 3: broadcast updated params to all ranks
    if (opt->stage >= 3) {
        // All-gather params
        size_t total = opt->partitions[0].numel * opt->world_size;
        float* buffer;
        cudaMalloc(&buffer, total * sizeof(float));
        cudaMemcpy(buffer + sneppx_zero_partition_start(
            total * opt->world_size, opt->world_size, opt->rank),
            opt->params_fp32, opt->local_numel * sizeof(float),
            cudaMemcpyDeviceToDevice);
        // Broadcast
        cudaMemcpy(opt->params_fp32, buffer, total * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaFree(buffer);
    }
    
    return 0;
}

int sneppx_zero_destroy(SNEPPX_ZeroOptimizer* opt) {
    if (!opt) return -1;
    if (opt->exp_avg) cudaFree(opt->exp_avg);
    if (opt->exp_avg_sq) cudaFree(opt->exp_avg_sq);
    if (opt->grad_buffer) cudaFree(opt->grad_buffer);
    free(opt->partitions);
    free(opt);
    return 0;
}