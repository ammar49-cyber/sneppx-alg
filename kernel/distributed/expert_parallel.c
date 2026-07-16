#ifdef SNEPPX_HAS_CUDA
#include "../../include/neural_core/architecture/distributed.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Expert Parallelism (Distributed MoE via All-to-All)
// ============================================================================

struct SNEPPX_ExpertParallel {
    int num_experts;
    int num_local_experts;
    int ep_size;
    int ep_rank;
    int* expert_to_rank;
    int* local_expert_ids;
};

int sneppx_ep_init(SNEPPX_ExpertParallel** ep,
                   const SNEPPX_DistributedConfig* config) {
    if (!ep || !config) return -1;
    
    SNEPPX_ExpertParallel* e = (SNEPPX_ExpertParallel*)calloc(1, sizeof(SNEPPX_ExpertParallel));
    if (!e) return -1;
    
    e->num_experts = config->num_experts;
    e->num_local_experts = config->num_experts / config->expert_parallel_size;
    e->ep_size = config->expert_parallel_size;
    e->ep_rank = config->rank % config->expert_parallel_size;
    
    // Build expert-to-rank mapping
    e->expert_to_rank = (int*)calloc(config->num_experts, sizeof(int));
    e->local_expert_ids = (int*)calloc(e->num_local_experts, sizeof(int));
    
    for (int i = 0; i < config->num_experts; i++) {
        e->expert_to_rank[i] = i % config->expert_parallel_size;
    }
    
    int local_idx = 0;
    for (int i = e->ep_rank; i < config->num_experts; i += config->expert_parallel_size) {
        if (local_idx < e->num_local_experts) {
            e->local_expert_ids[local_idx++] = i;
        }
    }
    
    *ep = e;
    return 0;
}

// All-to-All communication for expert dispatch
// sendbuf: [ep_size, send_size] per rank, recvbuf: [ep_size, recv_size]
int sneppx_ep_all_to_all(SNEPPX_ExpertParallel* ep,
                          const float* sendbuf, float* recvbuf,
                          int send_size, int recv_size,
                          cudaStream_t stream) {
    if (!ep || !sendbuf || !recvbuf) return -1;
    
    // Simplified: each rank copies its segment
    // In a real distributed system, this uses NCCL send/recv pairs
    int peer_rank = (ep->ep_rank + 1) % ep->ep_size;
    
    // Copy local portion
    cudaMemcpyAsync(recvbuf, sendbuf,
                    send_size * ep->ep_size * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    
    return 0;
}

int sneppx_ep_destroy(SNEPPX_ExpertParallel* ep) {
    if (!ep) return -1;
    free(ep->expert_to_rank);
    free(ep->local_expert_ids);
    free(ep);
    return 0;
}

// ============================================================================
// FM Distributed Communication
// ============================================================================

int sneppx_fm_distributed_all_reduce(float* data, int size,
                                      const SNEPPX_DistributedConfig* config,
                                      cudaStream_t stream) {
    if (!data || !config) return -1;
    
    // Placeholder: in real implementation, calls NCCL all-reduce
    return 0;
}

int sneppx_fm_distributed_broadcast(float* data, int size, int root,
                                     const SNEPPX_DistributedConfig* config,
                                     cudaStream_t stream) {
    if (!data || !config) return -1;
    
    // Placeholder: in real implementation, calls NCCL broadcast
    return 0;
}
#endif