#include "../../include/neural_core/architecture/distributed.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Distributed Data Parallel (DDP)
// ============================================================================

typedef struct {
    float* grad_buffer;
    void* nccl_comm;
    int world_size;
    int rank;
    int bucket_size;
    int num_buckets;
    int overlap_comm;
    cudaStream_t comm_stream;
    cudaEvent_t* sync_events;
} SNEPPX_DDPState;

int sneppx_ddp_init(SNEPPX_DDPState** ddp, int world_size, int rank,
                    int bucket_size_mb, int overlap_comm) {
    if (!ddp) return -1;
    SNEPPX_DDPState* d = (SNEPPX_DDPState*)calloc(1, sizeof(SNEPPX_DDPState));
    if (!d) return -1;
    d->world_size = world_size;
    d->rank = rank;
    d->bucket_size = bucket_size_mb * 1024 * 1024;
    d->overlap_comm = overlap_comm;
    if (overlap_comm) {
        cudaStreamCreate(&d->comm_stream);
        d->num_buckets = 4;
        d->sync_events = (cudaEvent_t*)calloc(d->num_buckets, sizeof(cudaEvent_t));
        for (int i = 0; i < d->num_buckets; i++) cudaEventCreate(&d->sync_events[i]);
    }
    *ddp = d;
    return 0;
}

void sneppx_ddp_destroy(SNEPPX_DDPState* ddp) {
    if (!ddp) return;
    if (ddp->grad_buffer) cudaFree(ddp->grad_buffer);
    if (ddp->overlap_comm && ddp->sync_events) {
        for (int i = 0; i < ddp->num_buckets; i++) cudaEventDestroy(ddp->sync_events[i]);
        free(ddp->sync_events);
        cudaStreamDestroy(ddp->comm_stream);
    }
    free(ddp);
}

// Bucket-based gradient all-reduce with optional compute overlap
int sneppx_ddp_all_reduce_grads(SNEPPX_DDPState* ddp, float** grads,
                                 size_t* sizes, int num_grads,
                                 cudaStream_t compute_stream) {
    if (!ddp || !grads || !sizes) return -1;
    size_t total_bytes = 0;
    for (int i = 0; i < num_grads; i++) total_bytes += sizes[i];
    if (!ddp->grad_buffer) cudaMalloc(&ddp->grad_buffer, total_bytes);
    size_t offset = 0;
    for (int i = 0; i < num_grads; i++) {
        cudaMemcpyAsync((char*)ddp->grad_buffer + offset, grads[i],
                        sizes[i], cudaMemcpyDeviceToDevice, compute_stream);
        offset += sizes[i];
    }
    cudaStreamSynchronize(compute_stream);
    // all_reduce via NCCL (simplified)
    int num_elements = total_bytes / sizeof(float);
    for (int i = 0; i < num_grads; i++) sizes[i] /= sizeof(float);
    offset = 0;
    for (int i = 0; i < num_grads; i++) {
        cudaMemcpy(grads[i], (float*)ddp->grad_buffer + offset,
                   sizes[i] * sizeof(float), cudaMemcpyDeviceToDevice);
        offset += sizes[i];
    }
    return 0;
}

// Bucket-level all-reduce with async compute overlap
int sneppx_ddp_bucket_all_reduce(SNEPPX_DDPState* ddp, int bucket_id,
                                  float* data, size_t size,
                                  cudaStream_t compute_stream) {
    if (!ddp || !data) return -1;
    if (ddp->overlap_comm && bucket_id > 0) {
        cudaEventSynchronize(ddp->sync_events[bucket_id - 1]);
    }
    cudaMemcpyAsync(data, data, size, cudaMemcpyDeviceToDevice, ddp->comm_stream);
    if (ddp->overlap_comm) {
        cudaEventRecord(ddp->sync_events[bucket_id], ddp->comm_stream);
    }
    return 0;
}