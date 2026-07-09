/*
 * RDMA Manager Implementation — SKELETON
 * VERSION: v1.0
 */

#include "rdma_manager.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_rdma_open(SNEPPXRDMAContext** ctx, int device_idx) {
    (void)device_idx;
    if (!ctx) return -1;
    *ctx = (SNEPPXRDMAContext*)calloc(1, sizeof(SNEPPXRDMAContext));
    return *ctx ? 0 : -1;
}

void SNEPPX_rdma_close(SNEPPXRDMAContext* ctx) { free(ctx); }

int SNEPPX_rdma_register_memory(SNEPPXRDMAContext* ctx, void* addr, size_t len, SNEPPXRDMARegion** region) {
    (void)ctx; (void)addr; (void)len;
    if (!region) return -1;
    *region = (SNEPPXRDMARegion*)calloc(1, sizeof(SNEPPXRDMARegion));
    return *region ? 0 : -1;
}

int SNEPPX_rdma_deregister_memory(SNEPPXRDMAContext* ctx, SNEPPXRDMARegion* region) {
    (void)ctx; free(region); return 0;
}

int SNEPPX_rdma_create_qp(SNEPPXRDMAContext* ctx, SNEPPXRDMAQueuePair** qp) {
    (void)ctx;
    if (!qp) return -1;
    *qp = (SNEPPXRDMAQueuePair*)calloc(1, sizeof(SNEPPXRDMAQueuePair));
    return *qp ? 0 : -1;
}

int SNEPPX_rdma_connect_qp(SNEPPXRDMAQueuePair* qp, int remote_qp_num, int remote_lid) {
    (void)qp; (void)remote_qp_num; (void)remote_lid; return 0;
}

int SNEPPX_rdma_disconnect_qp(SNEPPXRDMAQueuePair* qp) { (void)qp; return 0; }

void SNEPPX_rdma_destroy_qp(SNEPPXRDMAQueuePair* qp) { free(qp); }

int SNEPPX_rdma_read(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                   uint64_t remote_addr, uint32_t rkey, size_t len) {
    (void)qp; (void)local_addr; (void)lkey; (void)remote_addr; (void)rkey; (void)len;
    return 0;
}

int SNEPPX_rdma_write(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                    uint64_t remote_addr, uint32_t rkey, size_t len) {
    (void)qp; (void)local_addr; (void)lkey; (void)remote_addr; (void)rkey; (void)len;
    return 0;
}

int SNEPPX_rdma_poll_completion(SNEPPXRDMAContext* ctx, int* num_completions) {
    (void)ctx; if (num_completions) *num_completions = 0; return 0;
}

int SNEPPX_rdma_send_tensor(SNEPPXRDMAContext* ctx, const void* tensor, int dest_rank) {
    (void)ctx; (void)tensor; (void)dest_rank; return 0;
}

int SNEPPX_rdma_recv_tensor(SNEPPXRDMAContext* ctx, void** tensor, int src_rank) {
    (void)ctx; (void)src_rank; if (tensor) *tensor = NULL; return 0;
}
