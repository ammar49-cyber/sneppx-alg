#ifndef SNEPPX_RDMA_MANAGER_H
#define SNEPPX_RDMA_MANAGER_H
/*
 * RDMA Communication Manager — v1.0 (zero-copy distributed transport)
 *
 * PURPOSE: InfiniBand / RoCE one-sided read/write for fast tensor transfers.
 * Memory regions are pinned (registered) with the NIC.  Queue pairs (QPs)
 * manage send/receive work requests.  Completion queues (CQs) notify
 * the thread pool when transfers finish.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h, concurrent_workload_dispatch.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*   context;          /* ibv_context* */
    void*   pd;               /* ibv_pd* */
    int     port_num;
    int     active_qps;
} SNEPPXRDMAContext;

typedef struct {
    void*   mr;               /* ibv_mr* */
    void*   addr;
    size_t  length;
    uint32_t rkey;
    uint32_t lkey;
} SNEPPXRDMARegion;

typedef struct {
    void*   qp;               /* ibv_qp* */
    int     qp_num;
    int     state;
    uint64_t send_bytes;
    uint64_t recv_bytes;
} SNEPPXRDMAQueuePair;

typedef struct {
    void*   cq;               /* ibv_cq* */
    int     num_entries;
} SNEPPXRDMACompletionQueue;

/* ---------- Lifecycle ---------- */
int SNEPPX_rdma_open(SNEPPXRDMAContext** ctx, int device_idx);
void SNEPPX_rdma_close(SNEPPXRDMAContext* ctx);

/* ---------- Memory registration ---------- */
int SNEPPX_rdma_register_memory(SNEPPXRDMAContext* ctx, void* addr, size_t len, SNEPPXRDMARegion** region);
int SNEPPX_rdma_deregister_memory(SNEPPXRDMAContext* ctx, SNEPPXRDMARegion* region);

/* ---------- Queue pairs ---------- */
int SNEPPX_rdma_create_qp(SNEPPXRDMAContext* ctx, SNEPPXRDMAQueuePair** qp);
int SNEPPX_rdma_connect_qp(SNEPPXRDMAQueuePair* qp, int remote_qp_num, int remote_lid);
int SNEPPX_rdma_disconnect_qp(SNEPPXRDMAQueuePair* qp);
void SNEPPX_rdma_destroy_qp(SNEPPXRDMAQueuePair* qp);

/* ---------- One-sided operations ---------- */
int SNEPPX_rdma_read(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                   uint64_t remote_addr, uint32_t rkey, size_t len);
int SNEPPX_rdma_write(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                    uint64_t remote_addr, uint32_t rkey, size_t len);

/* ---------- Completion ---------- */
int SNEPPX_rdma_poll_completion(SNEPPXRDMAContext* ctx, int* num_completions);

/* ---------- Tensor helpers (v1.0) ---------- */
int SNEPPX_rdma_send_tensor(SNEPPXRDMAContext* ctx, const void* tensor, int dest_rank);
int SNEPPX_rdma_recv_tensor(SNEPPXRDMAContext* ctx, void** tensor, int src_rank);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_RDMA_MANAGER_H */
