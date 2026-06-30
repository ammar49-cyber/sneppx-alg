#ifndef ARIX_RDMA_MANAGER_H
#define ARIX_RDMA_MANAGER_H
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
} ArixRDMAContext;

typedef struct {
    void*   mr;               /* ibv_mr* */
    void*   addr;
    size_t  length;
    uint32_t rkey;
    uint32_t lkey;
} ArixRDMARegion;

typedef struct {
    void*   qp;               /* ibv_qp* */
    int     qp_num;
    int     state;
    uint64_t send_bytes;
    uint64_t recv_bytes;
} ArixRDMAQueuePair;

typedef struct {
    void*   cq;               /* ibv_cq* */
    int     num_entries;
} ArixRDMACompletionQueue;

/* ---------- Lifecycle ---------- */
int arix_rdma_open(ArixRDMAContext** ctx, int device_idx);
void arix_rdma_close(ArixRDMAContext* ctx);

/* ---------- Memory registration ---------- */
int arix_rdma_register_memory(ArixRDMAContext* ctx, void* addr, size_t len, ArixRDMARegion** region);
int arix_rdma_deregister_memory(ArixRDMAContext* ctx, ArixRDMARegion* region);

/* ---------- Queue pairs ---------- */
int arix_rdma_create_qp(ArixRDMAContext* ctx, ArixRDMAQueuePair** qp);
int arix_rdma_connect_qp(ArixRDMAQueuePair* qp, int remote_qp_num, int remote_lid);
int arix_rdma_disconnect_qp(ArixRDMAQueuePair* qp);
void arix_rdma_destroy_qp(ArixRDMAQueuePair* qp);

/* ---------- One-sided operations ---------- */
int arix_rdma_read(ArixRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                   uint64_t remote_addr, uint32_t rkey, size_t len);
int arix_rdma_write(ArixRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                    uint64_t remote_addr, uint32_t rkey, size_t len);

/* ---------- Completion ---------- */
int arix_rdma_poll_completion(ArixRDMAContext* ctx, int* num_completions);

/* ---------- Tensor helpers (v1.0) ---------- */
int arix_rdma_send_tensor(ArixRDMAContext* ctx, const void* tensor, int dest_rank);
int arix_rdma_recv_tensor(ArixRDMAContext* ctx, void** tensor, int src_rank);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_RDMA_MANAGER_H */
