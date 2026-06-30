#ifndef ARIX_GRPC_SERVICE_H
#define ARIX_GRPC_SERVICE_H
/*
 * gRPC Service Definitions — v1.0 (distributed coordination)
 *
 * PURPOSE: Training coordination over gRPC: node discovery, gradient
 * exchange, checkpoint synchronization, and control messages.
 * Protobuf message definitions live in a separate .proto file compiled
 * by protoc; this header declares the C service stubs.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h, concurrent_workload_dispatch.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Node identity ---------- */
typedef struct {
    uint64_t node_id;
    char     hostname[256];
    int      grpc_port;
    int      data_port;
    int      num_devices;
    int64_t  total_memory;
} ArixGRPCNodeInfo;

typedef enum {
    ARIX_GRPC_OK,
    ARIX_GRPC_UNAVAILABLE,
    ARIX_GRPC_DEADLINE_EXCEEDED,
    ARIX_GRPC_INTERNAL,
    ARIX_GRPC_UNAUTHENTICATED,
} ArixGRPCStatus;

/* ---------- Service stubs ---------- */
typedef struct ArixGRPCStub {
    void*   channel;           /* grpc_channel* */
    void*   coordination_stub; /* CoordinationService::Stub* */
    void*   transfer_stub;     /* TensorTransferService::Stub* */
    char    target[256];
    int     connected;
} ArixGRPCStub;

/* ---------- Service server ---------- */
typedef struct ArixGRPCServer {
    void*   server;            /* grpc_server* */
    int     port;
    int     is_running;
    void*   coordination_svc;  /* CoordinationService::Service* */
    void*   transfer_svc;      /* TensorTransferService::Service* */
} ArixGRPCServer;

/* ---------- Lifecycle ---------- */
int  arix_grpc_server_start(ArixGRPCServer** server, int port);
void arix_grpc_server_stop(ArixGRPCServer* server);
void arix_grpc_server_wait(ArixGRPCServer* server);

ArixGRPCStub* arix_grpc_stub_create(const char* target);
void          arix_grpc_stub_destroy(ArixGRPCStub* stub);

/* ---------- Coordination (v1.0) ---------- */
int arix_grpc_register_node(ArixGRPCStub* stub, const ArixGRPCNodeInfo* info);
int arix_grpc_get_world_size(ArixGRPCStub* stub, int* size);
int arix_grpc_get_rank(ArixGRPCStub* stub, int* rank);
int arix_grpc_barrier(ArixGRPCStub* stub);
int arix_grpc_all_gather(ArixGRPCStub* stub, const void* send_buf, void** recv_buf, size_t elem_size);

/* ---------- Tensor transfer (v1.0) ---------- */
int arix_grpc_send_tensor(ArixGRPCStub* stub, const void* tensor, int dest_rank);
int arix_grpc_recv_tensor(ArixGRPCStub* stub, void** tensor, int src_rank);

/* ---------- Auth (v1.0) ---------- */
int arix_grpc_set_auth_token(ArixGRPCStub* stub, const char* token);

/* ---------- Utility ---------- */
const char* arix_grpc_status_string(ArixGRPCStatus status);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_GRPC_SERVICE_H */
