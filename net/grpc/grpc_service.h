#ifndef SNEPPX_GRPC_SERVICE_H
#define SNEPPX_GRPC_SERVICE_H
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
} SNEPPXGRPCNodeInfo;

typedef enum {
    SNEPPX_GRPC_OK,
    SNEPPX_GRPC_UNAVAILABLE,
    SNEPPX_GRPC_DEADLINE_EXCEEDED,
    SNEPPX_GRPC_INTERNAL,
    SNEPPX_GRPC_UNAUTHENTICATED,
} SNEPPXGRPCStatus;

/* ---------- Service stubs ---------- */
typedef struct SNEPPXGRPCStub {
    void*   channel;           /* grpc_channel* */
    void*   coordination_stub; /* CoordinationService::Stub* */
    void*   transfer_stub;     /* TensorTransferService::Stub* */
    char    target[256];
    int     connected;
} SNEPPXGRPCStub;

/* ---------- Service server ---------- */
typedef struct SNEPPXGRPCServer {
    void*   server;            /* grpc_server* */
    int     port;
    int     is_running;
    void*   coordination_svc;  /* CoordinationService::Service* */
    void*   transfer_svc;      /* TensorTransferService::Service* */
} SNEPPXGRPCServer;

/* ---------- Lifecycle ---------- */
int  SNEPPX_grpc_server_start(SNEPPXGRPCServer** server, int port);
void SNEPPX_grpc_server_stop(SNEPPXGRPCServer* server);
void SNEPPX_grpc_server_wait(SNEPPXGRPCServer* server);

SNEPPXGRPCStub* SNEPPX_grpc_stub_create(const char* target);
void          SNEPPX_grpc_stub_destroy(SNEPPXGRPCStub* stub);

/* ---------- Coordination (v1.0) ---------- */
int SNEPPX_grpc_register_node(SNEPPXGRPCStub* stub, const SNEPPXGRPCNodeInfo* info);
int SNEPPX_grpc_get_world_size(SNEPPXGRPCStub* stub, int* size);
int SNEPPX_grpc_get_rank(SNEPPXGRPCStub* stub, int* rank);
int SNEPPX_grpc_barrier(SNEPPXGRPCStub* stub);
int SNEPPX_grpc_all_gather(SNEPPXGRPCStub* stub, const void* send_buf, void** recv_buf, size_t elem_size);

/* ---------- Tensor transfer (v1.0) ---------- */
int SNEPPX_grpc_send_tensor(SNEPPXGRPCStub* stub, const void* tensor, int dest_rank);
int SNEPPX_grpc_recv_tensor(SNEPPXGRPCStub* stub, void** tensor, int src_rank);

/* ---------- Auth (v1.0) ---------- */
int SNEPPX_grpc_set_auth_token(SNEPPXGRPCStub* stub, const char* token);

/* ---------- Utility ---------- */
const char* SNEPPX_grpc_status_string(SNEPPXGRPCStatus status);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_GRPC_SERVICE_H */
