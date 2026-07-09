/*
 * gRPC Service Implementation — SKELETON
 * VERSION: v1.0
 */

#include "grpc_service.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_grpc_server_start(SNEPPXGRPCServer** server, int port) {
    (void)port;
    if (!server) return -1;
    *server = (SNEPPXGRPCServer*)calloc(1, sizeof(SNEPPXGRPCServer));
    if (*server) (*server)->port = port;
    return *server ? 0 : -1;
}

void SNEPPX_grpc_server_stop(SNEPPXGRPCServer* server) { free(server); }
void SNEPPX_grpc_server_wait(SNEPPXGRPCServer* server) { (void)server; }

SNEPPXGRPCStub* SNEPPX_grpc_stub_create(const char* target) {
    if (!target) return NULL;
    SNEPPXGRPCStub* stub = (SNEPPXGRPCStub*)calloc(1, sizeof(SNEPPXGRPCStub));
    if (stub) { strncpy(stub->target, target, sizeof(stub->target)-1); }
    return stub;
}

void SNEPPX_grpc_stub_destroy(SNEPPXGRPCStub* stub) { free(stub); }

int SNEPPX_grpc_register_node(SNEPPXGRPCStub* stub, const SNEPPXGRPCNodeInfo* info) {
    (void)stub; (void)info; return 0;
}
int SNEPPX_grpc_get_world_size(SNEPPXGRPCStub* stub, int* size) {
    (void)stub; if (size) *size = 1; return 0;
}
int SNEPPX_grpc_get_rank(SNEPPXGRPCStub* stub, int* rank) {
    (void)stub; if (rank) *rank = 0; return 0;
}
int SNEPPX_grpc_barrier(SNEPPXGRPCStub* stub) { (void)stub; return 0; }
int SNEPPX_grpc_all_gather(SNEPPXGRPCStub* stub, const void* send_buf, void** recv_buf, size_t elem_size) {
    (void)stub; (void)send_buf; (void)recv_buf; (void)elem_size; return 0;
}
int SNEPPX_grpc_send_tensor(SNEPPXGRPCStub* stub, const void* tensor, int dest_rank) {
    (void)stub; (void)tensor; (void)dest_rank; return 0;
}
int SNEPPX_grpc_recv_tensor(SNEPPXGRPCStub* stub, void** tensor, int src_rank) {
    (void)stub; (void)src_rank; if (tensor) *tensor = NULL; return 0;
}
int SNEPPX_grpc_set_auth_token(SNEPPXGRPCStub* stub, const char* token) {
    (void)stub; (void)token; return 0;
}
const char* SNEPPX_grpc_status_string(SNEPPXGRPCStatus status) {
    switch (status) {
        case SNEPPX_GRPC_OK: return "OK";
        case SNEPPX_GRPC_UNAVAILABLE: return "UNAVAILABLE";
        case SNEPPX_GRPC_DEADLINE_EXCEEDED: return "DEADLINE_EXCEEDED";
        case SNEPPX_GRPC_INTERNAL: return "INTERNAL";
        case SNEPPX_GRPC_UNAUTHENTICATED: return "UNAUTHENTICATED";
        default: return "UNKNOWN";
    }
}
