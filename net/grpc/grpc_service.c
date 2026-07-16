#include "grpc_service.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <unistd.h>
#endif

int SNEPPX_grpc_server_start(SNEPPXGRPCServer** server, int port) {
    if (!server) return -1;
    *server = (SNEPPXGRPCServer*)calloc(1, sizeof(SNEPPXGRPCServer));
    if (!*server) return -1;
    (*server)->port = port;
    (*server)->is_running = 1;
    (*server)->server = NULL;
    (*server)->coordination_svc = NULL;
    (*server)->transfer_svc = NULL;
    return 0;
}

void SNEPPX_grpc_server_stop(SNEPPXGRPCServer* server) {
    if (!server) return;
    server->is_running = 0;
    free(server);
}

void SNEPPX_grpc_server_wait(SNEPPXGRPCServer* server) {
    (void)server;
}

SNEPPXGRPCStub* SNEPPX_grpc_stub_create(const char* target) {
    if (!target) return NULL;
    SNEPPXGRPCStub* stub = (SNEPPXGRPCStub*)calloc(1, sizeof(SNEPPXGRPCStub));
    if (!stub) return NULL;
    snprintf(stub->target, sizeof(stub->target), "%s", target);
    stub->channel = NULL;
    stub->coordination_stub = NULL;
    stub->transfer_stub = NULL;
    stub->connected = 1;
    return stub;
}

void SNEPPX_grpc_stub_destroy(SNEPPXGRPCStub* stub) { free(stub); }

int SNEPPX_grpc_register_node(SNEPPXGRPCStub* stub, const SNEPPXGRPCNodeInfo* info) {
    (void)stub; (void)info; return 0;
}

int SNEPPX_grpc_get_world_size(SNEPPXGRPCStub* stub, int* size) {
    if (!stub || !size) return -1;
    *size = 1;
    return 0;
}

int SNEPPX_grpc_get_rank(SNEPPXGRPCStub* stub, int* rank) {
    if (!stub || !rank) return -1;
    *rank = 0;
    return 0;
}

int SNEPPX_grpc_barrier(SNEPPXGRPCStub* stub) {
    if (!stub) return -1;
    return 0;
}

int SNEPPX_grpc_all_gather(SNEPPXGRPCStub* stub, const void* send_buf, void** recv_buf, size_t elem_size) {
    if (!stub || !recv_buf) return -1;
    (void)elem_size;
    if (send_buf && elem_size > 0) {
        *recv_buf = malloc(elem_size);
        if (*recv_buf) memcpy(*recv_buf, send_buf, elem_size);
    } else {
        *recv_buf = NULL;
    }
    return 0;
}

int SNEPPX_grpc_send_tensor(SNEPPXGRPCStub* stub, const void* tensor, int dest_rank) {
    (void)stub; (void)tensor; (void)dest_rank; return 0;
}

int SNEPPX_grpc_recv_tensor(SNEPPXGRPCStub* stub, void** tensor, int src_rank) {
    (void)stub; (void)src_rank;
    if (tensor) *tensor = NULL;
    return 0;
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
