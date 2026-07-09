/*
 * Socket Communication Implementation — SKELETON
 * VERSION: v1.0
 */

#include "socket_comm.h"
#include <stdlib.h>
#include <string.h>

SNEPPXSocket* SNEPPX_socket_create(SNEPPXSocketType type) {
    SNEPPXSocket* sock = (SNEPPXSocket*)calloc(1, sizeof(SNEPPXSocket));
    if (sock) { sock->type = type; sock->fd = -1; }
    return sock;
}

void SNEPPX_socket_destroy(SNEPPXSocket* sock) { free(sock); }

int SNEPPX_socket_bind(SNEPPXSocket* sock, int port) {
    (void)sock; (void)port; return 0;
}
int SNEPPX_socket_listen(SNEPPXSocket* sock, int backlog) {
    (void)sock; (void)backlog; return 0;
}
SNEPPXSocket* SNEPPX_socket_accept(SNEPPXSocket* server_sock) {
    (void)server_sock; return NULL;
}
int SNEPPX_socket_connect(SNEPPXSocket* sock, const char* host, int port) {
    (void)sock; (void)host; (void)port; return 0;
}
int SNEPPX_socket_send(SNEPPXSocket* sock, const void* data, size_t len) {
    (void)sock; (void)data; (void)len; return 0;
}
int SNEPPX_socket_recv(SNEPPXSocket* sock, void* buf, size_t len) {
    (void)sock; (void)buf; (void)len; return 0;
}
int SNEPPX_socket_send_message(SNEPPXSocket* sock, const void* data, size_t len) {
    (void)sock; (void)data; (void)len; return 0;
}
int SNEPPX_socket_recv_message(SNEPPXSocket* sock, void** buf, size_t* len) {
    (void)sock; if (buf) *buf = NULL; if (len) *len = 0; return 0;
}
void SNEPPX_socket_close(SNEPPXSocket* sock) { (void)sock; }
int SNEPPX_socket_send_tensor(SNEPPXSocket* sock, const void* tensor_handle) {
    (void)sock; (void)tensor_handle; return 0;
}
int SNEPPX_socket_recv_tensor(SNEPPXSocket* sock, void** tensor_handle) {
    (void)sock; if (tensor_handle) *tensor_handle = NULL; return 0;
}
int SNEPPX_socket_set_timeout(SNEPPXSocket* sock, int ms) {
    (void)sock; (void)ms; return 0;
}
const char* SNEPPX_socket_error_string(int err) {
    (void)err; return "socket skeleton";
}
