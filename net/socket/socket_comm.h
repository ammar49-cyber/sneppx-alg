#ifndef SNEPPX_SOCKET_COMM_H
#define SNEPPX_SOCKET_COMM_H
/*
 * Socket Communication — v1.0 (distributed training transport)
 *
 * PURPOSE: Low-level TCP socket abstraction for point-to-point tensor
 * transfers between training nodes.  Provides blocking and non-blocking
 * send/recv with length-prefixed message framing.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_SOCKET_TCP,
    SNEPPX_SOCKET_UNIX,
} SNEPPXSocketType;

typedef struct {
    int            fd;
    SNEPPXSocketType type;
    int            port;
    char           hostname[256];
    size_t         send_buf_size;
    size_t         recv_buf_size;
    int            is_connected;
    int            is_listening;
    uint64_t       bytes_sent;
    uint64_t       bytes_recv;
    int            timeout_ms;
} SNEPPXSocket;

/* ---------- Lifecycle ---------- */
SNEPPXSocket* SNEPPX_socket_create(SNEPPXSocketType type);
void        SNEPPX_socket_destroy(SNEPPXSocket* sock);

/* ---------- Server ---------- */
int SNEPPX_socket_bind(SNEPPXSocket* sock, int port);
int SNEPPX_socket_listen(SNEPPXSocket* sock, int backlog);
SNEPPXSocket* SNEPPX_socket_accept(SNEPPXSocket* server_sock);

/* ---------- Client ---------- */
int SNEPPX_socket_connect(SNEPPXSocket* sock, const char* host, int port);

/* ---------- I/O ---------- */
int  SNEPPX_socket_send(SNEPPXSocket* sock, const void* data, size_t len);
int  SNEPPX_socket_recv(SNEPPXSocket* sock, void* buf, size_t len);
int  SNEPPX_socket_send_message(SNEPPXSocket* sock, const void* data, size_t len);
int  SNEPPX_socket_recv_message(SNEPPXSocket* sock, void** buf, size_t* len);
void SNEPPX_socket_close(SNEPPXSocket* sock);

/* ---------- Tensor helpers (v1.0) ---------- */
int SNEPPX_socket_send_tensor(SNEPPXSocket* sock, const void* tensor_handle);
int SNEPPX_socket_recv_tensor(SNEPPXSocket* sock, void** tensor_handle);

/* ---------- Utility ---------- */
int         SNEPPX_socket_set_timeout(SNEPPXSocket* sock, int ms);
const char* SNEPPX_socket_error_string(int err);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_SOCKET_COMM_H */
