#ifndef ARIX_SOCKET_COMM_H
#define ARIX_SOCKET_COMM_H
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
    ARIX_SOCKET_TCP,
    ARIX_SOCKET_UNIX,
} ArixSocketType;

typedef struct {
    int            fd;
    ArixSocketType type;
    int            port;
    char           hostname[256];
    size_t         send_buf_size;
    size_t         recv_buf_size;
    int            is_connected;
    int            is_listening;
    uint64_t       bytes_sent;
    uint64_t       bytes_recv;
    int            timeout_ms;
} ArixSocket;

/* ---------- Lifecycle ---------- */
ArixSocket* arix_socket_create(ArixSocketType type);
void        arix_socket_destroy(ArixSocket* sock);

/* ---------- Server ---------- */
int arix_socket_bind(ArixSocket* sock, int port);
int arix_socket_listen(ArixSocket* sock, int backlog);
ArixSocket* arix_socket_accept(ArixSocket* server_sock);

/* ---------- Client ---------- */
int arix_socket_connect(ArixSocket* sock, const char* host, int port);

/* ---------- I/O ---------- */
int  arix_socket_send(ArixSocket* sock, const void* data, size_t len);
int  arix_socket_recv(ArixSocket* sock, void* buf, size_t len);
int  arix_socket_send_message(ArixSocket* sock, const void* data, size_t len);
int  arix_socket_recv_message(ArixSocket* sock, void** buf, size_t* len);
void arix_socket_close(ArixSocket* sock);

/* ---------- Tensor helpers (v1.0) ---------- */
int arix_socket_send_tensor(ArixSocket* sock, const void* tensor_handle);
int arix_socket_recv_tensor(ArixSocket* sock, void** tensor_handle);

/* ---------- Utility ---------- */
int         arix_socket_set_timeout(ArixSocket* sock, int ms);
const char* arix_socket_error_string(int err);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_SOCKET_COMM_H */
