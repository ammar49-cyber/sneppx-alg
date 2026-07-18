#ifndef SNEPPX_HTTP_TRANSPORT_H
#define SNEPPX_HTTP_TRANSPORT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* HTTP inference transport. When SNEPPX_BUILD_HTTP is defined this provides a
 * real, dependency-free client (GET/POST) and a minimal blocking server built
 * on BSD sockets. Without the flag every entry point reports
 * SNEPPX_DRIVER_UNSUPPORTED. */

int  SNEPPX_http_init(void);
void SNEPPX_http_shutdown(void);

/* Perform an HTTP GET against host:port/path. Response (status line + headers
 * + body) is written to `out` (at most out_max bytes). Returns 0 on success. */
int SNEPPX_http_get(const char* host, int port, const char* path,
                    char* out, size_t out_max);

/* Perform an HTTP POST with `body` (content-type application/json). */
int SNEPPX_http_post(const char* host, int port, const char* path,
                     const char* body, char* out, size_t out_max);

/* A request handler receives the method, path and body and must write a
 * response (status line + headers + body) into `resp` (resp_max bytes). */
typedef void (*SNEPPX_http_handler)(const char* method, const char* path,
                                    const char* body, char* resp, size_t resp_max);

/* Create and run a blocking server on `port`, invoking `handler` per request.
 * This call blocks until SNEPPX_http_server_stop(server) is called from
 * another thread. */
void* SNEPPX_http_server_create(int port, SNEPPX_http_handler handler);
int   SNEPPX_http_server_run(void* server);
void  SNEPPX_http_server_stop(void* server);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HTTP_TRANSPORT_H */
