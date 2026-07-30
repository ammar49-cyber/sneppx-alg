#ifndef SNEPPX_HTTP_SERVER_H
#define SNEPPX_HTTP_SERVER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* HTTP server types */
typedef struct SNEPPX_HttpServer SNEPPX_HttpServer;
typedef struct SNEPPX_HttpRequest SNEPPX_HttpRequest;
typedef struct SNEPPX_HttpResponse SNEPPX_HttpResponse;
typedef struct SNEPPX_HttpRoute SNEPPX_HttpRoute;
typedef struct SNEPPX_HttpMiddleware SNEPPX_HttpMiddleware;

/* Request handler: process a request and populate the response.
 * Return 0 on success, non-zero to abort with an error. */
typedef int (*SNEPPX_http_handler_fn)(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata);

/* Middleware function: called before handlers. Return 0 to continue, non-zero to short-circuit. */
typedef int (*SNEPPX_http_middleware_fn)(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata);

/* Server lifecycle */
SNEPPX_HttpServer* SNEPPX_http_server_create(int port, int num_threads);
int SNEPPX_http_server_start(SNEPPX_HttpServer* srv);
void SNEPPX_http_server_stop(SNEPPX_HttpServer* srv);
void SNEPPX_http_server_destroy(SNEPPX_HttpServer* srv);

/* Route registration */
int SNEPPX_http_server_add_route(SNEPPX_HttpServer* srv, const char* method, const char* path,
                                  SNEPPX_http_handler_fn handler, void* userdata);
int SNEPPX_http_server_add_static_dir(SNEPPX_HttpServer* srv, const char* url_prefix, const char* dir_path);

/* Middleware */
int SNEPPX_http_server_add_middleware(SNEPPX_HttpServer* srv, SNEPPX_http_middleware_fn mw, void* userdata);

/* Request accessors */
const char* SNEPPX_http_request_method(SNEPPX_HttpRequest* req);
const char* SNEPPX_http_request_path(SNEPPX_HttpRequest* req);
const char* SNEPPX_http_request_header(SNEPPX_HttpRequest* req, const char* name);
const char* SNEPPX_http_request_body(SNEPPX_HttpRequest* req, size_t* len);
const char* SNEPPX_http_request_query(SNEPPX_HttpRequest* req, const char* key);
void*       SNEPPX_http_request_userdata(SNEPPX_HttpRequest* req);

/* Response setters */
void SNEPPX_http_response_set_status(SNEPPX_HttpResponse* resp, int status_code);
void SNEPPX_http_response_set_header(SNEPPX_HttpResponse* resp, const char* name, const char* value);
void SNEPPX_http_response_set_body(SNEPPX_HttpResponse* resp, const char* data, size_t len);
void SNEPPX_http_response_set_body_str(SNEPPX_HttpResponse* resp, const char* str);
void SNEPPX_http_response_set_json(SNEPPX_HttpResponse* resp, const char* json_str);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HTTP_SERVER_H */
