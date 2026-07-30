#ifndef SNEPPX_HTTP_AUTH_H
#define SNEPPX_HTTP_AUTH_H

#include "http_server.h"

#ifdef __cplusplus
extern "C" {
#endif

/* API key auth middleware state */
typedef struct SNEPPX_HttpAuth SNEPPX_HttpAuth;

/* Create an auth middleware that validates Authorization: Bearer <key>
 * against the store at db_path. Returns NULL on failure. */
SNEPPX_HttpAuth* SNEPPX_http_auth_create(const char* db_path);

/* Destroy auth state */
void SNEPPX_http_auth_destroy(SNEPPX_HttpAuth* auth);

/* Get the underlying middleware function pointer.
 * Use with SNEPPX_http_server_add_middleware(). */
SNEPPX_http_middleware_fn SNEPPX_http_auth_middleware(SNEPPX_HttpAuth* auth);

/* Skip auth for specific path prefixes (e.g. /v1/health, /v1/auth).
 * Returns 0 on success, -1 if list is full. */
int SNEPPX_http_auth_add_public_path(SNEPPX_HttpAuth* auth, const char* path_prefix);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HTTP_AUTH_H */
