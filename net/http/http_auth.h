#ifndef SNEPPX_HTTP_AUTH_H
#define SNEPPX_HTTP_AUTH_H

#include "http_server.h"

#ifdef __cplusplus
/*
 * SNEPPX - Http Auth
 *
 * WHAT
 *   Http Auth.
 *
 * CONCEPT
 *   Provides the Http Auth.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* API key auth middleware state */
typedef struct SNEPPX_HttpAuth SNEPPX_HttpAuth;

/* Create an auth middleware that validates Authorization: Bearer <key>
 * against the store at db_path. Returns NULL on failure. */
/**
 * @brief Create Http Auth.
 *
 * @param db_path [in] Db Path value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPX_HttpAuth* SNEPPX_http_auth_create(const char* db_path);

/* Destroy auth state */
/**
 * @brief Destroy Http Auth.
 *
 * @param auth [out] Auth value.
 */
void SNEPPX_http_auth_destroy(SNEPPX_HttpAuth* auth);

/* Get the underlying middleware function pointer.
 * Use with SNEPPX_http_server_add_middleware(). */
/**
 * @brief Perform Http Auth Middleware.
 *
 * @param auth [out] Auth value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPX_http_middleware_fn SNEPPX_http_auth_middleware(SNEPPX_HttpAuth* auth);

/* Skip auth for specific path prefixes (e.g. /v1/health, /v1/auth).
 * Returns 0 on success, -1 if list is full. */
/**
 * @brief Perform Http Auth Add Public Path.
 *
 * @param auth [out] Auth value.
 * @param path_prefix [in] Path Prefix value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_auth_add_public_path(SNEPPX_HttpAuth* auth, const char* path_prefix);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HTTP_AUTH_H */
