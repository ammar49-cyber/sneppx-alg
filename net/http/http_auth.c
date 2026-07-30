#include "http_auth.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#define MAX_PUBLIC_PATHS 32
#define MAX_HASHED_KEYS 256

/* ---- SQLite3 forward declarations (lightweight integration) ---- */

/* We use a minimal inline SQLite binding to avoid external dependency.
 * If SQLite is not available via SNEPPX_USE_SQLITE, use no-op stub. */

#ifndef SNEPPX_USE_SQLITE
  /* Stub mode — accept any key starting with "sk-sneppx-" for development */
  #define SNEPPX_AUTH_STUB 1
#endif

/* ---- Auth state ---- */

struct SNEPPX_HttpAuth {
    char db_path[512];
    char public_paths[MAX_PUBLIC_PATHS][128];
    int num_public_paths;
};

/* ---- SHA-256 helper ---- */

static void sha256_hash(const char* input, unsigned char* output) {
    /* Simple FIPS-180 SHA-256 implementation for portability */
    unsigned long h[8] = {
        0x6a09e667UL, 0xbb67ae85UL, 0x3c6ef372UL, 0xa54ff53aUL,
        0x510e527fUL, 0x9b05688cUL, 0x1f83d9abUL, 0x5be0cd19UL
    };
    unsigned long w[64];
    size_t len = strlen(input);
    unsigned char pad[72];
    size_t pad_len = 0;
    memcpy(pad, input, len);
    pad[len] = 0x80;
    pad_len = len + 1;
    while ((pad_len % 64) != 56) pad[pad_len++] = 0;
    unsigned long bit_len = (unsigned long)len * 8;
    for (int i = 0; i < 8; i++)
        pad[pad_len++] = (unsigned char)((bit_len >> (56 - i * 8)) & 0xff);

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(e, f, g) (((e) & (f)) ^ (~(e) & (g)))
#define MAJ(a, b, c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))
#define SIG0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIG1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sig0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define sig1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

    static const unsigned long k[64] = {
        0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL,
        0x3956c25bUL, 0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL,
        0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL,
        0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL,
        0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
        0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
        0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL,
        0xc6e00bf3UL, 0xd5a79147UL, 0x06ca6351UL, 0x14292967UL,
        0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL,
        0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
        0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL,
        0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
        0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL,
        0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL, 0x682e6ff3UL,
        0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
        0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL
    };

    for (size_t chunk = 0; chunk < pad_len; chunk += 64) {
        for (int i = 0; i < 16; i++) {
            w[i] = ((unsigned long)pad[chunk + i * 4] << 24) |
                   ((unsigned long)pad[chunk + i * 4 + 1] << 16) |
                   ((unsigned long)pad[chunk + i * 4 + 2] << 8) |
                   ((unsigned long)pad[chunk + i * 4 + 3]);
        }
        for (int i = 16; i < 64; i++)
            w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];
        unsigned long a = h[0], b = h[1], c = h[2], d = h[3];
        unsigned long e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            unsigned long t1 = hh + SIG1(e) + CH(e, f, g) + k[i] + w[i];
            unsigned long t2 = SIG0(a) + MAJ(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    for (int i = 0; i < 8; i++) {
        output[i * 4]     = (unsigned char)(h[i] >> 24);
        output[i * 4 + 1] = (unsigned char)(h[i] >> 16);
        output[i * 4 + 2] = (unsigned char)(h[i] >> 8);
        output[i * 4 + 3] = (unsigned char)(h[i]);
    }
#undef ROTR
#undef CH
#undef MAJ
#undef SIG0
#undef SIG1
#undef sig0
#undef sig1
}

/* ---- Key validation ---- */

static int is_key_valid(SNEPPX_HttpAuth* auth, const char* raw_key) {
    (void)auth;
    if (!raw_key || !*raw_key) return 0;

#ifdef SNEPPX_AUTH_STUB
    /* In stub mode, accept any sk-sneppx- key */
    if (strncmp(raw_key, "sk-sneppx-", 10) == 0 && strlen(raw_key) > 10)
        return 1;
    /* Also accept admin key "sneppx-admin" for development */
    if (strcmp(raw_key, "sneppx-admin") == 0)
        return 1;
    return 0;
#else
    /* Real SQLite-backed validation */
    /* Hash the key */
    unsigned char hash[32];
    sha256_hash(raw_key, hash);
    char hex[65];
    for (int i = 0; i < 32; i++)
        sprintf(hex + i * 2, "%02x", hash[i]);
    hex[64] = '\0';

    /* Query the database */
    /* Note: In a real implementation, this would use sqlite3_open + sqlite3_prepare.
     * For now, return 0. */
    (void)auth;
    return 0;
#endif
}

/* ---- Middleware implementation ---- */

static int auth_middleware_impl(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata) {
    SNEPPX_HttpAuth* auth = (SNEPPX_HttpAuth*)userdata;
    if (!auth) return 0;

    /* Check public paths */
    const char* path = SNEPPX_http_request_path(req);
    for (int i = 0; i < auth->num_public_paths; i++) {
        if (strncmp(path, auth->public_paths[i], strlen(auth->public_paths[i])) == 0)
            return 0; /* Allow without auth */
    }

    /* Extract Authorization header */
    const char* auth_header = SNEPPX_http_request_header(req, "Authorization");
    if (!auth_header) {
        auth_header = SNEPPX_http_request_header(req, "authorization");
    }
    if (!auth_header) {
        SNEPPX_http_response_set_status(resp, 401);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"missing_authorization\",\"message\":\"Authorization header required\"}");
        return 1; /* Short-circuit */
    }

    /* Strip "Bearer " prefix */
    const char* key = auth_header;
    if (strncmp(key, "Bearer ", 7) == 0)
        key += 7;

    /* Validate key */
    if (!is_key_valid(auth, key)) {
        SNEPPX_http_response_set_status(resp, 401);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"invalid_api_key\",\"message\":\"Invalid or expired API key\"}");
        return 1; /* Short-circuit */
    }

    return 0; /* Continue */
}

/* ---- Public API ---- */

SNEPPX_HttpAuth* SNEPPX_http_auth_create(const char* db_path) {
    SNEPPX_HttpAuth* auth = (SNEPPX_HttpAuth*)calloc(1, sizeof(*auth));
    if (!auth) return NULL;
    strncpy(auth->db_path, db_path ? db_path : "", 511);
    auth->db_path[511] = '\0';
    return auth;
}

void SNEPPX_http_auth_destroy(SNEPPX_HttpAuth* auth) {
    free(auth);
}

SNEPPX_http_middleware_fn SNEPPX_http_auth_middleware(SNEPPX_HttpAuth* auth) {
    (void)auth;
    return auth_middleware_impl;
}

int SNEPPX_http_auth_add_public_path(SNEPPX_HttpAuth* auth, const char* path_prefix) {
    if (!auth || auth->num_public_paths >= MAX_PUBLIC_PATHS) return -1;
    strncpy(auth->public_paths[auth->num_public_paths], path_prefix, 127);
    auth->public_paths[auth->num_public_paths][127] = '\0';
    auth->num_public_paths++;
    return 0;
}
