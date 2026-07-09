#ifndef SNEPPX_HMAC_H
#define SNEPPX_HMAC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_HMAC_MAX_OUTPUT 64
#define SNEPPX_HMAC_MAX_KEY 128

typedef struct {
    uint8_t key[SNEPPX_HMAC_MAX_KEY];
    size_t key_len;
    int hash_type;
} SNEPPXHMAC;

int  SNEPPX_hmac_init(SNEPPXHMAC* ctx, const uint8_t* key, size_t key_len, int hash_type);
int  SNEPPX_hmac_compute(SNEPPXHMAC* ctx, const uint8_t* data, size_t data_len, uint8_t* out, size_t* out_len);
int  SNEPPX_hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[32]);
int  SNEPPX_hmac_sha512(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[64]);

#ifdef __cplusplus
}
#endif
#endif
