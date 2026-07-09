#ifndef SNEPPX_PBKDF2_H
#define SNEPPX_PBKDF2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int SNEPPX_pbkdf2_hmac_sha256(const uint8_t* password, size_t pwd_len, const uint8_t* salt, size_t salt_len, uint32_t iterations, uint8_t* out, size_t out_len);
int SNEPPX_pbkdf2_hmac_sha512(const uint8_t* password, size_t pwd_len, const uint8_t* salt, size_t salt_len, uint32_t iterations, uint8_t* out, size_t out_len);

#ifdef __cplusplus
}
#endif
#endif
