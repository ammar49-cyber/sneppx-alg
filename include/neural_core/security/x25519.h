#ifndef SNEPPX_X25519_H
#define SNEPPX_X25519_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_X25519_KEY_SIZE 32
#define SNEPPX_X25519_SHARED_SIZE 32

void SNEPPX_x25519_clamp(uint8_t scalar[SNEPPX_X25519_KEY_SIZE]);
void SNEPPX_x25519_scalar_mult(uint8_t out[SNEPPX_X25519_KEY_SIZE], const uint8_t scalar[SNEPPX_X25519_KEY_SIZE], const uint8_t point[SNEPPX_X25519_KEY_SIZE]);
void SNEPPX_x25519_keygen(uint8_t public_key[SNEPPX_X25519_KEY_SIZE], uint8_t secret_key[SNEPPX_X25519_KEY_SIZE]);
int  SNEPPX_x25519_shared_secret(uint8_t shared[SNEPPX_X25519_SHARED_SIZE], const uint8_t secret_key[SNEPPX_X25519_KEY_SIZE], const uint8_t public_key[SNEPPX_X25519_KEY_SIZE]);

void SNEPPX_curve25519_basepoint(uint8_t out[SNEPPX_X25519_KEY_SIZE]);
int  SNEPPX_x25519_scalar_valid(const uint8_t scalar[SNEPPX_X25519_KEY_SIZE]);

#ifdef __cplusplus
}
#endif
#endif
