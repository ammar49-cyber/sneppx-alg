#ifndef ARIX_C_SECURITY_WRAPPER_H
#define ARIX_C_SECURITY_WRAPPER_H
/*
 * C Language Security Bindings — v0.5
 *
 * PURPOSE: Thin C wrapper over the internal security/crypto/c library
 * for integration with systems that require a strict C89/C99 interface.
 * Functions here delegate to the implementations in security/crypto/c/
 * (aead.c, blake3.c, chacha20.c, etc.) with additional input validation.
 *
 * DEPENDENCIES: authenticated_encryption_module.h, chacha20_stream_cipher.h, cryptographic_hashing_blake3.h, keccak_sha3_hashing.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Hashing ---------- */
int arix_c_hash_blake3(const uint8_t* data, size_t len, uint8_t* out, size_t out_len);
int arix_c_hash_sha3_256(const uint8_t* data, size_t len, uint8_t out[32]);
int arix_c_hash_sha3_512(const uint8_t* data, size_t len, uint8_t out[64]);

/* ---------- Symmetric encryption ---------- */
int arix_c_chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* plaintext, size_t len, uint8_t* ciphertext);
int arix_c_chacha20_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* ciphertext, size_t len, uint8_t* plaintext);
int arix_c_aead_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* plaintext, size_t plaintext_len,
                         uint8_t* ciphertext, uint8_t tag[16]);
int arix_c_aead_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* ciphertext, size_t ciphertext_len,
                         const uint8_t tag[16], uint8_t* plaintext);

/* ---------- Key derivation ---------- */
int arix_c_argon2_hash(const char* password, size_t pwd_len,
                        const uint8_t* salt, size_t salt_len,
                        uint8_t* out, size_t out_len,
                        uint32_t t_cost, uint32_t m_cost, uint32_t parallelism);

/* ---------- Constant-time utilities ---------- */
int  arix_c_ct_memcmp(const void* a, const void* b, size_t len);
void arix_c_ct_memzero(void* ptr, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_C_SECURITY_WRAPPER_H */
