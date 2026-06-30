/*
 * C Security Wrapper Implementation — SKELETON
 * VERSION: v0.5
 */

#include "c_binding_interface.h"
#include <string.h>

int arix_c_hash_blake3(const uint8_t* data, size_t len, uint8_t* out, size_t out_len) {
    (void)data; (void)len; (void)out; (void)out_len; return 0;
}
int arix_c_hash_sha3_256(const uint8_t* data, size_t len, uint8_t out[32]) {
    (void)data; (void)len; if (out) memset(out, 0, 32); return 0;
}
int arix_c_hash_sha3_512(const uint8_t* data, size_t len, uint8_t out[64]) {
    (void)data; (void)len; if (out) memset(out, 0, 64); return 0;
}
int arix_c_chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* plaintext, size_t len, uint8_t* ciphertext) {
    (void)key; (void)nonce; (void)plaintext; (void)len; (void)ciphertext; return 0;
}
int arix_c_chacha20_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* ciphertext, size_t len, uint8_t* plaintext) {
    (void)key; (void)nonce; (void)ciphertext; (void)len; (void)plaintext; return 0;
}
int arix_c_aead_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* plaintext, size_t plaintext_len,
                         uint8_t* ciphertext, uint8_t tag[16]) {
    (void)key; (void)nonce; (void)aad; (void)aad_len; (void)plaintext; (void)plaintext_len;
    (void)ciphertext; (void)tag; return 0;
}
int arix_c_aead_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* ciphertext, size_t ciphertext_len,
                         const uint8_t tag[16], uint8_t* plaintext) {
    (void)key; (void)nonce; (void)aad; (void)aad_len; (void)ciphertext; (void)ciphertext_len;
    (void)tag; (void)plaintext; return 0;
}
int arix_c_argon2_hash(const char* password, size_t pwd_len,
                        const uint8_t* salt, size_t salt_len,
                        uint8_t* out, size_t out_len,
                        uint32_t t_cost, uint32_t m_cost, uint32_t parallelism) {
    (void)password; (void)pwd_len; (void)salt; (void)salt_len;
    (void)out; (void)out_len; (void)t_cost; (void)m_cost; (void)parallelism; return 0;
}
int arix_c_ct_memcmp(const void* a, const void* b, size_t len) {
    (void)a; (void)b; (void)len; return 0;
}
void arix_c_ct_memzero(void* ptr, size_t len) {
    if (ptr) memset(ptr, 0, len);
}
