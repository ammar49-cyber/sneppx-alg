#include "../include/c_binding_interface.h"
#include "authenticated_encryption_module.h"
#include "chacha20_stream_cipher.h"
#include "cryptographic_hashing_blake3.h"
#include "keccak_sha3_hashing.h"
#include "memory_hard_key_derivation.h"
#include "constant_time_operations.h"
#include <string.h>

int arix_c_hash_blake3(const uint8_t* data, size_t len, uint8_t* out, size_t out_len) {
    if (!data || !out || out_len < 32) return -1;
    ArixBlake3State ctx;
    arix_blake3_init(&ctx);
    arix_blake3_update(&ctx, data, len);
    arix_blake3_finalize(&ctx, out, out_len);
    return 0;
}

int arix_c_hash_sha3_256(const uint8_t* data, size_t len, uint8_t out[32]) {
    if (!data || !out) return -1;
    ArixSHA3State ctx;
    arix_sha3_256_init(&ctx);
    arix_sha3_update(&ctx, data, len);
    arix_sha3_finalize(&ctx, out, 32);
    return 0;
}

int arix_c_hash_sha3_512(const uint8_t* data, size_t len, uint8_t out[64]) {
    if (!data || !out) return -1;
    ArixSHA3State ctx;
    arix_sha3_512_init(&ctx);
    arix_sha3_update(&ctx, data, len);
    arix_sha3_finalize(&ctx, out, 64);
    return 0;
}

int arix_c_chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* plaintext, size_t len, uint8_t* ciphertext) {
    if (!key || !nonce || !plaintext || !ciphertext) return -1;
    ArixChaCha20State state;
    arix_chacha20_init(&state, key, nonce, 0);
    memcpy(ciphertext, plaintext, len);
    arix_chacha20_encrypt(&state, ciphertext, len);
    return 0;
}

int arix_c_chacha20_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* ciphertext, size_t len, uint8_t* plaintext) {
    if (!key || !nonce || !ciphertext || !plaintext) return -1;
    ArixChaCha20State state;
    arix_chacha20_init(&state, key, nonce, 0);
    memcpy(plaintext, ciphertext, len);
    arix_chacha20_encrypt(&state, plaintext, len);
    return 0;
}

int arix_c_aead_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* plaintext, size_t plaintext_len,
                         uint8_t* ciphertext, uint8_t tag[16]) {
    if (!key || !nonce || !plaintext || !ciphertext || !tag) return -1;
    return arix_aead_encrypt(ciphertext, tag, plaintext, plaintext_len, aad, aad_len, key, nonce);
}

int arix_c_aead_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* ciphertext, size_t ciphertext_len,
                         const uint8_t tag[16], uint8_t* plaintext) {
    if (!key || !nonce || !ciphertext || !plaintext) return -1;
    return arix_aead_decrypt(plaintext, ciphertext, ciphertext_len, tag, aad, aad_len, key, nonce);
}

int arix_c_argon2_hash(const char* password, size_t pwd_len,
                        const uint8_t* salt, size_t salt_len,
                        uint8_t* out, size_t out_len,
                        uint32_t t_cost, uint32_t m_cost, uint32_t parallelism) {
    if (!password || !salt || !out) return -1;
    ArixArgon2Config cfg;
    cfg.memory_kb = m_cost;
    cfg.iterations = t_cost;
    cfg.parallelism = parallelism;
    cfg.hash_len = out_len;
    return arix_argon2id((const uint8_t*)password, pwd_len, salt, salt_len, &cfg, out);
}

int arix_c_ct_memcmp(const void* a, const void* b, size_t len) {
    if (!a || !b) return -1;
    return arix_ct_equal(a, b, len) ? 0 : 1;
}

void arix_c_ct_memzero(void* ptr, size_t len) {
    if (ptr) {
        volatile unsigned char* p = (volatile unsigned char*)ptr;
        for (size_t i = 0; i < len; i++) p[i] = 0;
    }
}
