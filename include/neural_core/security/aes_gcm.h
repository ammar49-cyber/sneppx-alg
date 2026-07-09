#ifndef SNEPPX_AES_GCM_H
#define SNEPPX_AES_GCM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_AES_BLOCK_SIZE 16
#define SNEPPX_AES256_KEY_SIZE 32
#define SNEPPX_GCM_IV_SIZE 12
#define SNEPPX_GCM_TAG_SIZE 16
#define SNEPPX_GCM_MAX_AAD 65536
#define SNEPPX_GCM_MAX_PLAINTEXT (1ULL << 36)

typedef struct {
    uint32_t rk[60];
    int rounds;
    uint8_t h[SNEPPX_AES_BLOCK_SIZE];
    uint8_t j0[SNEPPX_AES_BLOCK_SIZE];
    uint8_t tag[SNEPPX_GCM_TAG_SIZE];
    int mode;
} SNEPPXAESGCM;

void SNEPPX_aes256_key_expansion(const uint8_t key[SNEPPX_AES256_KEY_SIZE], uint32_t rk[60]);
void SNEPPX_aes256_encrypt_block(const uint32_t rk[60], const uint8_t in[SNEPPX_AES_BLOCK_SIZE], uint8_t out[SNEPPX_AES_BLOCK_SIZE]);
void SNEPPX_aes256_decrypt_block(const uint32_t rk[60], const uint8_t in[SNEPPX_AES_BLOCK_SIZE], uint8_t out[SNEPPX_AES_BLOCK_SIZE]);

int  SNEPPX_aes_gcm_init(SNEPPXAESGCM* ctx, const uint8_t key[SNEPPX_AES256_KEY_SIZE], const uint8_t iv[SNEPPX_GCM_IV_SIZE], int encrypt);
void SNEPPX_aes_gcm_update_aad(SNEPPXAESGCM* ctx, const uint8_t* aad, size_t aad_len);
void SNEPPX_aes_gcm_encrypt(SNEPPXAESGCM* ctx, const uint8_t* plaintext, uint8_t* ciphertext, size_t len);
int  SNEPPX_aes_gcm_decrypt(SNEPPXAESGCM* ctx, const uint8_t* ciphertext, uint8_t* plaintext, size_t len);
void SNEPPX_aes_gcm_finalize(SNEPPXAESGCM* ctx, uint8_t tag[SNEPPX_GCM_TAG_SIZE]);
int  SNEPPX_aes_gcm_verify_tag(SNEPPXAESGCM* ctx, const uint8_t expected_tag[SNEPPX_GCM_TAG_SIZE]);

#ifdef __cplusplus
}
#endif
#endif
