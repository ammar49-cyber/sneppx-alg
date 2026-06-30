#ifndef ARIX_AEAD_H
#define ARIX_AEAD_H

#include <stddef.h>
#include <stdint.h>

int arix_aead_encrypt(uint8_t* ciphertext, uint8_t tag[16], const uint8_t* plaintext, size_t len,
                      const uint8_t* aad, size_t aad_len, const uint8_t key[32], const uint8_t nonce[12]);

int arix_aead_decrypt(uint8_t* plaintext, const uint8_t* ciphertext, size_t len,
                      const uint8_t tag[16], const uint8_t* aad, size_t aad_len,
                      const uint8_t key[32], const uint8_t nonce[12]);

#endif
