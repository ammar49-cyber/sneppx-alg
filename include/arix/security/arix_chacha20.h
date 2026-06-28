#ifndef ARIX_CHACHA20_H
#define ARIX_CHACHA20_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[16];
} ArixChaCha20State;

void arix_chacha20_init(ArixChaCha20State* state, const uint8_t key[32], const uint8_t nonce[12], uint32_t counter);
void arix_chacha20_block(ArixChaCha20State* state, uint8_t output[64]);
void arix_chacha20_encrypt(ArixChaCha20State* state, uint8_t* data, size_t len);

#endif
