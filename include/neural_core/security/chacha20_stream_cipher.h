#ifndef SNEPPX_CHACHA20_H
#define SNEPPX_CHACHA20_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[16];
} SNEPPXChaCha20State;

void SNEPPX_chacha20_init(SNEPPXChaCha20State* state, const uint8_t key[32], const uint8_t nonce[12], uint32_t counter);
void SNEPPX_chacha20_block(SNEPPXChaCha20State* state, uint8_t output[64]);
void SNEPPX_chacha20_encrypt(SNEPPXChaCha20State* state, uint8_t* data, size_t len);

#endif
