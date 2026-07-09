#ifndef SNEPPX_BLAKE3_H
#define SNEPPX_BLAKE3_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_BLAKE3_OUT_LEN 32
#define SNEPPX_BLAKE3_BLOCK_LEN 64
#define SNEPPX_BLAKE3_CHUNK_LEN 1024

typedef struct {
    uint32_t key[8];
    uint64_t counter;
    uint8_t buf[SNEPPX_BLAKE3_CHUNK_LEN];
    size_t buflen;
    uint8_t flags;
} SNEPPXBlake3State;

void SNEPPX_blake3_init(SNEPPXBlake3State* state);
void SNEPPX_blake3_update(SNEPPXBlake3State* state, const uint8_t* data, size_t len);
void SNEPPX_blake3_finish(SNEPPXBlake3State* state, uint8_t* hash);

#endif
