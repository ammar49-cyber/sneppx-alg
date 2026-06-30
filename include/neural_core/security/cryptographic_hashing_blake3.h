#ifndef ARIX_BLAKE3_H
#define ARIX_BLAKE3_H

#include <stddef.h>
#include <stdint.h>

#define ARIX_BLAKE3_OUT_LEN 32
#define ARIX_BLAKE3_BLOCK_LEN 64
#define ARIX_BLAKE3_CHUNK_LEN 1024

typedef struct {
    uint32_t key[8];
    uint64_t counter;
    uint8_t buf[ARIX_BLAKE3_CHUNK_LEN];
    size_t buflen;
    uint8_t flags;
} ArixBlake3State;

void arix_blake3_init(ArixBlake3State* state);
void arix_blake3_update(ArixBlake3State* state, const uint8_t* data, size_t len);
void arix_blake3_finish(ArixBlake3State* state, uint8_t* hash);

#endif
