#ifndef SNEPPX_SHA3_H
#define SNEPPX_SHA3_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_SHA3_256_DIGEST_SIZE 32
#define SNEPPX_SHA3_512_DIGEST_SIZE 64
#define SNEPPX_SHA3_STATE_SIZE 200

typedef struct {
    uint64_t state[25];
    unsigned int rate;
    unsigned int capacity;
    unsigned int buflen;
    uint8_t buffer[SNEPPX_SHA3_STATE_SIZE];
    unsigned int digest_size;
} SNEPPXSHA3State;

void SNEPPX_sha3_256_init(SNEPPXSHA3State* state);
void SNEPPX_sha3_512_init(SNEPPXSHA3State* state);
void SNEPPX_sha3_update(SNEPPXSHA3State* state, const uint8_t* data, size_t len);
void SNEPPX_sha3_finish(SNEPPXSHA3State* state, uint8_t* hash);

#endif
