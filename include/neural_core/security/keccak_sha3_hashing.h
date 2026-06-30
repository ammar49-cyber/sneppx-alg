#ifndef ARIX_SHA3_H
#define ARIX_SHA3_H

#include <stddef.h>
#include <stdint.h>

#define ARIX_SHA3_256_DIGEST_SIZE 32
#define ARIX_SHA3_512_DIGEST_SIZE 64
#define ARIX_SHA3_STATE_SIZE 200

typedef struct {
    uint64_t state[25];
    unsigned int rate;
    unsigned int capacity;
    unsigned int buflen;
    uint8_t buffer[ARIX_SHA3_STATE_SIZE];
    unsigned int digest_size;
} ArixSHA3State;

void arix_sha3_256_init(ArixSHA3State* state);
void arix_sha3_512_init(ArixSHA3State* state);
void arix_sha3_update(ArixSHA3State* state, const uint8_t* data, size_t len);
void arix_sha3_finish(ArixSHA3State* state, uint8_t* hash);

#endif
