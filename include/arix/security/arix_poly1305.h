#ifndef ARIX_POLY1305_H
#define ARIX_POLY1305_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t r[5];
    uint64_t h[5];
    uint64_t s[2];
    uint8_t buf[16];
    size_t buflen;
} ArixPoly1305State;

void arix_poly1305_init(ArixPoly1305State* state, const uint8_t key[32]);
void arix_poly1305_update(ArixPoly1305State* state, const uint8_t* data, size_t len);
void arix_poly1305_finish(ArixPoly1305State* state, uint8_t mac[16]);

#endif
