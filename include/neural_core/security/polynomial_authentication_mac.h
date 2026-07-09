#ifndef SNEPPX_POLY1305_H
#define SNEPPX_POLY1305_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t r[5];
    uint64_t h[5];
    uint64_t s[2];
    uint8_t buf[16];
    size_t buflen;
} SNEPPXPoly1305State;

void SNEPPX_poly1305_init(SNEPPXPoly1305State* state, const uint8_t key[32]);
void SNEPPX_poly1305_update(SNEPPXPoly1305State* state, const uint8_t* data, size_t len);
void SNEPPX_poly1305_finish(SNEPPXPoly1305State* state, uint8_t mac[16]);

#endif
