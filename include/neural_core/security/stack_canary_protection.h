#ifndef SNEPPX_CANARY_H
#define SNEPPX_CANARY_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_CANARY_SIZE 16

typedef struct {
    uint8_t value[SNEPPX_CANARY_SIZE];
    uint64_t generation;
} SNEPPXCanary;

void SNEPPX_canary_generate(SNEPPXCanary* canary);
int SNEPPX_canary_verify(const SNEPPXCanary* expected, const uint8_t* memory);
void SNEPPX_canary_write(const SNEPPXCanary* canary, uint8_t* memory);

#endif
