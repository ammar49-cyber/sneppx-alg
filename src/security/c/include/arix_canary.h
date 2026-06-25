#ifndef ARIX_CANARY_H
#define ARIX_CANARY_H

#include <stddef.h>
#include <stdint.h>

#define ARIX_CANARY_SIZE 16

typedef struct {
    uint8_t value[ARIX_CANARY_SIZE];
    uint64_t generation;
} ArixCanary;

void arix_canary_generate(ArixCanary* canary);
int arix_canary_verify(const ArixCanary* expected, const uint8_t* memory);
void arix_canary_write(const ArixCanary* canary, uint8_t* memory);

#endif
