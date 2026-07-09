#include "stack_canary_protection.h"
#include "cryptographic_random_generator.h"
#include "constant_time_operations.h"
#include <string.h>

static uint64_t generation_counter = 0;

void SNEPPX_canary_generate(SNEPPXCanary* canary) {
    if (!canary) return;
    SNEPPX_random_bytes(canary->value, SNEPPX_CANARY_SIZE);
    canary->generation = generation_counter++;
}

int SNEPPX_canary_verify(const SNEPPXCanary* expected, const uint8_t* memory) {
    if (!expected || !memory) return 0;
    return SNEPPX_ct_equal(expected->value, memory, SNEPPX_CANARY_SIZE);
}

void SNEPPX_canary_write(const SNEPPXCanary* canary, uint8_t* memory) {
    if (!canary || !memory) return;
    memcpy(memory, canary->value, SNEPPX_CANARY_SIZE);
}
