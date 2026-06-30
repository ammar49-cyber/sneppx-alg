#include "stack_canary_protection.h"
#include "cryptographic_random_generator.h"
#include "constant_time_operations.h"
#include <string.h>

static uint64_t generation_counter = 0;

void arix_canary_generate(ArixCanary* canary) {
    if (!canary) return;
    arix_random_bytes(canary->value, ARIX_CANARY_SIZE);
    canary->generation = generation_counter++;
}

int arix_canary_verify(const ArixCanary* expected, const uint8_t* memory) {
    if (!expected || !memory) return 0;
    return arix_ct_equal(expected->value, memory, ARIX_CANARY_SIZE);
}

void arix_canary_write(const ArixCanary* canary, uint8_t* memory) {
    if (!canary || !memory) return;
    memcpy(memory, canary->value, ARIX_CANARY_SIZE);
}
