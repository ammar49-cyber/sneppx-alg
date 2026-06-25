#include "arix_power.h"
#include <stdint.h>

static volatile uint64_t dummy_accumulator = 0;

void arix_power_balance_start(void) {
    dummy_accumulator = 0;
}

void arix_power_balance_end(void) {
    dummy_accumulator = 0;
}

void arix_power_dummy_op(void) {
    uint64_t x = 0xDEADBEEFCAFEBABEULL;
    uint64_t y = 0x0123456789ABCDEFULL;
    for (int i = 0; i < 16; i++) {
        x ^= y;
        x *= 0x9E3779B97F4A7C15ULL;
        y += x;
        y = (y << 13) | (y >> 51);
        dummy_accumulator ^= x ^ y;
    }
}
