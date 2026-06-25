#include "arix_sc.h"

uint32_t arix_sc_select_u32(uint32_t condition, uint32_t a, uint32_t b) {
    uint32_t mask = (uint32_t)((int32_t)condition >> 31);
    return (a & mask) | (b & ~mask);
}

uint64_t arix_sc_select_u64(uint64_t condition, uint64_t a, uint64_t b) {
    uint64_t mask = (uint64_t)((int64_t)condition >> 63);
    return (a & mask) | (b & ~mask);
}

uint32_t arix_sc_equal_u32(uint32_t a, uint32_t b) {
    uint32_t diff = a ^ b;
    diff |= diff >> 16;
    diff |= diff >> 8;
    diff |= diff >> 4;
    diff |= diff >> 2;
    diff |= diff >> 1;
    return (uint32_t)((diff & 1U) - 1U);
}

uint32_t arix_sc_lt_u32(uint32_t a, uint32_t b) {
    uint64_t diff = (uint64_t)a - (uint64_t)b;
    return (uint32_t)(diff >> 32);
}

uint32_t arix_sc_is_zero_u32(uint32_t a) {
    uint32_t t = a;
    t |= t >> 16;
    t |= t >> 8;
    t |= t >> 4;
    t |= t >> 2;
    t |= t >> 1;
    return (uint32_t)((t & 1U) - 1U);
}

void arix_sc_cond_copy(void* dst, const void* src, size_t len, uint32_t condition) {
    uint32_t mask = (uint32_t)((int32_t)condition >> 31);
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0; i < len; i++) {
        d[i] = (uint8_t)((s[i] & mask) | (d[i] & ~mask));
    }
}
