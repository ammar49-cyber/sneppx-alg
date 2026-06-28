#include "arix_ct.h"

int arix_ct_equal(const uint8_t* a, const uint8_t* b, size_t len) {
    uint8_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return (int)((diff == 0) ? 1 : 0);
}

uint8_t arix_ct_select(uint8_t mask, uint8_t a, uint8_t b) {
    return (uint8_t)(b ^ (mask & (a ^ b)));
}

void arix_ct_copy(uint8_t* dst, const uint8_t* src, size_t len, uint8_t condition) {
    uint16_t m = condition | ((uint16_t)condition << 8);
    m |= m >> 4; m |= m >> 2; m |= m >> 1;
    uint8_t mask = (uint8_t)(uint8_t)m;
    for (size_t i = 0; i < len; i++) dst[i] = (uint8_t)(dst[i] ^ (mask & (dst[i] ^ src[i])));
}

int arix_ct_is_zero(const uint8_t* data, size_t len) {
    uint8_t acc = 0;
    for (size_t i = 0; i < len; i++) acc |= data[i];
    return (int)((acc == 0) ? 1 : 0);
}

uint32_t arix_ct_compare_32(const uint32_t* a, const uint32_t* b, size_t len) {
    uint32_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return diff;
}
