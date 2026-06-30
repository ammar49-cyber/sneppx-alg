#ifndef ARIX_SC_H
#define ARIX_SC_H

#include <stddef.h>
#include <stdint.h>

uint32_t arix_sc_select_u32(uint32_t condition, uint32_t a, uint32_t b);
uint64_t arix_sc_select_u64(uint64_t condition, uint64_t a, uint64_t b);
uint32_t arix_sc_equal_u32(uint32_t a, uint32_t b);
uint32_t arix_sc_lt_u32(uint32_t a, uint32_t b);
uint32_t arix_sc_is_zero_u32(uint32_t a);
void arix_sc_cond_copy(void* dst, const void* src, size_t len, uint32_t condition);

#endif
