#ifndef SNEPPX_SC_H
#define SNEPPX_SC_H

#include <stddef.h>
#include <stdint.h>

uint32_t SNEPPX_sc_select_u32(uint32_t condition, uint32_t a, uint32_t b);
uint64_t SNEPPX_sc_select_u64(uint64_t condition, uint64_t a, uint64_t b);
uint32_t SNEPPX_sc_equal_u32(uint32_t a, uint32_t b);
uint32_t SNEPPX_sc_lt_u32(uint32_t a, uint32_t b);
uint32_t SNEPPX_sc_is_zero_u32(uint32_t a);
void SNEPPX_sc_cond_copy(void* dst, const void* src, size_t len, uint32_t condition);

#endif
