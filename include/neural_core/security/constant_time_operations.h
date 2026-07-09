#ifndef SNEPPX_CT_H
#define SNEPPX_CT_H

#include <stddef.h>
#include <stdint.h>

int SNEPPX_ct_equal(const uint8_t* a, const uint8_t* b, size_t len);
uint8_t SNEPPX_ct_select(uint8_t mask, uint8_t a, uint8_t b);
void SNEPPX_ct_copy(uint8_t* dst, const uint8_t* src, size_t len, uint8_t condition);
int SNEPPX_ct_is_zero(const uint8_t* data, size_t len);
uint32_t SNEPPX_ct_compare_32(const uint32_t* a, const uint32_t* b, size_t len);

#endif
