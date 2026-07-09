#ifndef SNEPPX_ASLR_H
#define SNEPPX_ASLR_H

#include <stddef.h>

size_t SNEPPX_aslr_random_offset(size_t max_offset);
void SNEPPX_aslr_apply(void** base_ptr, size_t* size, size_t max_random);

#endif
