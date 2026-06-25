#ifndef ARIX_ASLR_H
#define ARIX_ASLR_H

#include <stddef.h>

size_t arix_aslr_random_offset(size_t max_offset);
void arix_aslr_apply(void** base_ptr, size_t* size, size_t max_random);

#endif
