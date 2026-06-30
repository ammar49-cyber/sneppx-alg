#ifndef ARIX_RANDOM_H
#define ARIX_RANDOM_H

#include <stddef.h>
#include <stdint.h>

int arix_random_bytes(uint8_t* buffer, size_t len);
uint32_t arix_random_uint32(void);
uint32_t arix_random_uniform(uint32_t upper_bound);

#endif
