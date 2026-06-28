#ifndef ARIX_CACHE_H
#define ARIX_CACHE_H

#include <stddef.h>

void arix_cache_flush(const void* ptr, size_t len);
void arix_cache_prefetch(const void* ptr);
void arix_cache_barrier(void);

#endif
