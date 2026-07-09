#ifndef SNEPPX_CACHE_H
#define SNEPPX_CACHE_H

#include <stddef.h>

void SNEPPX_cache_flush(const void* ptr, size_t len);
void SNEPPX_cache_prefetch(const void* ptr);
void SNEPPX_cache_barrier(void);

#endif
