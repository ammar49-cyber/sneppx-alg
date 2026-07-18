#include "polymorphic_memory_allocator.h"
#include <stdlib.h>

/* NOTE: SNEPPX_malloc / SNEPPX_free / SNEPPX_realloc are intentionally NOT
 * defined here. The real, security-hardened implementations (aligned +
 * zeroing + SNEPPX_secure_zero on free) live in `allocator.c` and are linked
 * into the same library. Defining stub versions here would cause the linker
 * to discard the real ones. Only the helpers unique to this interface layer
 * are kept below. */

void* SNEPPX_calloc(size_t count, size_t size, size_t alignment) {
    (void)alignment;
    return calloc(count, size);
}

size_t SNEPPX_allocated_size(const void* ptr) {
    (void)ptr;
    return 0;
}

int SNEPPX_malloc_stats(size_t* total_allocated, size_t* peak_allocated, size_t* num_allocs) {
    (void)total_allocated; (void)peak_allocated; (void)num_allocs;
    return 0;
}
