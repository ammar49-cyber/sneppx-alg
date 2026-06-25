#include "arix_aslr.h"
#include "arix_random.h"

size_t arix_aslr_random_offset(size_t max_offset) {
    if (max_offset == 0) return 0;
    uint64_t r;
    arix_random_bytes((uint8_t*)&r, sizeof(r));
    size_t offset = (size_t)(r % max_offset);
    size_t page = 4096;
    offset &= ~(page - 1);
    return offset;
}

void arix_aslr_apply(void** base_ptr, size_t* size, size_t max_random) {
    if (!base_ptr || !*base_ptr || !size) return;
    size_t off = arix_aslr_random_offset(max_random);
    *base_ptr = (uint8_t*)*base_ptr + off;
    *size -= off;
}
