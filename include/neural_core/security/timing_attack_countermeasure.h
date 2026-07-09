#ifndef SNEPPX_TIMING_H
#define SNEPPX_TIMING_H

#include <stddef.h>
#include <stdint.h>

uint64_t SNEPPX_timing_start(void);
uint64_t SNEPPX_timing_end(void);
void SNEPPX_timing_random_delay(uint32_t min_ns, uint32_t max_ns);
int SNEPPX_timing_safe_equal(const uint8_t* a, const uint8_t* b, size_t len, uint64_t* timing_ns);

#endif
