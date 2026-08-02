#ifndef SNEPPX_TIMING_H
#define SNEPPX_TIMING_H

#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - Timing Attack Countermeasures
 *
 * WHAT
 *   Timing Attack Countermeasures.
 *
 * CONCEPT
 *   Defenses against timing attacks exploiting data-dependent execution time.
 *
 * ROLE
 *   Layer S5 side-channel resistance used by all crypto modules handling secret keys.
 *
 * REFERENCES
 *   None (internal utility).
 */



uint64_t SNEPPX_timing_start(void);
uint64_t SNEPPX_timing_end(void);
void SNEPPX_timing_random_delay(uint32_t min_ns, uint32_t max_ns);
int SNEPPX_timing_safe_equal(const uint8_t* a, const uint8_t* b, size_t len, uint64_t* timing_ns);

#endif
