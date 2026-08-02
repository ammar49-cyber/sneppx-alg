#ifndef SNEPPX_POWER_H
#define SNEPPX_POWER_H

#include <stdint.h>
/*
 * SNEPPX - Power Analysis Mitigation
 *
 * WHAT
 *   Power Analysis Mitigation.
 *
 * CONCEPT
 *   Defenses against differential and simple power analysis on cryptographic operations.
 *
 * ROLE
 *   Layer S5 side-channel resistance of the S0-S9 security stack.
 *
 * REFERENCES
 *   None (internal hardening).
 */



void SNEPPX_power_balance_start(void);
void SNEPPX_power_balance_end(void);
void SNEPPX_power_dummy_op(void);

int SNEPPX_power_mul_const_time(int a, int b);
int SNEPPX_power_cmp_const_time(int a, int b);
uint64_t SNEPPX_power_mask_gen(int condition);

#endif
