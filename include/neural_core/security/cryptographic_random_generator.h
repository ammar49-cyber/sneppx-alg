#ifndef SNEPPX_RANDOM_H
#define SNEPPX_RANDOM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Cryptographic Random Generator
 *
 * WHAT
 *   Cryptographic Random Generator.
 *
 * CONCEPT
 *   Provides randomness generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Random Bytes.
 *
 * @param buffer [out] Buffer value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_random_bytes(uint8_t* buffer, size_t len);
/**
 * @brief Perform Random Uint32.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_random_uint32(void);
/**
 * @brief Perform Random Uniform.
 *
 * @param upper_bound [in] Upper Bound value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_random_uniform(uint32_t upper_bound);


#ifdef __cplusplus
}
#endif
#endif
