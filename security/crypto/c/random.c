#include "cryptographic_random_generator.h"
#include "sha512_hashing_implementation.h"

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
/*
 * SNEPPX - Random
 *
 * WHAT
 *   Random.
 *
 * CONCEPT
 *   Provides randomness generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


#elif defined(__linux__) || defined(__unix__)
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#elif defined(__APPLE__)
#include <stdlib.h>
#endif

/**
 * @brief Perform Random Bytes.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
/* Test-only deterministic RNG (SHA-512 hash-chain). Disabled by default;
 * enabled via SNEPPX_random_bytes_set_seed for reproducible known-answer tests. */
static int g_test_rng = 0;
static uint8_t g_test_state[64];
static size_t g_test_state_used = 64;

void SNEPPX_random_bytes_set_seed(const uint8_t seed[32]) {
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, seed, 32);
    SNEPPX_sha512_finish(&ctx, g_test_state);
    g_test_state_used = 64;
    g_test_rng = 1;
}

void SNEPPX_random_bytes_clear_seed(void) {
    g_test_rng = 0;
    g_test_state_used = 64;
}

static int test_rng_generate(uint8_t* out, size_t len) {
    size_t produced = 0;
    while (produced < len) {
        if (g_test_state_used >= 64) {
            SNEPPXSHA512Context ctx;
            SNEPPX_sha512_init(&ctx);
            SNEPPX_sha512_update(&ctx, g_test_state, 64);
            SNEPPX_sha512_finish(&ctx, g_test_state);
            g_test_state_used = 0;
        }
        size_t avail = 64 - g_test_state_used;
        size_t take = (len - produced < avail) ? (len - produced) : avail;
        memcpy(out + produced, g_test_state + g_test_state_used, take);
        g_test_state_used += take;
        produced += take;
    }
    return 0;
}

int SNEPPX_random_bytes(uint8_t* buffer, size_t len) {
    if (!buffer || !len) return -1;
    if (g_test_rng) return test_rng_generate(buffer, len);
#ifdef _WIN32
    NTSTATUS status = BCryptGenRandom(NULL, buffer, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status < 0) return -1;
    return 0;
#elif defined(__linux__)
    long ret = syscall(SYS_getrandom, buffer, len, 0);
    if (ret == (long)len) return 0;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return -1;
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, buffer + total, len - total);
        if (n <= 0) { close(fd); return -1; }
        total += (size_t)n;
    }
    close(fd);
    return 0;
#elif defined(__APPLE__)
    arc4random_buf(buffer, len);
    return 0;
#else
    (void)buffer; (void)len;
    return -1;
#endif
/*
 * SNEPPX - Cryptographic Random Number Wrapper
 *
 * WHAT
 *   Cryptographic Random Number Wrapper.
 *
 * CONCEPT
 *   Public API wrapper routing all random requests through the CSPRNG with auto reseeding.
 *
 * ROLE
 *   Public entry point for all cryptographic randomness used by every crypto module.
 *
 * REFERENCES
 *   None (internal utility).
 */


}

/**
 * @brief Perform Random Uint32.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_random_uint32(void) {
    uint32_t val;
    if (SNEPPX_random_bytes((uint8_t*)&val, 4) != 0) return 0;
    return val;
}

/**
 * @brief Perform Random Uniform.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_random_uniform(uint32_t upper_bound) {
    if (upper_bound == 0) return 0;
    uint32_t threshold = -upper_bound % upper_bound;
    while (1) {
        uint32_t r = SNEPPX_random_uint32();
        if (r >= threshold) return r % upper_bound;
    }
}
