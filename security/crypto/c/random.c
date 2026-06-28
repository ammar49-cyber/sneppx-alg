#include "arix_random.h"

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#elif defined(__linux__) || defined(__unix__)
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#elif defined(__APPLE__)
#include <stdlib.h>
#endif

int arix_random_bytes(uint8_t* buffer, size_t len) {
    if (!buffer || !len) return -1;
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
}

uint32_t arix_random_uint32(void) {
    uint32_t val;
    if (arix_random_bytes((uint8_t*)&val, 4) != 0) return 0;
    return val;
}

uint32_t arix_random_uniform(uint32_t upper_bound) {
    if (upper_bound == 0) return 0;
    uint32_t threshold = -upper_bound % upper_bound;
    while (1) {
        uint32_t r = arix_random_uint32();
        if (r >= threshold) return r % upper_bound;
    }
}
