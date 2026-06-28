#include "arix_lock.h"
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <errno.h>
#endif

int arix_mlock(void* ptr, size_t len) {
    if (!ptr || !len) return -1;
#if defined(_WIN32)
    if (VirtualLock(ptr, len)) return 0;
    fprintf(stderr, "WARNING: VirtualLock failed (%lu)\n", GetLastError());
    return -1;
#else
    if (mlock(ptr, len) == 0) return 0;
    if (errno == EPERM) {
        fprintf(stderr, "WARNING: mlock failed: insufficient privileges\n");
    } else {
        fprintf(stderr, "WARNING: mlock failed: errno=%d\n", errno);
    }
    return -1;
#endif
}

int arix_munlock(void* ptr, size_t len) {
    if (!ptr || !len) return -1;
#if defined(_WIN32)
    if (VirtualUnlock(ptr, len)) return 0;
    return -1;
#else
    return munlock(ptr, len);
#endif
}

int arix_mlockall_possible(void) {
#if defined(_WIN32)
    return -1;
#else
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) return 0;
    fprintf(stderr, "WARNING: mlockall failed (not privileged)\n");
    return -1;
#endif
}
