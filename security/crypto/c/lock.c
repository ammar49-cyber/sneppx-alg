#include "synchronization_lock_interface.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <errno.h>
#include <pthread.h>
#endif

typedef struct SNEPPXLock {
#if defined(_WIN32)
    CRITICAL_SECTION cs;
#else
    pthread_mutex_t mutex;
#endif
    int held;
} SNEPPXLock;

SNEPPXLock* SNEPPX_lock_init(void) {
    SNEPPXLock* lock = (SNEPPXLock*)malloc(sizeof(SNEPPXLock));
    if (!lock) return NULL;
#if defined(_WIN32)
    InitializeCriticalSection(&lock->cs);
#else
    pthread_mutex_init(&lock->mutex, NULL);
#endif
    lock->held = 0;
    return lock;
}

void SNEPPX_lock_destroy(SNEPPXLock* lock) {
    if (!lock) return;
#if defined(_WIN32)
    DeleteCriticalSection(&lock->cs);
#else
    pthread_mutex_destroy(&lock->mutex);
#endif
    free(lock);
}

void SNEPPX_lock_acquire(SNEPPXLock* lock) {
    if (!lock) return;
#if defined(_WIN32)
    EnterCriticalSection(&lock->cs);
#else
    pthread_mutex_lock(&lock->mutex);
#endif
    lock->held = 1;
}

void SNEPPX_lock_release(SNEPPXLock* lock) {
    if (!lock) return;
    lock->held = 0;
#if defined(_WIN32)
    LeaveCriticalSection(&lock->cs);
#else
    pthread_mutex_unlock(&lock->mutex);
#endif
}

int SNEPPX_lock_try_acquire(SNEPPXLock* lock) {
    if (!lock) return 0;
#if defined(_WIN32)
    int result = TryEnterCriticalSection(&lock->cs);
    if (result) lock->held = 1;
    return result;
#else
    int result = pthread_mutex_trylock(&lock->mutex);
    if (result == 0) lock->held = 1;
    return result == 0;
#endif
}

int SNEPPX_lock_is_held(const SNEPPXLock* lock) {
    return lock ? lock->held : 0;
}

int SNEPPX_mlock(void* ptr, size_t len) {
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

int SNEPPX_munlock(void* ptr, size_t len) {
    if (!ptr || !len) return -1;
#if defined(_WIN32)
    if (VirtualUnlock(ptr, len)) return 0;
    return -1;
#else
    return munlock(ptr, len);
#endif
}

int SNEPPX_mlockall_possible(void) {
#if defined(_WIN32)
    return -1;
#else
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) return 0;
    fprintf(stderr, "WARNING: mlockall failed (not privileged)\n");
    return -1;
#endif
/*
 * SNEPPX - Synchronization Lock
 *
 * WHAT
 *   Synchronization Lock.
 *
 * CONCEPT
 *   Fast mutex/lock primitives for protecting shared crypto state across threads.
 *
 * ROLE
 *   Internal synchronization for the crypto module shared state.
 *
 * REFERENCES
 *   None (internal utility).
 */


}