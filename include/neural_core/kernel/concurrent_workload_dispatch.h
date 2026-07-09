#ifndef SNEPPX_THREAD_H
#define SNEPPX_THREAD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Task
 * ============================================================ */

typedef struct {
    void (*func)(void* arg);
    void* arg;
} SNEPPXTask;

/* ============================================================
 * Future  –  synchronise on a single task
 * ============================================================ */

typedef struct SNEPPXFuture SNEPPXFuture;

SNEPPXFuture* SNEPPX_future_create(void);
void        SNEPPX_future_destroy(SNEPPXFuture* fut);
void        SNEPPX_future_wait(SNEPPXFuture* fut);
int         SNEPPX_future_is_ready(SNEPPXFuture* fut);
void        SNEPPX_future_set_result(SNEPPXFuture* fut, void* result);
void*       SNEPPX_future_get_result(SNEPPXFuture* fut);

/* ============================================================
 * Thread Pool  –  work-stealing, N workers
 * ============================================================ */

typedef struct SNEPPXThreadPool SNEPPXThreadPool;

SNEPPXThreadPool* SNEPPX_threadpool_create(size_t num_threads);
void            SNEPPX_threadpool_destroy(SNEPPXThreadPool* pool);

/* Submit a task (copied).  Returns 0 on success. */
int SNEPPX_threadpool_submit(SNEPPXThreadPool* pool, SNEPPXTask task);

/* Submit a task with a future to track completion. */
int SNEPPX_threadpool_submit_future(SNEPPXThreadPool* pool, SNEPPXTask task, SNEPPXFuture* fut);

/* Wait until all submitted tasks complete. */
void SNEPPX_threadpool_wait(SNEPPXThreadPool* pool);

/* Default number of threads (hardware concurrency, min 2). */
size_t SNEPPX_threadpool_default_count(void);

/* ============================================================
 * Parallel For / Reduce
 * ============================================================ */

/* Callback for parallel_for – processes chunk [start, end). */
typedef void (*SNEPPXRangeFunc)(size_t start, size_t end, void* arg);

void SNEPPX_parallel_for(SNEPPXThreadPool* pool,
                       size_t start, size_t end,
                       SNEPPXRangeFunc func, void* arg);

/* Reduce callback – processes chunk [start, end) into result. */
typedef void (*SNEPPXReduceFunc)(size_t start, size_t end,
                               void* arg, void* result);

/* Combine callback – merges src into dst. */
typedef void (*SNEPPXCombineFunc)(void* dst, const void* src);

void SNEPPX_parallel_reduce(SNEPPXThreadPool* pool,
                          size_t start, size_t end,
                          void* init, size_t elem_size,
                          SNEPPXReduceFunc reduce_func,
                          SNEPPXCombineFunc combine_func,
                          void* result,
                          void* user_arg);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_THREAD_H */
