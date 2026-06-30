#ifndef ARIX_THREAD_H
#define ARIX_THREAD_H

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
} ArixTask;

/* ============================================================
 * Future  –  synchronise on a single task
 * ============================================================ */

typedef struct ArixFuture ArixFuture;

ArixFuture* arix_future_create(void);
void        arix_future_destroy(ArixFuture* fut);
void        arix_future_wait(ArixFuture* fut);
int         arix_future_is_ready(ArixFuture* fut);
void        arix_future_set_result(ArixFuture* fut, void* result);
void*       arix_future_get_result(ArixFuture* fut);

/* ============================================================
 * Thread Pool  –  work-stealing, N workers
 * ============================================================ */

typedef struct ArixThreadPool ArixThreadPool;

ArixThreadPool* arix_threadpool_create(size_t num_threads);
void            arix_threadpool_destroy(ArixThreadPool* pool);

/* Submit a task (copied).  Returns 0 on success. */
int arix_threadpool_submit(ArixThreadPool* pool, ArixTask task);

/* Submit a task with a future to track completion. */
int arix_threadpool_submit_future(ArixThreadPool* pool, ArixTask task, ArixFuture* fut);

/* Wait until all submitted tasks complete. */
void arix_threadpool_wait(ArixThreadPool* pool);

/* Default number of threads (hardware concurrency, min 2). */
size_t arix_threadpool_default_count(void);

/* ============================================================
 * Parallel For / Reduce
 * ============================================================ */

/* Callback for parallel_for – processes chunk [start, end). */
typedef void (*ArixRangeFunc)(size_t start, size_t end, void* arg);

void arix_parallel_for(ArixThreadPool* pool,
                       size_t start, size_t end,
                       ArixRangeFunc func, void* arg);

/* Reduce callback – processes chunk [start, end) into result. */
typedef void (*ArixReduceFunc)(size_t start, size_t end,
                               void* arg, void* result);

/* Combine callback – merges src into dst. */
typedef void (*ArixCombineFunc)(void* dst, const void* src);

void arix_parallel_reduce(ArixThreadPool* pool,
                          size_t start, size_t end,
                          void* init, size_t elem_size,
                          ArixReduceFunc reduce_func,
                          ArixCombineFunc combine_func,
                          void* result,
                          void* user_arg);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_THREAD_H */
