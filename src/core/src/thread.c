#include "arix_thread.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#pragma intrinsic(_InterlockedExchangeAdd)
typedef HANDLE            arix_thread_t;
typedef CRITICAL_SECTION  arix_mutex_t;
typedef CONDITION_VARIABLE arix_cond_t;
#else
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
typedef pthread_mutex_t arix_mutex_t;
typedef pthread_cond_t  arix_cond_t;
typedef pthread_t       arix_thread_t;
#endif

/* ==================================================================
 *  Platform abstractions  (macros to avoid MSVC C typedef + inline issues)
 * ================================================================== */

#ifdef _WIN32

typedef CRITICAL_SECTION  arix_mutex_t;
typedef CONDITION_VARIABLE arix_cond_t;
typedef HANDLE            arix_thread_t;

#define arix_mutex_init(m)     (InitializeCriticalSection(m), 0)
#define arix_mutex_destroy(m)  DeleteCriticalSection(m)
#define arix_mutex_lock(m)     EnterCriticalSection(m)
#define arix_mutex_unlock(m)   LeaveCriticalSection(m)

#define arix_cond_init(c)      (InitializeConditionVariable(c), 0)
#define arix_cond_destroy(c)   ((void)(c))
#define arix_cond_wait(c, m)   (SleepConditionVariableCS((c), (m), INFINITE) ? 0 : -1)
#define arix_cond_signal(c)    (WakeConditionVariable(c), 0)
#define arix_cond_broadcast(c) (WakeAllConditionVariable(c), 0)

#define arix_thread_create(t, func, arg) \
    (*(HANDLE*)(t) = CreateThread(NULL, 0, (func), (arg), 0, NULL), (*(HANDLE*)(t) ? 0 : -1))
#define arix_thread_join(t)    (WaitForSingleObject((HANDLE)(t), INFINITE), CloseHandle((HANDLE)(t)), 0)

#else /* POSIX */

typedef pthread_mutex_t arix_mutex_t;
typedef pthread_cond_t  arix_cond_t;
typedef pthread_t       arix_thread_t;

#define arix_mutex_init(m)     pthread_mutex_init((m), NULL)
#define arix_mutex_destroy(m)  pthread_mutex_destroy(m)
#define arix_mutex_lock(m)     pthread_mutex_lock(m)
#define arix_mutex_unlock(m)   pthread_mutex_unlock(m)

#define arix_cond_init(c)      pthread_cond_init((c), NULL)
#define arix_cond_destroy(c)   pthread_cond_destroy(c)
#define arix_cond_wait(c, m)   pthread_cond_wait((c), (m))
#define arix_cond_signal(c)    pthread_cond_signal(c)
#define arix_cond_broadcast(c) pthread_cond_broadcast(c)

#define arix_thread_create(t, func, arg) pthread_create((pthread_t*)(t), NULL, (func), (arg))
#define arix_thread_join(t)    pthread_join((pthread_t)(t), NULL)

#endif

/* ==================================================================
 *  Work-stealing deque  (lock-based, per-thread)
 * ================================================================== */

typedef struct WorkNode {
    ArixTask           task;
    struct WorkNode*   next;
} WorkNode;

typedef struct {
    WorkNode*          head;   /* front (stolen from)  */
    WorkNode*          tail;   /* back  (owner pushes) */
    arix_mutex_t       lock;
} WorkDeque;

static void deque_init(WorkDeque* q) {
    q->head = q->tail = NULL;
    arix_mutex_init(&q->lock);
}

static void deque_destroy(WorkDeque* q) {
    /* Drain remaining */
    arix_mutex_lock(&q->lock);
    WorkNode* n = q->head;
    while (n) { WorkNode* next = n->next; free(n); n = next; }
    q->head = q->tail = NULL;
    arix_mutex_unlock(&q->lock);
    arix_mutex_destroy(&q->lock);
}

/* Owner pushes to the back (LIFO for locality). */
static int deque_push(WorkDeque* q, ArixTask task) {
    WorkNode* n = (WorkNode*)malloc(sizeof(WorkNode));
    if (!n) return -1;
    n->task = task;
    n->next = NULL;
    arix_mutex_lock(&q->lock);
    if (q->tail) {
        q->tail->next = n;
        q->tail = n;
    } else {
        q->head = q->tail = n;
    }
    arix_mutex_unlock(&q->lock);
    return 0;
}

/* Owner pops from the back.  Returns 0 on success, -1 if empty. */
static int deque_pop(WorkDeque* q, ArixTask* out) {
    arix_mutex_lock(&q->lock);
    WorkNode* prev = NULL;
    WorkNode* cur  = q->head;
    while (cur && cur != q->tail) { prev = cur; cur = cur->next; }
    if (!cur) { arix_mutex_unlock(&q->lock); return -1; }
    *out = cur->task;
    if (prev) prev->next = NULL; else q->head = NULL;
    q->tail = prev;
    free(cur);
    arix_mutex_unlock(&q->lock);
    return 0;
}

/* Steal from the front.  Returns 0 on success, -1 if empty. */
static int deque_steal(WorkDeque* q, ArixTask* out) {
    arix_mutex_lock(&q->lock);
    WorkNode* n = q->head;
    if (!n) { arix_mutex_unlock(&q->lock); return -1; }
    *out = n->task;
    q->head = n->next;
    if (!q->head) q->tail = NULL;
    free(n);
    arix_mutex_unlock(&q->lock);
    return 0;
}

/* Number of tasks currently in the deque. */
static int deque_count(WorkDeque* q) {
    arix_mutex_lock(&q->lock);
    int cnt = 0;
    for (WorkNode* n = q->head; n; n = n->next) cnt++;
    arix_mutex_unlock(&q->lock);
    return cnt;
}

/* ==================================================================
 *  Future
 * ================================================================== */

struct ArixFuture {
    volatile int  ready;
    void*         result;
    arix_mutex_t  lock;
    arix_cond_t   cond;
};

ArixFuture* arix_future_create(void) {
    ArixFuture* fut = (ArixFuture*)malloc(sizeof(ArixFuture));
    if (!fut) return NULL;
    fut->ready  = 0;
    fut->result = NULL;
    arix_mutex_init(&fut->lock);
    arix_cond_init(&fut->cond);
    return fut;
}

void arix_future_destroy(ArixFuture* fut) {
    if (!fut) return;
    arix_cond_destroy(&fut->cond);
    arix_mutex_destroy(&fut->lock);
    free(fut);
}

void arix_future_set_result(ArixFuture* fut, void* result) {
    arix_mutex_lock(&fut->lock);
    fut->result  = result;
    fut->ready   = 1;
    arix_mutex_unlock(&fut->lock);
    arix_cond_broadcast(&fut->cond);
}

void arix_future_wait(ArixFuture* fut) {
    arix_mutex_lock(&fut->lock);
    while (!fut->ready)
        arix_cond_wait(&fut->cond, &fut->lock);
    arix_mutex_unlock(&fut->lock);
}

int arix_future_is_ready(ArixFuture* fut) {
    return fut->ready;
}

void* arix_future_get_result(ArixFuture* fut) {
    arix_future_wait(fut);
    return fut->result;
}

/* ==================================================================
 *  Thread Pool
 *
 *  struct defined with native types to avoid MSVC C typedef scoping
 * ================================================================== */

struct ArixThreadPool {
    size_t              num_threads;
    volatile int        shutdown;
    uintptr_t*          threads;       /* [num_threads] — cast to native type */
    WorkDeque*          queues;        /* [num_threads]  */
    arix_mutex_t        global_lock;
    arix_cond_t         global_cond;
    volatile int        active_tasks;  /* live + queued  */
    volatile int        waiting_count; /* sleepers       */
};

/* Helper define for allocating thread handles (native type per platform) */
#ifdef _WIN32
#define ARIX_THREAD_SIZE  sizeof(HANDLE)
#else
#define ARIX_THREAD_SIZE  sizeof(pthread_t)
#endif

/* Thread ID used for round-robin task distribution, stored in TLS */
#ifdef _MSC_VER
static __declspec(thread) int g_thread_id = -1;
#else
static __thread int g_thread_id = -1;
#endif

static void set_thread_id(int id) { g_thread_id = id; }
int  arix_thread_id(void)         { return g_thread_id; }

/* Worker thread entry point */
#ifdef _WIN32
static DWORD WINAPI worker_entry(LPVOID arg)
#else
static void* worker_entry(void* arg)
#endif
{
    ArixThreadPool* pool = (ArixThreadPool*)((void**)arg)[0];
    int             tid  = (int)(intptr_t)((void**)arg)[1];
    set_thread_id(tid);
    free(arg);

    ArixTask task;
    while (!pool->shutdown) {
        int got = 0;

        /* 1. Try own queue (pop back — LIFO) */
        if (deque_pop(&pool->queues[tid], &task) == 0) {
            got = 1;
        } else {
            /* 2. Steal from other queues (front — FIFO) */
            for (size_t i = 1; i < pool->num_threads; i++) {
                size_t victim = (tid + i) % pool->num_threads;
                if (deque_steal(&pool->queues[victim], &task) == 0) {
                    got = 1;
                    break;
                }
            }
        }

        if (got) {
            task.func(task.arg);
            arix_mutex_lock(&pool->global_lock);
            pool->active_tasks--;
            arix_mutex_unlock(&pool->global_lock);
            continue;
        }

        /* 3. No work — sleep until signaled (predicate check avoids lost wakeup) */
        arix_mutex_lock(&pool->global_lock);
        pool->waiting_count++;
        while (!pool->shutdown && pool->active_tasks == 0) {
            arix_cond_wait(&pool->global_cond, &pool->global_lock);
        }
        pool->waiting_count--;
        arix_mutex_unlock(&pool->global_lock);
    }
    return 0;
}

ArixThreadPool* arix_threadpool_create(size_t num_threads) {
    if (num_threads == 0) num_threads = arix_threadpool_default_count();

    ArixThreadPool* pool = (ArixThreadPool*)malloc(sizeof(ArixThreadPool));
    if (!pool) return NULL;

    pool->num_threads   = num_threads;
    pool->shutdown      = 0;
    pool->active_tasks  = 0;
    pool->waiting_count = 0;

    pool->threads = (uintptr_t*)malloc(num_threads * sizeof(uintptr_t));
    pool->queues  = (WorkDeque*)malloc(num_threads * sizeof(WorkDeque));
    if (!pool->threads || !pool->queues) {
        free(pool->threads); free(pool->queues); free(pool);
        return NULL;
    }

    arix_mutex_init(&pool->global_lock);
    arix_cond_init(&pool->global_cond);

    for (size_t i = 0; i < num_threads; i++)
        deque_init(&pool->queues[i]);

    /* Start worker threads */
    size_t i;
    for (i = 0; i < num_threads; i++) {
        void** args = (void**)malloc(2 * sizeof(void*));
        if (!args) goto fail;
        args[0] = (void*)pool;
        args[1] = (void*)(intptr_t)i;
        if (arix_thread_create(&pool->threads[i], worker_entry, args) != 0) {
            free(args);
            goto fail;
        }
    }

    return pool;

fail:
    for (size_t j = 0; j < num_threads; j++) {
        if (j < i) {
            arix_thread_join(pool->threads[j]);
            deque_destroy(&pool->queues[j]);
        } else {
            deque_destroy(&pool->queues[j]);
        }
    }
    arix_cond_destroy(&pool->global_cond);
    arix_mutex_destroy(&pool->global_lock);
    free(pool->threads); free(pool->queues); free(pool);
    return NULL;
}

void arix_threadpool_destroy(ArixThreadPool* pool) {
    if (!pool) return;
    pool->shutdown = 1;
    arix_cond_broadcast(&pool->global_cond);
    for (size_t i = 0; i < pool->num_threads; i++) {
        arix_thread_join(pool->threads[i]);
        deque_destroy(&pool->queues[i]);
    }
    arix_cond_destroy(&pool->global_cond);
    arix_mutex_destroy(&pool->global_lock);
    free(pool->threads);
    free(pool->queues);
    free(pool);
}

/* Submit a task to a thread's queue (round-robin via atomic index or random). */
static volatile long g_rr_counter = 0;

int arix_threadpool_submit(ArixThreadPool* pool, ArixTask task) {
    size_t idx;
    if (g_thread_id >= 0) {
        idx = (size_t)g_thread_id;   /* own queue  */
    } else {
#ifdef _WIN32
        idx = (size_t)(_InterlockedExchangeAdd(&g_rr_counter, 1) % (long)pool->num_threads);
#else
        idx = (size_t)(__sync_fetch_and_add(&g_rr_counter, 1) % (long)pool->num_threads);
#endif
    }
    if (idx >= pool->num_threads) idx = 0;

    int ret = deque_push(&pool->queues[idx], task);
    if (ret != 0) return ret;

    arix_mutex_lock(&pool->global_lock);
    pool->active_tasks++;
    /* Wake one sleeper */
    if (pool->waiting_count > 0)
        arix_cond_signal(&pool->global_cond);
    arix_mutex_unlock(&pool->global_lock);
    return 0;
}

typedef struct {
    ArixTask    inner;
    ArixFuture* future;
} FutureWrapper;

static void future_wrapper_func(void* arg) {
    FutureWrapper* fw = (FutureWrapper*)arg;
    fw->inner.func(fw->inner.arg);
    arix_future_set_result(fw->future, NULL);
    free(fw);
}

int arix_threadpool_submit_future(ArixThreadPool* pool, ArixTask task, ArixFuture* fut) {
    if (!fut) return arix_threadpool_submit(pool, task);
    FutureWrapper* fw = (FutureWrapper*)malloc(sizeof(FutureWrapper));
    if (!fw) return -1;
    fw->inner  = task;
    fw->future = fut;
    ArixTask wrapper;
    wrapper.func = future_wrapper_func;
    wrapper.arg  = fw;
    return arix_threadpool_submit(pool, wrapper);
}

void arix_threadpool_wait(ArixThreadPool* pool) {
    /* Busy-wait until all tasks are done.  Workers are guaranteed to wake
     * for every submitted task because the predicate-based cond_wait loop
     * cannot lose signals. */
    while (1) {
        int all_empty = 1;
        for (size_t i = 0; i < pool->num_threads; i++) {
            if (deque_count(&pool->queues[i]) > 0) {
                all_empty = 0;
                break;
            }
        }
        if (all_empty && pool->active_tasks == 0) break;
#ifdef _WIN32
        SwitchToThread();
#else
        sched_yield();
#endif
    }
}

size_t arix_threadpool_default_count(void) {
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    size_t n = (size_t)info.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) n = 4;
#endif
    return (n < 2) ? 2 : n;
}

/* ==================================================================
 *  Parallel For
 * ================================================================== */

typedef struct {
    ArixRangeFunc func;
    void*         arg;
    size_t        start;
    size_t        end;
    size_t        chunk;
} ForTaskArg;

static void for_task_func(void* arg) {
    ForTaskArg* fta = (ForTaskArg*)arg;
    fta->func(fta->start, fta->end, fta->arg);
    free(fta);
}

void arix_parallel_for(ArixThreadPool* pool,
                       size_t start, size_t end,
                       ArixRangeFunc func, void* arg) {
    size_t range   = end - start;
    size_t nthreads = pool->num_threads;
    if (range == 0) return;
    if (nthreads <= 1 || range < 64) {
        func(start, end, arg);
        return;
    }

    size_t chunk_size = (range + nthreads - 1) / nthreads;
    size_t num_tasks  = 0;

    for (size_t s = start; s < end; s += chunk_size) {
        size_t e = s + chunk_size;
        if (e > end) e = end;
        ForTaskArg* fta = (ForTaskArg*)malloc(sizeof(ForTaskArg));
        if (!fta) { func(s, e, arg); continue; }
        fta->func  = func;
        fta->arg   = arg;
        fta->start = s;
        fta->end   = e;
        fta->chunk = chunk_size;

        ArixTask task;
        task.func = for_task_func;
        task.arg  = fta;
        if (arix_threadpool_submit(pool, task) == 0)
            num_tasks++;
        else
            func(s, e, arg);
    }

    arix_threadpool_wait(pool);
}

/* ==================================================================
 *  Parallel Reduce
 * ================================================================== */

typedef struct {
    ArixReduceFunc  reduce_func;
    ArixCombineFunc combine_func;
    void*           arg;
    size_t          start;
    size_t          end;
    void*           result;   /* partial result buffer */
    size_t          elem_size;
} ReduceTaskArg;

static void reduce_task_func(void* arg) {
    ReduceTaskArg* rta = (ReduceTaskArg*)arg;
    memset(rta->result, 0, rta->elem_size);
    rta->reduce_func(rta->start, rta->end, rta->arg, rta->result);
    free(rta);
}

void arix_parallel_reduce(ArixThreadPool* pool,
                          size_t start, size_t end,
                          void* init, size_t elem_size,
                          ArixReduceFunc reduce_func,
                          ArixCombineFunc combine_func,
                          void* result,
                          void* user_arg) {
    size_t range    = end - start;
    size_t nthreads = pool->num_threads;

    if (init) memcpy(result, init, elem_size);
    else      memset(result, 0, elem_size);
    if (range == 0) return;

    /* Single-threaded path */
    if (nthreads <= 1 || range < 64) {
        memset(result, 0, elem_size);
        reduce_func(start, end, user_arg, result);
        return;
    }

    size_t chunk_size  = (range + nthreads - 1) / nthreads;
    size_t num_chunks  = (range + chunk_size - 1) / chunk_size;
    size_t max_chunks  = num_chunks < nthreads ? num_chunks : nthreads;

    /* Heap-allocate partial results */
    unsigned char* partials = (unsigned char*)calloc(max_chunks, elem_size);
    if (!partials) { reduce_func(start, end, NULL, result); return; }

    size_t chunks_submitted = 0;
    for (size_t s = start; s < end; s += chunk_size) {
        size_t e = s + chunk_size;
        if (e > end) e = end;

        ReduceTaskArg* rta = (ReduceTaskArg*)malloc(sizeof(ReduceTaskArg));
        if (!rta) { reduce_func(s, e, NULL, result); free(partials); return; }
        rta->reduce_func  = reduce_func;
        rta->combine_func = combine_func;
        rta->arg          = user_arg;
        rta->start        = s;
        rta->end          = e;
        rta->result       = partials + chunks_submitted * elem_size;
        rta->elem_size    = elem_size;

        ArixTask task;
        task.func = reduce_task_func;
        task.arg  = rta;
        arix_threadpool_submit(pool, task);
        chunks_submitted++;
    }

    arix_threadpool_wait(pool);

    /* Combine partials */
    if (chunks_submitted > 0) {
        memcpy(result, partials, elem_size);
        for (size_t i = 1; i < chunks_submitted; i++) {
            combine_func(result, partials + i * elem_size);
        }
    }

    free(partials);
}
