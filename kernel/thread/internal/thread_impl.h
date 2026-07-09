#ifndef SNEPPX_THREAD_INTERNAL_H
#define SNEPPX_THREAD_INTERNAL_H
/*
 * Thread Pool Internal — v0.5
 *
 * PURPOSE: Internal work-stealing scheduler, thread-safe task queue,
 * and per-worker data structures for the SNEPPX_threadpool.  Implements
 * a work-stealing algorithm: each worker has a double-ended queue (deque)
 * of tasks; idle workers steal from the tail of another worker's deque.
 *
 * DEPENDENCIES: concurrent_workload_dispatch.h, polymorphic_memory_allocator.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*SNEPPXTaskFn)(void* arg);

typedef struct {
    SNEPPXTaskFn  fn;
    void*       arg;
    uint64_t    id;
    int         priority;
} SNEPPXTask;

typedef struct SNEPPXWorker {
    int            worker_id;
    SNEPPXTask*      deque;
    size_t         deque_capacity;
    size_t         deque_head;
    size_t         deque_tail;
    void*          thread_handle;
    int            is_running;
    uint64_t       tasks_executed;
    uint64_t       tasks_stolen;
} SNEPPXWorker;

typedef struct {
    SNEPPXWorker** workers;
    int          num_workers;
    int          num_threads;
    volatile int shutdown_flag;
    void*        global_lock;
    SNEPPXTask*    global_queue;
    size_t       global_queue_capacity;
    size_t       global_queue_size;
} SNEPPXThreadScheduler;

/* ---------- Scheduler lifecycle ---------- */
int  SNEPPX_scheduler_init(SNEPPXThreadScheduler* sched, int num_workers);
void SNEPPX_scheduler_destroy(SNEPPXThreadScheduler* sched);
int  SNEPPX_scheduler_start(SNEPPXThreadScheduler* sched);
int  SNEPPX_scheduler_stop(SNEPPXThreadScheduler* sched);

/* ---------- Task submission ---------- */
int  SNEPPX_scheduler_submit(SNEPPXThreadScheduler* sched, SNEPPXTaskFn fn, void* arg, int priority);
int  SNEPPX_scheduler_submit_stealable(SNEPPXThreadScheduler* sched, SNEPPXTaskFn fn, void* arg);

/* ---------- Worker operations ---------- */
int  SNEPPX_worker_init(SNEPPXWorker* worker, int id);
void SNEPPX_worker_destroy(SNEPPXWorker* worker);
int  SNEPPX_worker_push_task(SNEPPXWorker* worker, SNEPPXTask task);
int  SNEPPX_worker_steal_task(SNEPPXWorker* thief, SNEPPXWorker* victim, SNEPPXTask* out);

/* ---------- Synchronization ---------- */
void SNEPPX_scheduler_wait_idle(SNEPPXThreadScheduler* sched);
int  SNEPPX_scheduler_worker_count(const SNEPPXThreadScheduler* sched);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_THREAD_INTERNAL_H */
