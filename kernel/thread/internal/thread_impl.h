#ifndef ARIX_THREAD_INTERNAL_H
#define ARIX_THREAD_INTERNAL_H
/*
 * Thread Pool Internal — v0.5
 *
 * PURPOSE: Internal work-stealing scheduler, thread-safe task queue,
 * and per-worker data structures for the arix_threadpool.  Implements
 * a work-stealing algorithm: each worker has a double-ended queue (deque)
 * of tasks; idle workers steal from the tail of another worker's deque.
 *
 * DEPENDENCIES: arix_thread.h, arix_memory.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ArixTaskFn)(void* arg);

typedef struct {
    ArixTaskFn  fn;
    void*       arg;
    uint64_t    id;
    int         priority;
} ArixTask;

typedef struct ArixWorker {
    int            worker_id;
    ArixTask*      deque;
    size_t         deque_capacity;
    size_t         deque_head;
    size_t         deque_tail;
    void*          thread_handle;
    int            is_running;
    uint64_t       tasks_executed;
    uint64_t       tasks_stolen;
} ArixWorker;

typedef struct {
    ArixWorker** workers;
    int          num_workers;
    int          num_threads;
    volatile int shutdown_flag;
    void*        global_lock;
    ArixTask*    global_queue;
    size_t       global_queue_capacity;
    size_t       global_queue_size;
} ArixThreadScheduler;

/* ---------- Scheduler lifecycle ---------- */
int  arix_scheduler_init(ArixThreadScheduler* sched, int num_workers);
void arix_scheduler_destroy(ArixThreadScheduler* sched);
int  arix_scheduler_start(ArixThreadScheduler* sched);
int  arix_scheduler_stop(ArixThreadScheduler* sched);

/* ---------- Task submission ---------- */
int  arix_scheduler_submit(ArixThreadScheduler* sched, ArixTaskFn fn, void* arg, int priority);
int  arix_scheduler_submit_stealable(ArixThreadScheduler* sched, ArixTaskFn fn, void* arg);

/* ---------- Worker operations ---------- */
int  arix_worker_init(ArixWorker* worker, int id);
void arix_worker_destroy(ArixWorker* worker);
int  arix_worker_push_task(ArixWorker* worker, ArixTask task);
int  arix_worker_steal_task(ArixWorker* thief, ArixWorker* victim, ArixTask* out);

/* ---------- Synchronization ---------- */
void arix_scheduler_wait_idle(ArixThreadScheduler* sched);
int  arix_scheduler_worker_count(const ArixThreadScheduler* sched);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_THREAD_INTERNAL_H */
