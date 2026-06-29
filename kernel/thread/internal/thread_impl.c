/*
 * Thread Pool Internal Implementation — SKELETON
 * VERSION: v0.5
 */

#include "thread_impl.h"
#include <stdlib.h>
#include <string.h>

int arix_scheduler_init(ArixThreadScheduler* sched, int num_workers) {
    if (!sched) return -1;
    memset(sched, 0, sizeof(*sched));
    sched->num_workers = num_workers;
    return 0;
}

void arix_scheduler_destroy(ArixThreadScheduler* sched) {
    if (!sched) return;
    for (int i = 0; i < sched->num_workers; i++) {
        arix_worker_destroy(sched->workers[i]);
    }
    free(sched->workers);
    free(sched->global_queue);
}

int arix_scheduler_start(ArixThreadScheduler* sched) { (void)sched; return 0; }
int arix_scheduler_stop(ArixThreadScheduler* sched) { (void)sched; return 0; }

int arix_scheduler_submit(ArixThreadScheduler* sched, ArixTaskFn fn, void* arg, int priority) {
    (void)sched; (void)fn; (void)arg; (void)priority; return 0;
}

int arix_scheduler_submit_stealable(ArixThreadScheduler* sched, ArixTaskFn fn, void* arg) {
    (void)sched; (void)fn; (void)arg; return 0;
}

int arix_worker_init(ArixWorker* worker, int id) {
    if (!worker) return -1;
    memset(worker, 0, sizeof(*worker));
    worker->worker_id = id;
    return 0;
}

void arix_worker_destroy(ArixWorker* worker) {
    if (worker) free(worker->deque);
}

int arix_worker_push_task(ArixWorker* worker, ArixTask task) {
    (void)worker; (void)task; return 0;
}

int arix_worker_steal_task(ArixWorker* thief, ArixWorker* victim, ArixTask* out) {
    (void)thief; (void)victim; (void)out; return 0;
}

void arix_scheduler_wait_idle(ArixThreadScheduler* sched) { (void)sched; }
int arix_scheduler_worker_count(const ArixThreadScheduler* sched) {
    return sched ? sched->num_workers : 0;
}
