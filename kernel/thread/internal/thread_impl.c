/*
 * Thread Pool Internal Implementation — SKELETON
 * VERSION: v0.5
 */

#include "thread_impl.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_scheduler_init(SNEPPXThreadScheduler* sched, int num_workers) {
    if (!sched) return -1;
    memset(sched, 0, sizeof(*sched));
    sched->num_workers = num_workers;
    return 0;
}

void SNEPPX_scheduler_destroy(SNEPPXThreadScheduler* sched) {
    if (!sched) return;
    for (int i = 0; i < sched->num_workers; i++) {
        SNEPPX_worker_destroy(sched->workers[i]);
    }
    free(sched->workers);
    free(sched->global_queue);
}

int SNEPPX_scheduler_start(SNEPPXThreadScheduler* sched) { (void)sched; return 0; }
int SNEPPX_scheduler_stop(SNEPPXThreadScheduler* sched) { (void)sched; return 0; }

int SNEPPX_scheduler_submit(SNEPPXThreadScheduler* sched, SNEPPXTaskFn fn, void* arg, int priority) {
    (void)sched; (void)fn; (void)arg; (void)priority; return 0;
}

int SNEPPX_scheduler_submit_stealable(SNEPPXThreadScheduler* sched, SNEPPXTaskFn fn, void* arg) {
    (void)sched; (void)fn; (void)arg; return 0;
}

int SNEPPX_worker_init(SNEPPXWorker* worker, int id) {
    if (!worker) return -1;
    memset(worker, 0, sizeof(*worker));
    worker->worker_id = id;
    return 0;
}

void SNEPPX_worker_destroy(SNEPPXWorker* worker) {
    if (worker) free(worker->deque);
}

int SNEPPX_worker_push_task(SNEPPXWorker* worker, SNEPPXTask task) {
    (void)worker; (void)task; return 0;
}

int SNEPPX_worker_steal_task(SNEPPXWorker* thief, SNEPPXWorker* victim, SNEPPXTask* out) {
    (void)thief; (void)victim; (void)out; return 0;
}

void SNEPPX_scheduler_wait_idle(SNEPPXThreadScheduler* sched) { (void)sched; }
int SNEPPX_scheduler_worker_count(const SNEPPXThreadScheduler* sched) {
    return sched ? sched->num_workers : 0;
}
