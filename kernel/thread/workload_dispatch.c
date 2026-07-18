#include "concurrent_workload_dispatch.h"
#include <stdlib.h>

typedef void (*SNEPPXTaskFn)(void* arg);

typedef struct SNEPPXWorkload {
    int dummy;
} SNEPPXWorkload;

typedef struct SNEPPXTaskGroup {
    int dummy;
} SNEPPXTaskGroup;

SNEPPXWorkload* SNEPPX_workload_create(size_t max_tasks) {
    (void)max_tasks;
    return (SNEPPXWorkload*)calloc(1, sizeof(SNEPPXWorkload));
}

void SNEPPX_workload_destroy(SNEPPXWorkload* wl) {
    free(wl);
}

int SNEPPX_workload_add_task(SNEPPXWorkload* wl, const char* name, SNEPPXTaskFn fn, void* args) {
    (void)wl; (void)name; (void)fn; (void)args;
    return 0;
}

int SNEPPX_workload_submit(SNEPPXWorkload* wl, int num_threads) {
    (void)wl; (void)num_threads;
    return 0;
}

int SNEPPX_workload_wait(SNEPPXWorkload* wl, int timeout_ms) {
    (void)wl; (void)timeout_ms;
    return 0;
}

size_t SNEPPX_workload_num_completed(const SNEPPXWorkload* wl) {
    (void)wl;
    return 0;
}

int SNEPPX_workload_cancel(SNEPPXWorkload* wl) {
    (void)wl;
    return 0;
}

SNEPPXTaskGroup* SNEPPX_task_group_create(size_t num_tasks) {
    (void)num_tasks;
    return (SNEPPXTaskGroup*)calloc(1, sizeof(SNEPPXTaskGroup));
}

void SNEPPX_task_group_destroy(SNEPPXTaskGroup* group) {
    free(group);
}

int SNEPPX_task_group_add(SNEPPXTaskGroup* group, SNEPPXTaskFn fn, void* args) {
    (void)group; (void)fn; (void)args;
    return 0;
}

int SNEPPX_task_group_run_all(SNEPPXTaskGroup* group) {
    (void)group;
    return 0;
}

int SNEPPX_task_group_wait_all(SNEPPXTaskGroup* group, int timeout_ms) {
    (void)group; (void)timeout_ms;
    return 0;
}
