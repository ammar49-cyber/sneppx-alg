#ifndef ARIX_MEMORY_LEAK_DETECTOR_H
#define ARIX_MEMORY_LEAK_DETECTOR_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    void *ptr;
    size_t size;
    const char *file;
    int line;
    const char *func;
    uint64_t timestamp;
} leak_report_t;

typedef struct {
    int total_allocations;
    int active_allocations;
    uint64_t total_allocated;
    uint64_t total_freed;
    uint64_t current_usage;
    uint64_t peak_usage;
    int leak_threshold;
    int tracking_enabled;
} leak_stats_t;

int arix_leak_init(void);
int arix_leak_track_alloc(void *ptr, size_t size, const char *file, int line, const char *func);
int arix_leak_track_free(void *ptr);
int arix_leak_check(leak_report_t *reports, int max_reports);
int arix_leak_get_stats(leak_stats_t *stats);
int arix_leak_set_threshold(int bytes);
int arix_leak_enable_tracking(void);
int arix_leak_disable_tracking(void);
int arix_leak_reset(void);

void *arix_leak_malloc(size_t size, const char *file, int line, const char *func);
void arix_leak_free(void *ptr);
void *arix_leak_calloc(size_t nmemb, size_t size, const char *file, int line, const char *func);
void *arix_leak_realloc(void *ptr, size_t size, const char *file, int line, const char *func);

#endif
