#ifndef SNEPPX_MEMORY_LEAK_DETECTOR_H
#define SNEPPX_MEMORY_LEAK_DETECTOR_H

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

int SNEPPX_leak_init(void);
int SNEPPX_leak_track_alloc(void *ptr, size_t size, const char *file, int line, const char *func);
int SNEPPX_leak_track_free(void *ptr);
int SNEPPX_leak_check(leak_report_t *reports, int max_reports);
int SNEPPX_leak_get_stats(leak_stats_t *stats);
int SNEPPX_leak_set_threshold(int bytes);
int SNEPPX_leak_enable_tracking(void);
int SNEPPX_leak_disable_tracking(void);
int SNEPPX_leak_reset(void);

void *SNEPPX_leak_malloc(size_t size, const char *file, int line, const char *func);
void SNEPPX_leak_free(void *ptr);
void *SNEPPX_leak_calloc(size_t nmemb, size_t size, const char *file, int line, const char *func);
void *SNEPPX_leak_realloc(void *ptr, size_t size, const char *file, int line, const char *func);

#endif
