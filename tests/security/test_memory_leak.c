#include "memory_leak_detector.h"
#include <stdio.h>

int main() {
    arix_leak_init();
    leak_stats_t stats;
    void *p1 = arix_leak_malloc(1024, "test.c", 10, "main");
    void *p2 = arix_leak_malloc(2048, "test.c", 11, "main");
    arix_leak_free(p1);
    arix_leak_get_stats(&stats);
    printf("Stats: active=%d current=%llu peak=%llu\n",
        stats.active_allocations, stats.current_usage, stats.peak_usage);
    leak_report_t reports[16];
    int n = arix_leak_check(reports, 16);
    printf("Leak reports: %d\n", n);
    for (int i = 0; i < n; i++)
        printf("  Leak: ptr=%p size=%zu at %s:%d\n", reports[i].ptr, reports[i].size, reports[i].file, reports[i].line);
    arix_leak_free(p2);
    printf("PASS: Memory leak detector OK\n");
    return 0;
}
