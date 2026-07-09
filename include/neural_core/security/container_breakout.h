#ifndef SNEPPX_CONTAINER_BREAKOUT_H
#define SNEPPX_CONTAINER_BREAKOUT_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint64_t total_breakout_attempts;
    uint64_t total_blocks;
    int active_processes;
    int suspicious_processes;
    int watch_files;
} breakout_stats_t;

int SNEPPX_breakout_detect_ns_change(int pid, const char *comm);
int SNEPPX_breakout_detect_mount(int pid, const char *target, const char *comm);
int SNEPPX_breakout_detect_capability(int pid, uint64_t cap_effective, const char *comm);
int SNEPPX_breakout_check_file_access(const char *path);
int SNEPPX_breakout_detect_syscall(int pid, const char *syscall_name, const char *comm);
int SNEPPX_breakout_add_watch_file(const char *path);
int SNEPPX_breakout_get_stats(breakout_stats_t *stats);
int SNEPPX_breakout_reset(void);

#endif
