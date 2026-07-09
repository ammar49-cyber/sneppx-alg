#ifndef SNEPPX_INTEGRITY_MONITOR_H
#define SNEPPX_INTEGRITY_MONITOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_MONITOR_EVENT_TEXT_MODIFIED,
    SNEPPX_MONITOR_EVENT_CANARY_TRIGGERED,
    SNEPPX_MONITOR_EVENT_FUNC_PTR_MODIFIED,
    SNEPPX_MONITOR_EVENT_HEAP_CORRUPTION,
    SNEPPX_MONITOR_EVENT_HEARTBEAT_MISS,
    SNEPPX_MONITOR_EVENT_SELF_TAMPER,
    SNEPPX_MONITOR_EVENT_PATTERN_FOUND,
    SNEPPX_MONITOR_EVENT_FREQ_ANOMALY,
} SNEPPXMonitorEventType;

typedef struct {
    SNEPPXMonitorEventType type;
    const char*          description;
    uint64_t             address;
    size_t               size;
    uint64_t             timestamp;
} SNEPPXMonitorEvent;

typedef void (*SNEPPXMonitorCallback)(const SNEPPXMonitorEvent* event);

int  SNEPPX_monitor_init(void);
void SNEPPX_monitor_shutdown(void);
int  SNEPPX_monitor_start(uint64_t interval_ms);
int  SNEPPX_monitor_stop(void);

int  SNEPPX_monitor_register_region(const char* name, const void* addr, size_t size);
int  SNEPPX_monitor_unregister_region(const char* name);

int  SNEPPX_monitor_verify_all(void);
int  SNEPPX_monitor_verify_region(const char* name);

int  SNEPPX_monitor_check_canary(void);
void SNEPPX_monitor_refresh_canary(void);

void SNEPPX_monitor_set_callback(SNEPPXMonitorCallback cb);

int  SNEPPX_monitor_freq_analyze(void);
void SNEPPX_monitor_freq_reset(void);

void SNEPPX_monitor_timing_set_baseline(double mean, double stddev);
int  SNEPPX_monitor_timing_check(uint64_t elapsed_us);

int  SNEPPX_monitor_api_hook_check(void);
void SNEPPX_monitor_api_hook_enable(const void* base, size_t size);

int  SNEPPX_monitor_syscall_track(int syscall_num);
int  SNEPPX_monitor_syscall_analyze(void);
void SNEPPX_monitor_syscall_learn_baseline(void);
void SNEPPX_monitor_syscall_enable(void);

int  SNEPPX_monitor_verify_single_region(const char* name);
int  SNEPPX_monitor_set_canary(int depth);
int  SNEPPX_monitor_check_canary_at(int depth);
int  SNEPPX_monitor_get_events(SNEPPXMonitorEvent* buffer, int max);
int  SNEPPX_monitor_scan_memory_for_pattern(const unsigned char* pattern, size_t pattern_len, const void* start, const void* end);
void SNEPPX_monitor_set_anomaly_threshold(int threshold);
int  SNEPPX_monitor_check_self(void);
int  SNEPPX_monitor_set_heartbeat(uint64_t interval_ms);
int  SNEPPX_monitor_add_callback(SNEPPXMonitorCallback cb);
int  SNEPPX_monitor_remove_callback(SNEPPXMonitorCallback cb);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_INTEGRITY_MONITOR_H */
