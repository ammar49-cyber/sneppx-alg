#ifndef ARIX_INTEGRITY_MONITOR_H
#define ARIX_INTEGRITY_MONITOR_H
/*
 * Runtime Integrity Monitor — v3.0 (security monitoring)
 *
 * PURPOSE: Monitors process memory for unauthorized modifications.
 * Periodically computes CRC32/CBC-MAC over text and read-only data
 * segments, checks stack canaries, and verifies that critical function
 * pointers (in driver ops tables, etc.) have not been overwritten.
 *
 * Alerts are delivered via a registered callback (log, crash, or
 * secure attestation).
 *
 * DEPENDENCIES: constant_time_operations.h, keccak_sha3_hashing.h
 * VERSION: v3.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARIX_MONITOR_EVENT_TEXT_MODIFIED,
    ARIX_MONITOR_EVENT_CANARY_TRIGGERED,
    ARIX_MONITOR_EVENT_FUNC_PTR_MODIFIED,
    ARIX_MONITOR_EVENT_HEAP_CORRUPTION,
} ArixMonitorEventType;

typedef struct {
    ArixMonitorEventType type;
    const char*          description;
    uint64_t             address;
    size_t               size;
    uint64_t             timestamp;
} ArixMonitorEvent;

typedef void (*ArixMonitorCallback)(const ArixMonitorEvent* event);

/* ---------- Monitor lifecycle ---------- */
int  arix_monitor_init(void);
void arix_monitor_shutdown(void);

int  arix_monitor_start(uint64_t interval_ms);
int  arix_monitor_stop(void);

/* ---------- Region registration ---------- */
int  arix_monitor_register_region(const char* name, const void* addr, size_t size);
int  arix_monitor_unregister_region(const char* name);

/* ---------- Checksum verification ---------- */
int  arix_monitor_verify_all(void);
int  arix_monitor_verify_region(const char* name);

/* ---------- Stack canary ---------- */
int  arix_monitor_check_canary(void);
void arix_monitor_refresh_canary(void);

/* ---------- Callback ---------- */
void arix_monitor_set_callback(ArixMonitorCallback cb);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_INTEGRITY_MONITOR_H */
