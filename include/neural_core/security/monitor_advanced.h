#ifndef SNEPPX_MONITOR_ADVANCED_H
#define SNEPPX_MONITOR_ADVANCED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MON_MAX_REGIONS 128
#define SNEPPX_MON_MAX_EVENTS 1024
#define SNEPPX_MON_ML_FEATURES 8

/* Code segment tamper detection */
typedef struct {
    const void* code_addr;
    size_t code_size;
    uint8_t baseline_hash[32];
    int enabled;
} SNEPPXCodeTamperDetector;

int  SNEPPX_code_tamper_init(SNEPPXCodeTamperDetector* ctd, const void* addr, size_t size);
int  SNEPPX_code_tamper_check(SNEPPXCodeTamperDetector* ctd);

/* Function pointer hook detection */
typedef struct {
    const void** func_ptrs[64];
    uintptr_t original_values[64];
    int count;
} SNEPPXFuncPtrDetector;

int  SNEPPX_func_ptr_detector_init(SNEPPXFuncPtrDetector* fpd);
int  SNEPPX_func_ptr_detector_watch(SNEPPXFuncPtrDetector* fpd, const void** func_ptr);
int  SNEPPX_func_ptr_detector_scan(SNEPPXFuncPtrDetector* fpd);

/* Heap corruption detector */
typedef struct {
    uint64_t sentinel_value;
    int enabled;
} SNEPPXHeapCorruptionDetector;

int  SNEPPX_heap_corruption_init(SNEPPXHeapCorruptionDetector* hcd);
int  SNEPPX_heap_corruption_apply_sentinel(SNEPPXHeapCorruptionDetector* hcd, void* alloc, size_t size);
int  SNEPPX_heap_corruption_check(SNEPPXHeapCorruptionDetector* hcd, void* alloc, size_t size);

/* Stack overflow detection */
int  SNEPPX_stack_overflow_guard_install(void);
int  SNEPPX_stack_overflow_check(void);

/* Return address verification */
int  SNEPPX_ret_addr_verify(void* ret_addr, void* expected_ret_addr);

/* Instruction-level tracing (stub: platform-specific) */
typedef struct {
    int enabled;
    uint64_t trace_buffer_size;
    void* trace_buffer;
} SNEPPXInstructionTracer;

int  SNEPPX_inst_tracer_init(SNEPPXInstructionTracer* tracer);
int  SNEPPX_inst_tracer_start(SNEPPXInstructionTracer* tracer);
int  SNEPPX_inst_tracer_stop(SNEPPXInstructionTracer* tracer);

/* ML anomaly detector */
typedef struct {
    double means[SNEPPX_MON_ML_FEATURES];
    double stds[SNEPPX_MON_ML_FEATURES];
    int trained;
    double threshold;
} SNEPPXMLAnomalyDetector;

int  SNEPPX_ml_anomaly_init(SNEPPXMLAnomalyDetector* ml);
int  SNEPPX_ml_anomaly_train(SNEPPXMLAnomalyDetector* ml, const double features[][SNEPPX_MON_ML_FEATURES], int n_samples);
double SNEPPX_ml_anomaly_score(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES]);
int  SNEPPX_ml_anomaly_is_anomaly(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES]);

/* File system integrity */
typedef struct {
    char paths[64][256];
    uint8_t hashes[64][32];
    int count;
    int enabled;
} SNEPPXFSIntegrity;

int  SNEPPX_fs_integrity_init(SNEPPXFSIntegrity* fsi);
int  SNEPPX_fs_integrity_watch(SNEPPXFSIntegrity* fsi, const char* path);
int  SNEPPX_fs_integrity_scan(SNEPPXFSIntegrity* fsi);

/* Registry key monitoring (Windows) / file monitoring (Linux) */
int  SNEPPX_persistence_monitor_init(void);
int  SNEPPX_persistence_monitor_scan(void);

/* Process injection detection */
int  SNEPPX_proc_injection_detect(void);

/* Network connection monitoring */
int  SNEPPX_net_conn_monitor_init(void);
int  SNEPPX_net_conn_monitor_check(void);

/* USB/device insertion detection */
int  SNEPPX_device_insertion_detect(void);

/* Kernel object reference monitor */
int  SNEPPX_kernel_obj_monitor_init(void);
int  SNEPPX_kernel_obj_monitor_check(void);

/* TOCTOU detection */
typedef struct {
    uint8_t baseline[32];
    int initialized;
} SNEPPXTOCTOUDetector;

int  SNEPPX_toctou_init(SNEPPXTOCTOUDetector* td, const char* path);
int  SNEPPX_toctou_check(SNEPPXTOCTOUDetector* td, const char* path);

/* IMA-style integrity */
int  SNEPPX_ima_measure(const char* path, uint8_t hash[32]);
int  SNEPPX_ima_appraise(const char* path, const uint8_t hash[32]);

/* Alert correlation engine */
typedef struct {
    struct { uint64_t timestamp; int type; const char* desc; } events[SNEPPX_MON_MAX_EVENTS];
    int count;
    int alerts_triggered;
} SNEPPXAlertCorrelator;

int  SNEPPX_alert_correlator_init(SNEPPXAlertCorrelator* ac);
int  SNEPPX_alert_correlator_add(SNEPPXAlertCorrelator* ac, int type, const char* desc);
int  SNEPPX_alert_correlator_evaluate(SNEPPXAlertCorrelator* ac);

#ifdef __cplusplus
}
#endif
#endif
