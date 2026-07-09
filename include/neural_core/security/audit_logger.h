#ifndef SNEPPX_AUDIT_LOGGER_H
#define SNEPPX_AUDIT_LOGGER_H
/*
 * S6 Security UI — Audit Logger
 * Tamper-evident audit log for security events, key access, and violations.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_AUDIT_MAX_ENTRIES 1024
#define SNEPPX_AUDIT_DESC_LEN 256

typedef struct {
    uint64_t timestamp;
    int event_type;
    char description[SNEPPX_AUDIT_DESC_LEN];
    uint64_t related_address;
    uint32_t crc;
} SNEPPXAuditEntry;

typedef struct {
    SNEPPXAuditEntry entries[SNEPPX_AUDIT_MAX_ENTRIES];
    int entry_count;
    int enabled;
    const char* log_file_path;
    uint32_t chain_crc;
} SNEPPXAuditLogger;

int  SNEPPX_audit_init(SNEPPXAuditLogger* audit, const char* log_path);
void SNEPPX_audit_shutdown(SNEPPXAuditLogger* audit);
int  SNEPPX_audit_log(SNEPPXAuditLogger* audit, int event_type,
                     const char* description, uint64_t related_address);
int  SNEPPX_audit_export(SNEPPXAuditLogger* audit, const char* output_path);
int  SNEPPX_audit_verify_chain(SNEPPXAuditLogger* audit);
int  SNEPPX_audit_search(SNEPPXAuditLogger* audit, int event_type,
                        SNEPPXAuditEntry* results, int max_results);

#ifdef __cplusplus
}
#endif
#endif
