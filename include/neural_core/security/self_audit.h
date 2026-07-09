#ifndef SNEPPX_SELF_AUDIT_H
#define SNEPPX_SELF_AUDIT_H
/*
 * S9 Penetration Testing — Self-Audit Framework
 * Automated security assessment, vulnerability scanning, and
 * compliance verification.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_AUDIT_MAX_CHECKS 128
#define SNEPPX_CHECK_DESC_LEN 256

typedef enum {
    SNEPPX_AUDIT_PASS = 0,
    SNEPPX_AUDIT_FAIL = 1,
    SNEPPX_AUDIT_WARN = 2,
    SNEPPX_AUDIT_INFO = 3,
} SNEPPXAuditStatus;

typedef struct {
    char check_name[SNEPPX_CHECK_DESC_LEN];
    SNEPPXAuditStatus status;
    char details[SNEPPX_CHECK_DESC_LEN];
} SNEPPXAuditCheck;

typedef struct {
    SNEPPXAuditCheck checks[SNEPPX_AUDIT_MAX_CHECKS];
    int check_count;
    int total_passed;
    int total_failed;
    int total_warnings;
    double security_score;
} SNEPPXSelfAudit;

int  SNEPPX_self_audit_init(SNEPPXSelfAudit* audit);
void SNEPPX_self_audit_destroy(SNEPPXSelfAudit* audit);
int  SNEPPX_self_audit_add_check(SNEPPXSelfAudit* audit, const char* name,
                                 SNEPPXAuditStatus status, const char* details);
int  SNEPPX_self_audit_run_all(SNEPPXSelfAudit* audit);
int  SNEPPX_self_audit_run_category(SNEPPXSelfAudit* audit, const char* category);
double SNEPPX_self_audit_score(SNEPPXSelfAudit* audit);
int  SNEPPX_self_audit_export_report(SNEPPXSelfAudit* audit, const char* output_path);
int  SNEPPX_self_audit_check_crypto(SNEPPXSelfAudit* audit);
int  SNEPPX_self_audit_check_memory(SNEPPXSelfAudit* audit);
int  SNEPPX_self_audit_check_network(SNEPPXSelfAudit* audit);
int  SNEPPX_self_audit_check_ai_safety(SNEPPXSelfAudit* audit);

#ifdef __cplusplus
}
#endif
#endif
