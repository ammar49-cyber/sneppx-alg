#ifndef SNEPPX_IDENTITY_MANAGEMENT_H
#define SNEPPX_IDENTITY_MANAGEMENT_H
/*
 * S4 Network Security — Identity & Access Management
 * Certificate pinning, certificate validation, DDoS protection.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MAX_PINNED_CERTS 16
#define SNEPPX_CERT_FINGERPRINT_LEN 32

typedef struct {
    uint8_t fingerprint[SNEPPX_CERT_FINGERPRINT_LEN];
    char subject[256];
    uint64_t expiry;
    int is_active;
} SNEPPXPinnedCert;

typedef struct {
    SNEPPXPinnedCert certs[SNEPPX_MAX_PINNED_CERTS];
    int cert_count;
    int ddos_protection_enabled;
    uint64_t ddos_request_limit;
    uint64_t ddos_window_ms;
    uint64_t ddos_current_count;
    uint64_t ddos_window_start;
} SNEPPXIdentityManager;

int  SNEPPX_identity_init(SNEPPXIdentityManager* mgr);
void SNEPPX_identity_shutdown(SNEPPXIdentityManager* mgr);
int  SNEPPX_identity_pin_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint,
                             const char* subject, uint64_t expiry);
int  SNEPPX_identity_verify_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint);
int  SNEPPX_identity_unpin_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint);
int  SNEPPX_identity_ddos_check(SNEPPXIdentityManager* mgr);
void SNEPPX_identity_ddos_reset(SNEPPXIdentityManager* mgr);
int  SNEPPX_identity_tls_verify(const char* hostname, const uint8_t* cert_der, size_t cert_len);

#ifdef __cplusplus
}
#endif
#endif
