#ifndef SNEPPX_S6_EXTENSIONS_H
#define SNEPPX_S6_EXTENSIONS_H
/* S6 extensions: HSM-backed keys, Shamir's Secret Sharing, key ceremony,
   auto rotation scheduler, web dashboard, threat viz, policy editor DSL,
   compliance report generator */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_SHAMIR_MAX_SHARES 16
#define SNEPPX_SHAMIR_MIN_SHARES 5

/* HSM-backed key storage */
typedef struct {
    int hsm_connected;
    uint8_t session_handle[32];
} SNEPPXHSMKeyStore;

int  SNEPPX_hsm_init(SNEPPXHSMKeyStore* hsm);
int  SNEPPX_hsm_store_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id, const uint8_t* key_data, size_t key_len);
int  SNEPPX_hsm_load_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id, uint8_t* key_data, size_t* key_len);
int  SNEPPX_hsm_delete_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id);

/* Shamir's Secret Sharing */
int  SNEPPX_shamir_split(const uint8_t* secret, size_t secret_len, int n, int k, uint8_t** shares, size_t* share_lens);
int  SNEPPX_shamir_reconstruct(uint8_t** shares, size_t* share_lens, int k, uint8_t* secret, size_t* secret_len);

/* Key ceremony workflow */
typedef struct {
    int participants_required;
    int participants_present;
    int approved;
} SNEPPXKeyCeremony;

int  SNEPPX_key_ceremony_init(SNEPPXKeyCeremony* kc, int participants);
int  SNEPPX_key_ceremony_participant_approve(SNEPPXKeyCeremony* kc, const uint8_t* token, size_t token_len);
int  SNEPPX_key_ceremony_execute(SNEPPXKeyCeremony* kc, uint8_t* generated_key, size_t key_len);

/* Auto rotation scheduler */
typedef struct {
    uint64_t rotation_interval_seconds;
    uint64_t last_rotation;
    int auto_rotate;
} SNEPPXKeyRotationScheduler;

int  SNEPPX_key_rotation_init(SNEPPXKeyRotationScheduler* ks, uint64_t interval_seconds);
int  SNEPPX_key_rotation_check(SNEPPXKeyRotationScheduler* ks);

/* Web dashboard stub */
int  SNEPPX_security_dashboard_init(const char* listen_addr, int port);
int  SNEPPX_security_dashboard_update(const char* json_payload);

/* Threat visualization */
int  SNEPPX_threat_viz_init(void);
int  SNEPPX_threat_viz_add_edge(const char* from, const char* to, const char* label);

/* Policy editor DSL */
typedef struct {
    char rules[32][256];
    int rule_count;
} SNEPPXPolicyDSL;

int  SNEPPX_policy_dsl_init(SNEPPXPolicyDSL* dsl);
int  SNEPPX_policy_dsl_add_rule(SNEPPXPolicyDSL* dsl, const char* rule);
int  SNEPPX_policy_dsl_compile(SNEPPXPolicyDSL* dsl, uint8_t* bytecode, size_t* bytecode_len);

/* Compliance report generator */
int  SNEPPX_compliance_report(const char* report_type, const char* output_path);

#ifdef __cplusplus
}
#endif
#endif
