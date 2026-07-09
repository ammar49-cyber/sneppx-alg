#ifndef SNEPPX_S7_EXTENSIONS_H
#define SNEPPX_S7_EXTENSIONS_H
/* S7 extensions: TUF compliance, bsdiff delta, A/B partitions, manifest
   verification, TPM attestation, canary rollout, offline bundles, dependency resolver */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_TUF_MAX_KEYS 8
#define TPM_PCR_COUNT 24

/* TUF (The Update Framework) */
typedef struct {
    uint8_t root_key[32];
    uint8_t targets_key[32];
    uint8_t snapshot_key[32];
    uint8_t timestamp_key[32];
    int initialized;
} SNEPPXTUFMetadata;

int  SNEPPX_tuf_init(SNEPPXTUFMetadata* tuf);
int  SNEPPX_tuf_sign_root(SNEPPXTUFMetadata* tuf, const uint8_t* data, size_t len, uint8_t* sig, size_t* sig_len);
int  SNEPPX_tuf_verify_targets(SNEPPXTUFMetadata* tuf, const uint8_t* targets_json, size_t len);

/* bsdiff delta generation */
int  SNEPPX_bsdiff(const uint8_t* old_data, size_t old_len, const uint8_t* new_data, size_t new_len, uint8_t* patch, size_t* patch_len);
int  SNEPPX_bspatch(const uint8_t* old_data, size_t old_len, const uint8_t* patch, size_t patch_len, uint8_t* new_data, size_t* new_len);

/* A/B partition management */
typedef struct {
    int active_slot;
    int inactive_slot;
    uint8_t slot_a_hash[32];
    uint8_t slot_b_hash[32];
    int swap_ready;
} SNEPPXABPartition;

int  SNEPPX_ab_partition_init(SNEPPXABPartition* ab);
int  SNEPPX_ab_partition_mark_good(SNEPPXABPartition* ab, int slot);
int  SNEPPX_ab_partition_swap(SNEPPXABPartition* ab);

/* Manifest verification */
int  SNEPPX_manifest_verify(const char* manifest_path, const uint8_t* signature, size_t sig_len);

/* TPM attestation */
int  SNEPPX_tpm_pcr_read(int pcr_index, uint8_t* out, size_t* out_len);
int  SNEPPX_tpm_quote(const uint8_t* nonce, size_t nonce_len, uint8_t* quote, size_t* quote_len);

/* Canary rollout */
typedef struct {
    int total_nodes;
    int canary_nodes;
    int promoted;
} SNEPPXCanaryRollout;

int  SNEPPX_canary_rollout_init(SNEPPXCanaryRollout* cr, int total, int canary);
int  SNEPPX_canary_rollout_promote(SNEPPXCanaryRollout* cr);

/* Offline update bundle */
typedef struct {
    uint8_t bundle_hash[32];
    size_t bundle_size;
    int signed_offline;
} SNEPPXOfflineBundle;

int  SNEPPX_offline_bundle_create(SNEPPXOfflineBundle* ob, const uint8_t* data, size_t data_len, const uint8_t* signing_key, size_t key_len);

/* Dependency resolver */
typedef struct {
    char name[64];
    uint32_t version_major, version_minor, version_patch;
    int resolved;
} SNEPPXDepResolver;

int  SNEPPX_dep_resolver_init(SNEPPXDepResolver* dr);
int  SNEPPX_dep_resolver_add_dep(SNEPPXDepResolver* dr, const char* name, uint32_t maj, uint32_t min, uint32_t pat);
int  SNEPPX_dep_resolver_resolve(SNEPPXDepResolver* dr);

#ifdef __cplusplus
}
#endif
#endif
