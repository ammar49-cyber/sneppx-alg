#ifndef SNEPPX_SIGNED_UPDATE_H
#define SNEPPX_SIGNED_UPDATE_H
/*
 * S7 Secure Updates — Signed Update Verification
 * Cryptographic verification of signed delta updates with rollback protection.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_UPDATE_SIG_LEN 64
#define SNEPPX_UPDATE_HASH_LEN 32
#define SNEPPX_MAX_UPDATE_CHUNKS 256

typedef struct {
    uint32_t chunk_index;
    uint8_t chunk_hash[SNEPPX_UPDATE_HASH_LEN];
    uint32_t chunk_size;
} SNEPPXUpdateChunk;

typedef struct {
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_patch;
    uint8_t signature[SNEPPX_UPDATE_SIG_LEN];
    uint8_t update_hash[SNEPPX_UPDATE_HASH_LEN];
    SNEPPXUpdateChunk chunks[SNEPPX_MAX_UPDATE_CHUNKS];
    int chunk_count;
    uint64_t timestamp;
    int is_delta;
} SNEPPXSignedUpdate;

typedef struct {
    uint32_t current_version[3];
    uint32_t min_allowed_version[3];
    int rollback_protection_enabled;
    int verification_enabled;
} SNEPPXUpdateVerifier;

int  SNEPPX_update_verifier_init(SNEPPXUpdateVerifier* uv);
void SNEPPX_update_verifier_destroy(SNEPPXUpdateVerifier* uv);
int  SNEPPX_update_verifier_set_min_version(SNEPPXUpdateVerifier* uv,
                                           uint32_t major, uint32_t minor, uint32_t patch);
int  SNEPPX_update_verifier_check(SNEPPXUpdateVerifier* uv, const SNEPPXSignedUpdate* update);
int  SNEPPX_update_verifier_apply(SNEPPXUpdateVerifier* uv, const SNEPPXSignedUpdate* update,
                                 const uint8_t* update_data, size_t data_len);
int  SNEPPX_update_verifier_rollback_check(SNEPPXUpdateVerifier* uv, uint32_t target_version[3]);

#ifdef __cplusplus
}
#endif
#endif
