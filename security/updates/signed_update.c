#include "signed_update.h"
#include "cryptographic_hashing_blake3.h"
#include "constant_time_operations.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define SNEPPX_UPDATE_HISTORY_MAX 10

typedef struct {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    uint64_t timestamp;
} SNEPPXUpdateHistoryEntry;

static SNEPPXUpdateHistoryEntry update_history[SNEPPX_UPDATE_HISTORY_MAX];
static int update_history_count = 0;
static int rollback_protection_global = 1;

static int canary_percentage = 100;

int SNEPPX_update_verifier_init(SNEPPXUpdateVerifier* uv) {
    if (!uv) return -1;
    memset(uv, 0, sizeof(*uv));
    uv->current_version[0] = 1;
    uv->current_version[1] = 0;
    uv->current_version[2] = 0;
    uv->min_allowed_version[0] = 1;
    uv->min_allowed_version[1] = 0;
    uv->min_allowed_version[2] = 0;
    uv->rollback_protection_enabled = 1;
    uv->verification_enabled = 1;
    rollback_protection_global = 1;
    update_history_count = 0;
    memset(update_history, 0, sizeof(update_history));
    canary_percentage = 100;
    return 0;
}

void SNEPPX_update_verifier_destroy(SNEPPXUpdateVerifier* uv) {
    if (uv) memset(uv, 0, sizeof(*uv));
}

int SNEPPX_update_verifier_set_min_version(SNEPPXUpdateVerifier* uv,
                                           uint32_t major, uint32_t minor, uint32_t patch) {
    if (!uv) return -1;
    uv->min_allowed_version[0] = major;
    uv->min_allowed_version[1] = minor;
    uv->min_allowed_version[2] = patch;
    return 0;
}

int SNEPPX_update_verifier_check(SNEPPXUpdateVerifier* uv, const SNEPPXSignedUpdate* update) {
    if (!uv || !update) return 0;
    if (!uv->verification_enabled) return 1;

    if (uv->rollback_protection_enabled) {
        uint32_t target[3] = {update->version_major, update->version_minor, update->version_patch};
        if (SNEPPX_update_verifier_rollback_check(uv, target) != 0) return 0;
        if (target[0] < uv->min_allowed_version[0]) return 0;
        if (target[0] == uv->min_allowed_version[0] && target[1] < uv->min_allowed_version[1]) return 0;
        if (target[0] == uv->min_allowed_version[0] && target[1] == uv->min_allowed_version[1] && target[2] < uv->min_allowed_version[2]) return 0;
    }

    SNEPPXBlake3State ctx;
    uint8_t computed_hash[SNEPPX_UPDATE_HASH_LEN];
    SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx, (const uint8_t*)&update->version_major, sizeof(uint32_t) * 3);
    SNEPPX_blake3_update(&ctx, (const uint8_t*)&update->chunk_count, sizeof(int));
    SNEPPX_blake3_update(&ctx, (const uint8_t*)&update->timestamp, sizeof(uint64_t));
    SNEPPX_blake3_finish(&ctx, computed_hash);

    int sig_ok = SNEPPX_ct_equal(computed_hash, update->update_hash, SNEPPX_UPDATE_HASH_LEN) ? 1 : 0;
    return sig_ok;
}

int SNEPPX_update_verifier_apply(SNEPPXUpdateVerifier* uv, const SNEPPXSignedUpdate* update,
                                 const uint8_t* update_data, size_t data_len) {
    (void)update_data; (void)data_len;
    if (!uv || !update) return -1;
    if (!SNEPPX_update_verifier_check(uv, update)) return -1;
    uv->current_version[0] = update->version_major;
    uv->current_version[1] = update->version_minor;
    uv->current_version[2] = update->version_patch;
    if (update_history_count < SNEPPX_UPDATE_HISTORY_MAX) {
        update_history[update_history_count].major = update->version_major;
        update_history[update_history_count].minor = update->version_minor;
        update_history[update_history_count].patch = update->version_patch;
        update_history[update_history_count].timestamp = (uint64_t)time(NULL);
        update_history_count++;
    }
    return 0;
}

int SNEPPX_update_verifier_rollback_check(SNEPPXUpdateVerifier* uv, uint32_t target_version[3]) {
    if (!uv || !target_version) return -1;
    if (!uv->rollback_protection_enabled) return 0;
    if (target_version[0] < uv->current_version[0]) return 1;
    if (target_version[0] == uv->current_version[0] && target_version[1] < uv->current_version[1]) return 1;
    if (target_version[0] == uv->current_version[0] && target_version[1] == uv->current_version[1] && target_version[2] < uv->current_version[2]) return 1;
    return 0;
}

int SNEPPX_update_verifier_get_current_version(SNEPPXUpdateVerifier* uv, uint32_t* major, uint32_t* minor, uint32_t* patch) {
    if (!uv || !major || !minor || !patch) return -1;
    *major = uv->current_version[0];
    *minor = uv->current_version[1];
    *patch = uv->current_version[2];
    return 0;
}

int SNEPPX_update_verifier_get_min_version(SNEPPXUpdateVerifier* uv, uint32_t* major, uint32_t* minor, uint32_t* patch) {
    if (!uv || !major || !minor || !patch) return -1;
    *major = uv->min_allowed_version[0];
    *minor = uv->min_allowed_version[1];
    *patch = uv->min_allowed_version[2];
    return 0;
}

int SNEPPX_update_verifier_set_rollback_protection(int enabled) {
    rollback_protection_global = enabled;
    return 0;
}

int SNEPPX_update_verifier_get_update_history(SNEPPXUpdateVerifier* uv, uint32_t* history, int max) {
    if (!uv || !history || max <= 0) return -1;
    int written = 0;
    for (int i = 0; i < update_history_count && written < max; i++) {
        history[written * 3 + 0] = update_history[i].major;
        history[written * 3 + 1] = update_history[i].minor;
        history[written * 3 + 2] = update_history[i].patch;
        written++;
    }
    return written;
}

int SNEPPX_update_verifier_sign_update(SNEPPXSignedUpdate* update, const uint8_t* signing_key, size_t key_len) {
    if (!update || !signing_key) return -1;
    SNEPPXBlake3State ctx;
    uint8_t hash[SNEPPX_UPDATE_HASH_LEN];
    SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx, (const uint8_t*)&update->version_major, sizeof(uint32_t) * 3);
    SNEPPX_blake3_update(&ctx, (const uint8_t*)&update->chunk_count, sizeof(int));
    SNEPPX_blake3_update(&ctx, (const uint8_t*)&update->timestamp, sizeof(uint64_t));
    SNEPPX_blake3_update(&ctx, signing_key, key_len);
    SNEPPX_blake3_finish(&ctx, hash);
    memcpy(update->update_hash, hash, SNEPPX_UPDATE_HASH_LEN);
    memset(update->signature, 0, SNEPPX_UPDATE_SIG_LEN);
    for (size_t i = 0; i < SNEPPX_UPDATE_SIG_LEN && i < SNEPPX_UPDATE_HASH_LEN; i++)
        update->signature[i] = hash[i] ^ 0xAA;
    return 0;
}

int SNEPPX_update_verifier_verify_signature(const SNEPPXSignedUpdate* update, const uint8_t* public_key, size_t key_len) {
    (void)public_key; (void)key_len;
    if (!update) return -1;
    uint8_t expected_hash[SNEPPX_UPDATE_HASH_LEN];
    for (size_t i = 0; i < SNEPPX_UPDATE_HASH_LEN && i < SNEPPX_UPDATE_SIG_LEN; i++)
        expected_hash[i] = update->signature[i] ^ 0xAA;
    for (size_t i = SNEPPX_UPDATE_HASH_LEN; i < SNEPPX_UPDATE_SIG_LEN; i++)
        expected_hash[i] = 0;
    return (memcmp(expected_hash, update->update_hash, SNEPPX_UPDATE_HASH_LEN) == 0) ? 0 : -1;
}

int SNEPPX_update_verifier_get_version_string(char* buf, size_t size) {
    if (!buf || size == 0) return -1;
    snprintf(buf, size, "1.0.%d", (int)time(NULL) % 100);
    return 0;
}

int SNEPPX_update_verifier_get_percentage(void) {
    return canary_percentage;
}

int SNEPPX_update_verifier_record_history(const SNEPPXSignedUpdate* update) {
    if (!update) return -1;
    if (update_history_count >= SNEPPX_UPDATE_HISTORY_MAX) {
        memmove(update_history, update_history + 1, (SNEPPX_UPDATE_HISTORY_MAX - 1) * sizeof(SNEPPXUpdateHistoryEntry));
        update_history_count--;
    }
    update_history[update_history_count].major = update->version_major;
    update_history[update_history_count].minor = update->version_minor;
    update_history[update_history_count].patch = update->version_patch;
    update_history[update_history_count].timestamp = (uint64_t)time(NULL);
    update_history_count++;
    return 0;
}

int SNEPPX_update_verifier_get_history(uint32_t* entries, int max) {
    if (!entries || max <= 0) return -1;
    int written = 0;
    for (int i = 0; i < update_history_count && written < max; i++) {
        entries[written * 4 + 0] = update_history[i].major;
        entries[written * 4 + 1] = update_history[i].minor;
        entries[written * 4 + 2] = update_history[i].patch;
        entries[written * 4 + 3] = (uint32_t)update_history[i].timestamp;
        written++;
    }
    return written;
}
static int compare_versions(uint32_t a[3], uint32_t b[3]) {
    if (a[0] != b[0]) return (a[0] < b[0]) ? -1 : 1;
    if (a[1] != b[1]) return (a[1] < b[1]) ? -1 : 1;
    if (a[2] != b[2]) return (a[2] < b[2]) ? -1 : 1;
    return 0;
}

static void version_from_update(uint32_t* out, const SNEPPXSignedUpdate* update) {
    out[0] = update->version_major;
    out[1] = update->version_minor;
    out[2] = update->version_patch;
}

int SNEPPX_update_verifier_is_update_applicable(SNEPPXUpdateVerifier* uv, const SNEPPXSignedUpdate* update) {
    if (!uv || !update) return 0;
    uint32_t target[3] = {update->version_major, update->version_minor, update->version_patch};
    if (compare_versions(target, uv->current_version) <= 0) return 0;
    if (compare_versions(target, uv->min_allowed_version) < 0) return 0;
    return 1;
}

int SNEPPX_update_verifier_set_current_version(SNEPPXUpdateVerifier* uv, uint32_t major, uint32_t minor, uint32_t patch) {
    if (!uv) return -1;
    uv->current_version[0] = major;
    uv->current_version[1] = minor;
    uv->current_version[2] = patch;
    return 0;
}

int SNEPPX_update_verifier_enable_verification(SNEPPXUpdateVerifier* uv, int enable) {
    if (!uv) return -1;
    uv->verification_enabled = (enable != 0);
    return 0;
}

int SNEPPX_update_verifier_is_verification_enabled(SNEPPXUpdateVerifier* uv) {
    return uv ? uv->verification_enabled : 0;
}

int SNEPPX_update_verifier_is_rollback_protection_enabled(SNEPPXUpdateVerifier* uv) {
    if (!uv) return rollback_protection_global;
    return uv->rollback_protection_enabled;
}

int SNEPPX_update_verifier_get_history_count(void) {
    return update_history_count;
}

void SNEPPX_update_verifier_clear_history(void) {
    memset(update_history, 0, sizeof(update_history));
    update_history_count = 0;
}

uint64_t SNEPPX_update_verifier_get_history_timestamp(int index) {
    if (index < 0 || index >= update_history_count) return 0;
    return update_history[index].timestamp;
}

int SNEPPX_update_verifier_set_canary_percentage(int pct) {
    if (pct < 0 || pct > 100) return -1;
    canary_percentage = pct;
    return 0;
}
int SNEPPX_update_verifier_check_version_major(SNEPPXUpdateVerifier* uv, uint32_t major) {
    if (!uv) return 0;
    return (major >= uv->min_allowed_version[0] && major <= uv->current_version[0] + 1) ? 1 : 0;
}

int SNEPPX_update_verifier_check_version_minor(SNEPPXUpdateVerifier* uv, uint32_t minor) {
    if (!uv) return 0;
    return (minor >= uv->current_version[1]) ? 1 : 0;
}

int SNEPPX_update_verifier_check_version_patch(SNEPPXUpdateVerifier* uv, uint32_t patch) {
    if (!uv) return 0;
    return (patch >= uv->current_version[2]) ? 1 : 0;
}

int SNEPPX_update_verifier_get_history_entry(uint32_t index, uint32_t* major, uint32_t* minor, uint32_t* patch, uint64_t* timestamp) {
    if (index >= (uint32_t)update_history_count || !major || !minor || !patch || !timestamp) return -1;
    *major = update_history[index].major;
    *minor = update_history[index].minor;
    *patch = update_history[index].patch;
    *timestamp = update_history[index].timestamp;
    return 0;
}

int SNEPPX_update_verifier_compare_versions(uint32_t a_major, uint32_t a_minor, uint32_t a_patch, uint32_t b_major, uint32_t b_minor, uint32_t b_patch) {
    if (a_major != b_major) return (a_major < b_major) ? -1 : 1;
    if (a_minor != b_minor) return (a_minor < b_minor) ? -1 : 1;
    if (a_patch != b_patch) return (a_patch < b_patch) ? -1 : 1;
    return 0;
}

int SNEPPX_update_verifier_is_delta_update(const SNEPPXSignedUpdate* update) {
    return update ? update->is_delta : 0;
}

uint64_t SNEPPX_update_verifier_get_update_timestamp(const SNEPPXSignedUpdate* update) {
    return update ? update->timestamp : 0;
}

int SNEPPX_update_verifier_get_chunk_count(const SNEPPXSignedUpdate* update) {
    return update ? update->chunk_count : 0;
}
int SNEPPX_update_verifier_get_rollback_protection(void) { return rollback_protection_global; }
void SNEPPX_update_verifier_set_history_count(int c) { update_history_count = c; }

int SNEPPX_update_verifier_get_signing_key(SNEPPXUpdateVerifier* uv, uint8_t* key_out, size_t key_len) {
    if (!uv || !key_out || key_len < 32) return -1;
    (void)uv;
    memset(key_out, 0, key_len);
    return 0;
}

int SNEPPX_update_verifier_set_signing_key(SNEPPXUpdateVerifier* uv, const uint8_t* key, size_t key_len) {
    if (!uv || !key) return -1;
    (void)key;
    (void)key_len;
    return 0;
}

int SNEPPX_update_verifier_compute_hash(const uint8_t* data, size_t len, uint8_t* hash_out) {
    if (!data || !hash_out) return -1;
    SNEPPXBlake3State ctx;
    SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx, data, len);
    SNEPPX_blake3_finish(&ctx, hash_out);
    return 0;
}

int SNEPPX_update_verifier_verify_integrity(SNEPPXUpdateVerifier* uv, const uint8_t* data, size_t len, const uint8_t* signature, size_t sig_len) {
    (void)uv;
    if (!data || !signature) return -1;
    uint8_t hash[SNEPPX_UPDATE_HASH_LEN];
    SNEPPX_update_verifier_compute_hash(data, len, hash);
    for (size_t i = 0; i < sig_len && i < SNEPPX_UPDATE_HASH_LEN; i++) {
        if ((signature[i] ^ 0xAA) != hash[i]) return -1;
    }
    return 0;
}

int SNEPPX_update_verifier_get_version(SNEPPXUpdateVerifier* uv, uint32_t* major, uint32_t* minor, uint32_t* patch) {
    return SNEPPX_update_verifier_get_current_version(uv, major, minor, patch);
}

int SNEPPX_update_verifier_is_update_available(SNEPPXUpdateVerifier* uv, uint32_t candidate_major, uint32_t candidate_minor, uint32_t candidate_patch) {
    if (!uv) return 0;
    uint32_t cm[3] = {candidate_major, candidate_minor, candidate_patch};
    if (compare_versions(cm, uv->current_version) <= 0) return 0;
    if (compare_versions(cm, uv->min_allowed_version) < 0) return 0;
    return 1;
}
