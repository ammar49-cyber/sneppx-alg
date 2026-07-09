#ifndef SNEPPX_KEY_VAULT_H
#define SNEPPX_KEY_VAULT_H
/*
 * S6 Security UI — Key Management Vault
 * Secure key generation, storage, rotation, and access auditing.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_VAULT_MAX_KEYS 64
#define SNEPPX_VAULT_KEY_LEN 32
#define SNEPPX_VAULT_ID_LEN 16

typedef struct {
    uint8_t id[SNEPPX_VAULT_ID_LEN];
    uint8_t key_data[SNEPPX_VAULT_KEY_LEN];
    uint64_t created_at;
    uint64_t expires_at;
    int is_active;
    int access_count;
} SNEPPXVaultKey;

typedef struct {
    SNEPPXVaultKey keys[SNEPPX_VAULT_MAX_KEYS];
    int key_count;
    int is_locked;
} SNEPPXKeyVault;

int  SNEPPX_key_vault_init(SNEPPXKeyVault* vault);
void SNEPPX_key_vault_destroy(SNEPPXKeyVault* vault);
int  SNEPPX_key_vault_generate_key(SNEPPXKeyVault* vault, uint8_t* key_id, uint64_t ttl_seconds);
int  SNEPPX_key_vault_get_key(SNEPPXKeyVault* vault, const uint8_t* key_id, uint8_t* key_out);
int  SNEPPX_key_vault_rotate_key(SNEPPXKeyVault* vault, const uint8_t* key_id);
int  SNEPPX_key_vault_revoke_key(SNEPPXKeyVault* vault, const uint8_t* key_id);
int  SNEPPX_key_vault_lock(SNEPPXKeyVault* vault);
int  SNEPPX_key_vault_unlock(SNEPPXKeyVault* vault, const uint8_t* master_key);

#ifdef __cplusplus
}
#endif
#endif
