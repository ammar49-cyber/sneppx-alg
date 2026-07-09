#ifndef SNEPPX_OBFUSCATION_ADVANCED_H
#define SNEPPX_OBFUSCATION_ADVANCED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_OBF_MAX_BINARY_OPS 256
#define SNEPPX_OBF_MAX_IAT_ENTRIES 128
#define SNEPPX_OBF_MAX_VM_SLOTS 8

/* Binary-level instruction substitution */
typedef struct {
    uint8_t original_opcode;
    uint8_t substitute_opcode;
    uint8_t prefix_bytes[4];
    int prefix_count;
    uint8_t suffix_bytes[4];
    int suffix_count;
} SNEPPXBinarySubstRule;

typedef struct {
    SNEPPXBinarySubstRule rules[SNEPPX_OBF_MAX_BINARY_OPS];
    int rule_count;
} SNEPPXBinarySubst;

int  SNEPPX_binary_subst_init(SNEPPXBinarySubst* bs);
int  SNEPPX_binary_subst_add_rule(SNEPPXBinarySubst* bs, uint8_t orig, uint8_t subst, const uint8_t* prefix, int pcount, const uint8_t* suffix, int scount);
int  SNEPPX_binary_subst_apply(SNEPPXBinarySubst* bs, uint8_t* code, size_t* code_len, size_t max_len);

/* Junk code generation */
typedef struct {
    uint8_t junk_code[64][16];
    int junk_count;
} SNEPPXJunkCodeGen;

int  SNEPPX_junk_code_init(SNEPPXJunkCodeGen* jcg);
int  SNEPPX_junk_code_add_pattern(SNEPPXJunkCodeGen* jcg, const uint8_t* pattern, size_t len);
int  SNEPPX_junk_code_insert(SNEPPXJunkCodeGen* jcg, uint8_t* code, size_t* code_len, size_t max_len, int position);

/* Constant unfolding */
int  SNEPPX_constant_unfold_int32(uint32_t value, uint8_t* expr_out, size_t* expr_len);
int  SNEPPX_constant_unfold_int64(uint64_t value, uint8_t* expr_out, size_t* expr_len);

/* Array dimension obfuscation */
int  SNEPPX_array_obfuscate_indices(const size_t* dims, int ndim, size_t* linearized, size_t* obfuscated_indices, int n_indices);

/* Bogus control flow */
typedef struct {
    uintptr_t fake_entry;
    uintptr_t real_entry;
} SNEPPXBogusCF;

int  SNEPPX_bogus_cf_init(SNEPPXBogusCF* bcf);
int  SNEPPX_bogus_cf_add_fake_block(SNEPPXBogusCF* bcf, const uint8_t* fake_code, size_t fake_len);
int  SNEPPX_bogus_cf_redirect(SNEPPXBogusCF* bcf, uint8_t* code, size_t code_len);

/* Anti-hook (IAT protection) */
typedef struct {
    struct { const char* name; void* original; void* current; } entries[SNEPPX_OBF_MAX_IAT_ENTRIES];
    int count;
} SNEPPXIATProtect;

int  SNEPPX_iat_protect_init(SNEPPXIATProtect* iat);
int  SNEPPX_iat_protect_add_entry(SNEPPXIATProtect* iat, const char* name, void* original);
int  SNEPPX_iat_protect_scan(SNEPPXIATProtect* iat);
int  SNEPPX_iat_protect_restore(SNEPPXIATProtect* iat);

/* White-box cryptography wrapper */
typedef struct {
    uint32_t te0[256],te1[256],te2[256],te3[256];
    uint32_t td0[256],td1[256],td2[256],td3[256];
    uint8_t embedded_key[16];
    int initialized;
} SNEPPXWhiteBoxAES;

int  SNEPPX_whitebox_aes_init(SNEPPXWhiteBoxAES* wb, const uint8_t key[16]);
void SNEPPX_whitebox_aes_encrypt(SNEPPXWhiteBoxAES* wb, const uint8_t in[16], uint8_t out[16]);

/* Import address table obfuscation */
typedef struct {
    uint32_t api_hashes[SNEPPX_OBF_MAX_IAT_ENTRIES];
    void* resolved_ptrs[SNEPPX_OBF_MAX_IAT_ENTRIES];
    int count;
} SNEPPXIATObfuscation;

int  SNEPPX_iat_obfuscation_init(SNEPPXIATObfuscation* io);
uint32_t SNEPPX_iat_hash_name(const char* name);
void* SNEPPX_iat_resolve_by_hash(SNEPPXIATObfuscation* io, uint32_t hash);

/* Exception handler obfuscation */
typedef struct {
    uintptr_t handler;
    uintptr_t next;
} SNEPPXSEHObfuscation;

int  SNEPPX_seh_obfuscation_init(SNEPPXSEHObfuscation* seh);
int  SNEPPX_seh_obfuscation_install(SNEPPXSEHObfuscation* seh, void* handler);

/* TLS callback obfuscation */
int  SNEPPX_tls_callback_register(void (*cb)(void*, int, void*));
int  SNEPPX_tls_callback_obfuscate(void);

/* Anti-dump */
typedef struct {
    uint8_t section_hash[32];
    uintptr_t image_base;
    size_t image_size;
    int is_protected;
} SNEPPXAntiDump;

int  SNEPPX_antidump_init(SNEPPXAntiDump* ad);
int  SNEPPX_antidump_protect(SNEPPXAntiDump* ad);
int  SNEPPX_antidump_verify(SNEPPXAntiDump* ad);

/* Multi-VM diversity */
typedef struct {
    uint8_t vm_slots[SNEPPX_OBF_MAX_VM_SLOTS][4096];
    size_t vm_sizes[SNEPPX_OBF_MAX_VM_SLOTS];
    int current_slot;
} SNEPPXMultiVM;

int  SNEPPX_multi_vm_init(SNEPPXMultiVM* mvm);
int  SNEPPX_multi_vm_switch(SNEPPXMultiVM* mvm);

/* Instruction scheduling */
int  SNEPPX_inst_schedule_randomize(uint8_t* code, size_t* code_len, size_t max_len);

#ifdef __cplusplus
}
#endif
#endif
