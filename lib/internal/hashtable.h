#ifndef SNEPPX_HASHTABLE_H
#define SNEPPX_HASHTABLE_H
/*
 * Hash Table — v0.5 (generic library)
 *
 * PURPOSE: Open-addressing hash table with quadratic probing for
 * tensor name lookups, parameter registries, and operator caches.
 *
 * Load factor threshold triggers automatic resize (2×).  Hash
 * function is pluggable via the `hash_fn` callback.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t hash;
    uint64_t key;
    void*    value;
    int      occupied;
    int      deleted;
} SNEPPXHTEntry;

typedef struct {
    SNEPPXHTEntry* buckets;
    size_t       capacity;
    size_t       count;
    float        load_factor_threshold;
    uint64_t     (*hash_fn)(uint64_t key);
} SNEPPXHashTable;

SNEPPXHashTable* SNEPPX_ht_create(size_t initial_capacity);
void           SNEPPX_ht_destroy(SNEPPXHashTable* ht);

int   SNEPPX_ht_insert(SNEPPXHashTable* ht, uint64_t key, void* value);
void* SNEPPX_ht_lookup(const SNEPPXHashTable* ht, uint64_t key);
int   SNEPPX_ht_delete(SNEPPXHashTable* ht, uint64_t key);

void  SNEPPX_ht_clear(SNEPPXHashTable* ht);
int   SNEPPX_ht_resize(SNEPPXHashTable* ht, size_t new_capacity);
void  SNEPPX_ht_foreach(const SNEPPXHashTable* ht,
                      void (*fn)(uint64_t key, void* value, void* ctx), void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HASHTABLE_H */
