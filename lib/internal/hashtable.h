#ifndef ARIX_HASHTABLE_H
#define ARIX_HASHTABLE_H
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
} ArixHTEntry;

typedef struct {
    ArixHTEntry* buckets;
    size_t       capacity;
    size_t       count;
    float        load_factor_threshold;
    uint64_t     (*hash_fn)(uint64_t key);
} ArixHashTable;

ArixHashTable* arix_ht_create(size_t initial_capacity);
void           arix_ht_destroy(ArixHashTable* ht);

int   arix_ht_insert(ArixHashTable* ht, uint64_t key, void* value);
void* arix_ht_lookup(const ArixHashTable* ht, uint64_t key);
int   arix_ht_delete(ArixHashTable* ht, uint64_t key);

void  arix_ht_clear(ArixHashTable* ht);
int   arix_ht_resize(ArixHashTable* ht, size_t new_capacity);
void  arix_ht_foreach(const ArixHashTable* ht,
                      void (*fn)(uint64_t key, void* value, void* ctx), void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_HASHTABLE_H */
