#ifndef ARIX_RBTREE_H
#define ARIX_RBTREE_H
/*
 * Red-Black Tree — v0.5 (generic library)
 *
 * PURPOSE: Balanced binary search tree for memory region tracking
 * (secure allocator), timer scheduling, and ordered maps.
 *
 * Invariants:
 *   1. Every node is red or black.
 *   2. Root is black.
 *   3. Leaves (NULL) are black.
 *   4. Red nodes have black children.
 *   5. Every path from root to leaf has the same black height.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { ARIX_RBTREE_RED, ARIX_RBTREE_BLACK } ArixRBColor;

typedef struct ArixRBNode {
    struct ArixRBNode* parent;
    struct ArixRBNode* left;
    struct ArixRBNode* right;
    ArixRBColor        color;
    uint64_t           key;
    void*              value;
} ArixRBNode;

typedef struct {
    ArixRBNode* root;
    size_t      size;
    int         (*compare)(uint64_t a, uint64_t b);
} ArixRBTree;

ArixRBTree* arix_rbtree_create(void);
void        arix_rbtree_destroy(ArixRBTree* tree);

int   arix_rbtree_insert(ArixRBTree* tree, uint64_t key, void* value);
void* arix_rbtree_search(const ArixRBTree* tree, uint64_t key);
int   arix_rbtree_delete(ArixRBTree* tree, uint64_t key);

uint64_t arix_rbtree_min(const ArixRBTree* tree);
uint64_t arix_rbtree_max(const ArixRBTree* tree);
uint64_t arix_rbtree_successor(const ArixRBTree* tree, uint64_t key);
uint64_t arix_rbtree_predecessor(const ArixRBTree* tree, uint64_t key);

void arix_rbtree_foreach(const ArixRBTree* tree, void (*fn)(uint64_t key, void* value, void* ctx), void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_RBTREE_H */
