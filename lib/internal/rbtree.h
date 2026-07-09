#ifndef SNEPPX_RBTREE_H
#define SNEPPX_RBTREE_H
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

typedef enum { SNEPPX_RBTREE_RED, SNEPPX_RBTREE_BLACK } SNEPPXRBColor;

typedef struct SNEPPXRBNode {
    struct SNEPPXRBNode* parent;
    struct SNEPPXRBNode* left;
    struct SNEPPXRBNode* right;
    SNEPPXRBColor        color;
    uint64_t           key;
    void*              value;
} SNEPPXRBNode;

typedef struct {
    SNEPPXRBNode* root;
    size_t      size;
    int         (*compare)(uint64_t a, uint64_t b);
} SNEPPXRBTree;

SNEPPXRBTree* SNEPPX_rbtree_create(void);
void        SNEPPX_rbtree_destroy(SNEPPXRBTree* tree);

int   SNEPPX_rbtree_insert(SNEPPXRBTree* tree, uint64_t key, void* value);
void* SNEPPX_rbtree_search(const SNEPPXRBTree* tree, uint64_t key);
int   SNEPPX_rbtree_delete(SNEPPXRBTree* tree, uint64_t key);

uint64_t SNEPPX_rbtree_min(const SNEPPXRBTree* tree);
uint64_t SNEPPX_rbtree_max(const SNEPPXRBTree* tree);
uint64_t SNEPPX_rbtree_successor(const SNEPPXRBTree* tree, uint64_t key);
uint64_t SNEPPX_rbtree_predecessor(const SNEPPXRBTree* tree, uint64_t key);

void SNEPPX_rbtree_foreach(const SNEPPXRBTree* tree, void (*fn)(uint64_t key, void* value, void* ctx), void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_RBTREE_H */
