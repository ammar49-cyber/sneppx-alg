#ifndef SNEPPX_PQUEUE_H
#define SNEPPX_PQUEUE_H
/*
 * Priority Queue — v0.5 (generic library)
 *
 * PURPOSE: Binary max-heap for task scheduling (thread pool),
 * gradient compression priority, and timer management.
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
    uint64_t priority;
    void*    data;
} SNEPPXPQElement;

typedef struct {
    SNEPPXPQElement* heap;
    size_t         capacity;
    size_t         size;
    int            (*compare)(uint64_t a, uint64_t b);
} SNEPPXPriorityQueue;

SNEPPXPriorityQueue* SNEPPX_pq_create(size_t initial_capacity);
void               SNEPPX_pq_destroy(SNEPPXPriorityQueue* pq);

int   SNEPPX_pq_push(SNEPPXPriorityQueue* pq, uint64_t priority, void* data);
int   SNEPPX_pq_pop(SNEPPXPriorityQueue* pq, uint64_t* priority, void** data);
int   SNEPPX_pq_peek(const SNEPPXPriorityQueue* pq, uint64_t* priority, void** data);
int   SNEPPX_pq_is_empty(const SNEPPXPriorityQueue* pq);
size_t SNEPPX_pq_size(const SNEPPXPriorityQueue* pq);

void  SNEPPX_pq_heapify(SNEPPXPriorityQueue* pq, size_t idx);
void  SNEPPX_pq_clear(SNEPPXPriorityQueue* pq);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_PQUEUE_H */
