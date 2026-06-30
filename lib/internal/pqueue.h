#ifndef ARIX_PQUEUE_H
#define ARIX_PQUEUE_H
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
} ArixPQElement;

typedef struct {
    ArixPQElement* heap;
    size_t         capacity;
    size_t         size;
    int            (*compare)(uint64_t a, uint64_t b);
} ArixPriorityQueue;

ArixPriorityQueue* arix_pq_create(size_t initial_capacity);
void               arix_pq_destroy(ArixPriorityQueue* pq);

int   arix_pq_push(ArixPriorityQueue* pq, uint64_t priority, void* data);
int   arix_pq_pop(ArixPriorityQueue* pq, uint64_t* priority, void** data);
int   arix_pq_peek(const ArixPriorityQueue* pq, uint64_t* priority, void** data);
int   arix_pq_is_empty(const ArixPriorityQueue* pq);
size_t arix_pq_size(const ArixPriorityQueue* pq);

void  arix_pq_heapify(ArixPriorityQueue* pq, size_t idx);
void  arix_pq_clear(ArixPriorityQueue* pq);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_PQUEUE_H */
