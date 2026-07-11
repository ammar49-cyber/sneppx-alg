#include "pqueue.h"
#include "polymorphic_memory_allocator.h"
#include <stdlib.h>

static int default_compare(uint64_t a, uint64_t b) {
    return (a < b) - (a > b);
}

SNEPPXPriorityQueue* SNEPPX_pq_create(size_t initial_capacity) {
    if (initial_capacity == 0) initial_capacity = 16;
    SNEPPXPriorityQueue* pq = (SNEPPXPriorityQueue*)SNEPPX_malloc(sizeof(SNEPPXPriorityQueue), 16);
    if (!pq) return NULL;
    pq->heap = (SNEPPXPQElement*)SNEPPX_malloc(initial_capacity * sizeof(SNEPPXPQElement), 16);
    if (!pq->heap) { SNEPPX_free(pq, sizeof(SNEPPXPriorityQueue)); return NULL; }
    pq->capacity = initial_capacity;
    pq->size = 0;
    pq->compare = default_compare;
    return pq;
}

void SNEPPX_pq_destroy(SNEPPXPriorityQueue* pq) {
    if (!pq) return;
    SNEPPX_free(pq->heap, pq->capacity * sizeof(SNEPPXPQElement));
    SNEPPX_free(pq, sizeof(SNEPPXPriorityQueue));
}

static void pq_sift_up(SNEPPXPriorityQueue* pq, size_t idx) {
    while (idx > 0) {
        size_t parent = (idx - 1) / 2;
        if (pq->compare(pq->heap[parent].priority, pq->heap[idx].priority) >= 0) break;
        SNEPPXPQElement tmp = pq->heap[parent];
        pq->heap[parent] = pq->heap[idx];
        pq->heap[idx] = tmp;
        idx = parent;
    }
}

static void pq_sift_down(SNEPPXPriorityQueue* pq, size_t idx) {
    while (1) {
        size_t left = 2 * idx + 1;
        size_t right = 2 * idx + 2;
        size_t largest = idx;
        
        if (left < pq->size && pq->compare(pq->heap[left].priority, pq->heap[largest].priority) > 0) largest = left;
        if (right < pq->size && pq->compare(pq->heap[right].priority, pq->heap[largest].priority) > 0) largest = right;
        if (largest == idx) break;
        
        SNEPPXPQElement tmp = pq->heap[idx];
        pq->heap[idx] = pq->heap[largest];
        pq->heap[largest] = tmp;
        idx = largest;
    }
}

int SNEPPX_pq_push(SNEPPXPriorityQueue* pq, uint64_t priority, void* data) {
    if (!pq || pq->size >= pq->capacity) return -1;
    pq->heap[pq->size] = (SNEPPXPQElement){priority, data};
    pq->size++;
    pq_sift_up(pq, pq->size - 1);
    return 0;
}

int SNEPPX_pq_pop(SNEPPXPriorityQueue* pq, uint64_t* priority, void** data) {
    if (!pq || pq->size == 0) return -1;
    *priority = pq->heap[0].priority;
    *data = pq->heap[0].data;
    pq->heap[0] = pq->heap[pq->size - 1];
    pq->size--;
    if (pq->size > 0) pq_sift_down(pq, 0);
    return 0;
}

int SNEPPX_pq_peek(const SNEPPXPriorityQueue* pq, uint64_t* priority, void** data) {
    if (!pq || pq->size == 0) return -1;
    *priority = pq->heap[0].priority;
    *data = pq->heap[0].data;
    return 0;
}

int SNEPPX_pq_is_empty(const SNEPPXPriorityQueue* pq) {
    return pq && pq->size == 0;
}

size_t SNEPPX_pq_size(const SNEPPXPriorityQueue* pq) {
    return pq ? pq->size : 0;
}

void SNEPPX_pq_heapify(SNEPPXPriorityQueue* pq, size_t idx) {
    if (!pq || idx >= pq->size) return;
    pq_sift_down(pq, idx);
    pq_sift_up(pq, idx);
}

void SNEPPX_pq_clear(SNEPPXPriorityQueue* pq) {
    if (pq) pq->size = 0;
}