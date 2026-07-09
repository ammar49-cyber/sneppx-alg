/*
 * Virtual Memory Implementation — SKELETON
 * VERSION: v0.5
 */

#include "vmem.h"
#include <stdlib.h>
#include <string.h>

void SNEPPX_vmem_init(SNEPPXVMemAllocator* alloc) {
    if (alloc) memset(alloc, 0, sizeof(*alloc));
}
void SNEPPX_vmem_cleanup(SNEPPXVMemAllocator* alloc) { (void)alloc; }
void* SNEPPX_vmem_reserve(SNEPPXVMemAllocator* alloc, size_t bytes, size_t alignment, int flags) {
    (void)alloc; (void)bytes; (void)alignment; (void)flags; return NULL;
}
int SNEPPX_vmem_commit(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes) {
    (void)alloc; (void)addr; (void)bytes; return 0;
}
int SNEPPX_vmem_decommit(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes) {
    (void)alloc; (void)addr; (void)bytes; return 0;
}
void SNEPPX_vmem_release(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes) {
    (void)alloc; (void)addr; (void)bytes;
}
int SNEPPX_vmem_advise_hugepage(void* addr, size_t bytes) {
    (void)addr; (void)bytes; return 0;
}
int SNEPPX_vmem_advise_nohugepage(void* addr, size_t bytes) {
    (void)addr; (void)bytes; return 0;
}
int SNEPPX_vmem_register_evict_strategy(SNEPPXVMemAllocator* alloc, SNEPPXEvictStrategy* strat) {
    (void)alloc; (void)strat; return 0;
}
int SNEPPX_vmem_evict_page(SNEPPXVMemAllocator* alloc, void* addr, size_t size) {
    (void)alloc; (void)addr; (void)size; return 0;
}
void SNEPPX_vmem_set_oom_handler(SNEPPXVMemAllocator* alloc, int (*handler)(size_t)) {
    (void)alloc; (void)handler;
}
