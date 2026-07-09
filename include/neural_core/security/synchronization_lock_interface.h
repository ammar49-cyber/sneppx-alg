#ifndef SNEPPX_LOCK_H
#define SNEPPX_LOCK_H

#include <stddef.h>

int SNEPPX_mlock(void* ptr, size_t len);
int SNEPPX_munlock(void* ptr, size_t len);
int SNEPPX_mlockall_possible(void);

#endif
