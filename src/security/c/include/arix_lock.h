#ifndef ARIX_LOCK_H
#define ARIX_LOCK_H

#include <stddef.h>

int arix_mlock(void* ptr, size_t len);
int arix_munlock(void* ptr, size_t len);
int arix_mlockall_possible(void);

#endif
