#ifndef SNEPPX_DRIVER_REGISTRY_H
#define SNEPPX_DRIVER_REGISTRY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_DRIVER_MAX_NAME 64
#define SNEPPX_DRIVER_MAX_REGISTERED 32

typedef struct {
    void* lib_handle;
    struct {
        int (*init)(void);
        int (*get_device_count)(int*);
        int (*get_device_props)(int, void*, size_t);
        void* (*create_context)(int);
        void (*destroy_context)(void*);
        int (*alloc)(void**, size_t);
        int (*free)(void*);
        int (*htod)(void*, const void*, size_t);
        int (*dtoh)(void*, const void*, size_t);
        int (*launch)(const char*, void*, void**, size_t, size_t, size_t);
        int (*synchronize)(void*);
    } api;
    int loaded;
} SNEPPXDriverEntry;

typedef struct {
    char name[SNEPPX_DRIVER_MAX_NAME];
    int count;
} SNEPPXDriverInfo;

int SNEPPX_driver_register(const char* name, SNEPPXDriverEntry* entry);
int SNEPPX_driver_unregister(const char* name);
SNEPPXDriverEntry* SNEPPX_driver_get(const char* name);
int SNEPPX_driver_get_info(SNEPPXDriverInfo* info, size_t max_info, size_t* count);
int SNEPPX_driver_load_library(const char* name, const char* lib_path);
int SNEPPX_driver_unload_all(void);

#ifdef __cplusplus
}
#endif

#endif
