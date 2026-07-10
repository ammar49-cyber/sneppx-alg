#include "memory_cuda.h"
#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// CUBLAS Handle (TLS singleton)
// ============================================================================

#include <cublas_v2.h>

// Thread-local CUBLAS handle
__thread cublasHandle_t tls_cublas_handle = nullptr;

cublasHandle_t sneppx_cublas_get_handle(void) {
    if (!tls_cublas_handle) {
        cublasCreate(&tls_cublas_handle);
        cublasSetMathMode(tls_cublas_handle, CUBLAS_TENSOR_OP_MATH);
        
        // Enable TF32 for tensor cores (Ampere+)
        #if CUBLAS_VERSION >= 11000
        cublasSetMathMode(tls_cublas_handle, CUBLAS_DEFAULT_MATH);
        #endif
    }
    return tls_cublas_handle;
}

void sneppx_cublas_destroy_handle(void) {
    if (tls_cublas_handle) {
        cublasDestroy(tls_cublas_handle);
        tls_cublas_handle = nullptr;
    }
}

// ============================================================================
// Device Properties
// ============================================================================

SNEPPX_CudaError sneppx_cuda_get_device_props(int device_id, SNEPPX_DeviceProps* props) {
    if (!props) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return SNEPPX_CUDA_ERROR_INVALID_DEVICE;
    
    props->device_id = device_id;
    snprintf(props->name, sizeof(props->name), "%s", prop.name);
    props->compute_capability_major = prop.major;
    props->compute_capability_minor = prop.minor;
    props->global_mem_bytes = prop.totalGlobalMem;
    props->shared_mem_per_block = prop.sharedMemPerBlock;
    props->shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
    props->max_threads_per_block = prop.maxThreadsPerBlock;
    props->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    props->max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    props->warp_size = prop.warpSize;
    props->num_sms = prop.multiProcessorCount;
    props->clock_rate_khz = prop.clockRate;
    props->memory_clock_rate_khz = prop.memoryClockRate;
    props->memory_bus_width = prop.memoryBusWidth;
    props->l2_cache_size = prop.l2CacheSize;
    props->max_shared_mem_per_block_optin = prop.sharedMemPerBlockOptin;
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_INVALID_DEVICE;
}

SNEPPX_CudaError sneppx_cuda_get_device(int* device_id) {
    cudaError_t err = cudaGetDevice(device_id);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_INVALID_DEVICE;
}

SNEPPX_CudaError sneppx_cuda_device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Memory Pool Implementation
// ============================================================================

SNEPPX_CudaError sneppx_mempool_create(
    SNEPPX_MemoryPool** pool,
    int device_id,
    size_t block_size,
    int initial_blocks,
    int max_blocks
) {
    if (!pool || initial_blocks <= 0 || max_blocks <= 0) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    SNEPPX_MemoryPool* p = (SNEPPX_MemoryPool*)malloc(sizeof(SNEPPX_MemoryPool));
    if (!p) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    
    p->device_id = device_id;
    p->block_size = block_size;
    p->num_blocks = 0;
    p->max_blocks = max_blocks;
    p->total_capacity = (size_t)max_blocks * block_size;
    p->total_used = 0;
    
    p->blocks = (void**)calloc(max_blocks, sizeof(void*));
    p->block_sizes = (size_t*)calloc(max_blocks, sizeof(size_t));
    p->block_used = (int*)calloc(max_blocks, sizeof(int));
    
    if (!p->blocks || !p->block_sizes || !p->block_used) {
        free(p->blocks);
        free(p->block_sizes);
        free(p->block_used);
        free(p);
        return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    cudaSetDevice(device_id);
    
    // Pre-allocate initial blocks
    for (int i = 0; i < initial_blocks; i++) {
        cudaError_t err = cudaMalloc(&p->blocks[i], block_size);
        if (err != cudaSuccess) break;
        p->block_sizes[i] = block_size;
        p->block_used[i] = 0;
        p->num_blocks++;
    }
    
    *pool = p;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_mempool_destroy(SNEPPX_MemoryPool* pool) {
    if (!pool) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaSetDevice(pool->device_id);
    
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i]) {
            cudaFree(pool->blocks[i]);
        }
    }
    
    free(pool->blocks);
    free(pool->block_sizes);
    free(pool->block_used);
    free(pool);
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_mempool_alloc(
    SNEPPX_MemoryPool* pool,
    void** ptr,
    size_t size
) {
    if (!pool || !ptr || size == 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    // Find a free block large enough
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->block_used[i] && pool->block_sizes[i] >= size) {
            pool->block_used[i] = 1;
            pool->total_used += pool->block_sizes[i];
            *ptr = pool->blocks[i];
            return SNEPPX_CUDA_SUCCESS;
        }
    }
    
    // Allocate new block if possible
    if (pool->num_blocks < pool->max_blocks) {
        int i = pool->num_blocks;
        cudaError_t err = cudaMalloc(&pool->blocks[i], size);
        if (err != cudaSuccess) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
        
        pool->block_sizes[i] = size;
        pool->block_used[i] = 1;
        pool->total_used += size;
        pool->num_blocks++;
        *ptr = pool->blocks[i];
        return SNEPPX_CUDA_SUCCESS;
    }
    