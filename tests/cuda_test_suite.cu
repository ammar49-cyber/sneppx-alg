// SNEPPX CUDA Test Suite
// Compile: nvcc -o cuda_test_suite cuda_test_suite.cu -lcublas -lcurand -I../kernel/cuda
// Run: ./cuda_test_suite

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "../kernel/cuda/common.cuh"
#include "../kernel/cuda/tensor_cuda.h"
#include "../kernel/cuda/attention_cuda.h"
#include "../kernel/cuda/autodiff_cuda.h"
#include "../kernel/cuda/optim_cuda.h"
#include "../kernel/cuda/memory_cuda.h"
#include "../kernel/cuda/rng_cuda.h"

// ============================================================================
// Test Utilities
// ============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) do { \
    printf("  TEST: %s ... ", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { \
    printf("PASSED\n"); \
    g_tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAILED: %s\n", msg); \
    g_tests_failed++; \
} while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

// ============================================================================
// Device Properties Test
// ============================================================================

void test_device_properties() {
    TEST("device properties query");
    
    SNEPPX_DeviceProps props;
    SNEPPX_CudaError err = sneppx_cuda_get_device_props(0, &props);
    ASSERT(err == SNEPPX_CUDA_SUCCESS, "get_device_props failed");
    ASSERT(props.num_sms > 0, "no SMs detected");
    ASSERT(props.global_mem_bytes > 0, "no global memory");
    ASSERT(props.warp_size == 32, "warp size != 32");
    
    PASS();
}

// ============================================================================
// GEMM Test (vs cuBLAS reference)
// ============================================================================

void test_gemm() {
    TEST("GEMM forward (comparison with cuBLAS)");
    
    const int M = 128, N = 128, K = 128;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half *a, *b, *c_cublas, *c_sneppx;
    cudaMallocAsync(&a, M * K * sizeof(half), stream);
    cudaMallocAsync(&b, K * N * sizeof(half), stream);
    cudaMallocAsync(&c_cublas, M * N * sizeof(half), stream);
    cudaMallocAsync(&c_sneppx, M * N * sizeof(half), stream);
    
    // Fill with random data
    float* h_a = (float*)malloc(M * K * sizeof(float));
    float* h_b = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = (float)rand() / RAND_MAX;
    
    // Convert to half and copy
    half* h_a_half = (half*)malloc(M * K * sizeof(half));
    half* h_b_half = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_a_half[i] = __float2half_rn(h_a[i]);
    for (int i = 0; i < K * N; i++) h_b_half[i] = __float2half_rn(h_b[i]);
    
    cudaMemcpyAsync(a, h_a_half, M * K * sizeof(half), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b, h_b_half, K * N * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    // cuBLAS reference
    cublasHandle_t handle = sneppx_cublas_get_handle();
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K,
                 &beta, c_cublas, CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // SNEPPX GEMM
    SNEPPX_GemmParams gemm_params;
    gemm_params.a = a;
    gemm_params.b = b;
    gemm_params.c = c_sneppx;
    gemm_params.M = M;
    gemm_params.N = N;
    gemm_params.K = K;
    gemm_params.alpha = 1.0f;
    gemm_params.beta = 0.0f;
    gemm_params.act = SNEPPX_ACT_NONE;
    
    sneppx_cuda_gemm_forward(stream, &gemm_params);
    
    cudaStreamSynchronize(stream);
    
    // Compare
    half* h_c_cublas = (half*)malloc(M * N * sizeof(half));
    half* h_c_sneppx = (half*)malloc(M * N * sizeof(half));
    cudaMemcpy(h_c_cublas, c_cublas, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_sneppx, c_sneppx, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(__half2float(h_c_cublas[i]) - __half2float(h_c_sneppx[i]));
        if (d > max_diff) max_diff = d;
    }
    
    ASSERT(max_diff < 0.1f, "GEMM results differ too much from cuBLAS");
    
    free(h_a); free(h_b); free(h_a_half); free(h_b_half);
    free(h_c_cublas); free(h_c_sneppx);
    cudaFreeAsync(a, stream); cudaFreeAsync(b, stream);
    cudaFreeAsync(c_cublas, stream); cudaFreeAsync(c_sneppx, stream);
    cudaStreamDestroy(stream);
    
    PASS();
}

// ============================================================================
// Element-wise Test
// ============================================================================

void test_elementwise() {
    TEST("element-wise add");
    
    const int N = 1024;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half *a, *b, *c;
    cudaMallocAsync(&a, N * sizeof(half), stream);
    cudaMallocAsync(&b, N * sizeof(half), stream);
    cudaMallocAsync(&c, N * sizeof(half), stream);
    
    half h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half_rn((float)i);
        h_b[i] = __float2half_rn((float)(N - i));
    }
    cudaMemcpyAsync(a, h_a, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b, h_b, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    sneppx_cuda_elementwise_add(stream, a, b, c, N);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_c, c, N * sizeof(half), cudaMemcpyDeviceToHost);
    