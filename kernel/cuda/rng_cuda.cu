#include "rng_cuda.h"
#include "common.cuh"
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void init_rng_states_kernel(
    curandStatePhilox4_32_10_t* states,
    unsigned long long seed,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    curand_init(seed, idx, 0, &states[idx]);
}

SNEPPX_CudaError sneppx_cuda_rng_create(
    SNEPPX_CudaRNG** rng,
    int num_states,
    unsigned long long seed,
    SNEPPX_CudaStream_t stream
) {
    if (!rng || num_states <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    SNEPPX_CudaRNG* r = (SNEPPX_CudaRNG*)malloc(sizeof(SNEPPX_CudaRNG));
    if (!r) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    r->num_states = num_states;
    r->seed = seed;
    r->offset = 0;
    cudaMallocAsync(&r->states, num_states * sizeof(curandStatePhilox4_32_10_t), stream);
    int block = 256;
    int grid = (num_states + block - 1) / block;
    init_rng_states_kernel<<<grid, block, 0, stream>>>(r->states, seed, num_states);
    *rng = r;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_rng_destroy(SNEPPX_CudaRNG* rng) {
    if (!rng) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    cudaFree(rng->states);
    free(rng);
    return SNEPPX_CUDA_SUCCESS;
}

__global__ void reseed_rng_kernel(
    curandStatePhilox4_32_10_t* states,
    unsigned long long seed,
    unsigned long long offset,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    curand_init(seed, idx, offset, &states[idx]);
}

SNEPPX_CudaError sneppx_cuda_rng_seed(
    SNEPPX_CudaRNG* rng,
    unsigned long long seed,
    unsigned long long offset,
    SNEPPX_CudaStream_t stream
) {
    if (!rng) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    rng->seed = seed;
    rng->offset = offset;
    int block = 256;
    int grid = (rng->num_states + block - 1) / block;
    reseed_rng_kernel<<<grid, block, 0, stream>>>(rng->states, seed, offset, rng->num_states);
    return SNEPPX_CUDA_SUCCESS;
}

__device__ __forceinline__ curandStatePhilox4_32_10_t sneppx_get_rng_state(
    curandStatePhilox4_32_10_t* states, int num_states
) {
    return states[(blockIdx.x * blockDim.x + threadIdx.x) % num_states];
}

__device__ __forceinline__ void sneppx_write_rng_state(
    curandStatePhilox4_32_10_t* states, int num_states, curandStatePhilox4_32_10_t state
) {
    states[(blockIdx.x * blockDim.x + threadIdx.x) % num_states] = state;
}

__global__ void rand_uniform_f32_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float low, float high
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_uniform(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = low + val * (high - low);
}

SNEPPX_CudaError sneppx_cuda_rand_uniform_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int numel, float low, float high
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_uniform_f32_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, low, high);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_uniform_f16_kernel(
    curandStatePhilox4_32_10_t* states, half* output,
    int num_states, int numel, float low, float high
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_uniform(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = __float2half_rn(low + val * (high - low));
}

SNEPPX_CudaError sneppx_cuda_rand_uniform_f16(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    half* output, int numel, float low, float high
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_uniform_f16_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, low, high);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_normal_f32_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float mean, float std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_normal(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = mean + val * std;
}

SNEPPX_CudaError sneppx_cuda_rand_normal_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int numel, float mean, float std
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_normal_f32_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, mean, std);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_normal_f16_kernel(
    curandStatePhilox4_32_10_t* states, half* output,
    int num_states, int numel, float mean, float std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_normal(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = __float2half_rn(mean + val * std);
}

SNEPPX_CudaError sneppx_cuda_rand_normal_f16(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,