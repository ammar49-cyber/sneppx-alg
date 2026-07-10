#include "fm_cuda.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Ring all-reduce step
__global__ void ring_reduce_scatter_kernel(
    half* data, int chunk_size, int rank, int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int send_to = (rank + 1) % world_size;
    int recv_from = (rank - 1 + world_size) % world_size;
    
    // In a real implementation, this would use peer-to-peer or NCCL
    // Here we simulate with a simple accumulation
    if (idx < chunk_size) {
        // Each rank's chunk accumulates data from the previous rank
        data[rank * chunk_size + idx] += data[recv_from * chunk_size + idx];
    }
}

SNEPPX_CudaError sneppx_cuda_all_reduce_ring(
    SNEPPX_CudaStream_t stream,
    half* data, size_t numel,
    int rank, int world_size
) {
    if (!data || world_size <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int chunk_size = numel / world_size;
    int block = 256;
    int grid = (chunk_size + block - 1) / block;
    
    for (int step = 0; step < world_size - 1; step++) {
        ring_reduce_scatter_kernel<<<grid, block, 0, stream>>>(
            data, chunk_size, rank, world_size
        );
    }
    
    // All-gather phase (broadcast reduced chunks)
    auto all_gather = [] __global__ (half* d, int cs, int ws) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < cs * ws) {
            int chunk = idx / cs;
            d[idx] = d[chunk * cs + (idx % cs)];
        }
    };
    all_gather<<<grid * world_size, block, 0, stream>>>(data, chunk_size, world_size);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Butterfly all-reduce
SNEPPX_CudaError sneppx_cuda_all_reduce_butterfly(
    SNEPPX_CudaStream_t stream,
    half* data, size_t numel,
    int rank, int world_size
) {
    if (!data || world_size <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    if (!(world_size && !(world_size & (world_size - 1)))) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;  // Must be power of 2
    }
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    for (int stride = 1; stride < world_size; stride <<= 1) {
        int partner = rank ^ stride;
        
        auto butterfly_step = [] __global__ (half* d, int n, int r, int p) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n && r < p) {
                d[idx] += d[idx];
            }
        };
        butterfly_step<<<grid, block, 0, stream>>>(data, numel, rank, partner);
    }
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Gradient quantization (FP16 -> INT8)
__global__ void quantize_gradients_kernel(
    const half* gradients, int8_t* quantized, float* scale,
    int numel, int bits
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int start = bid * block_size;
    int end = min(start + block_size, numel);
    
    float local_max = 0.0f;
    for (int i = start + tid; i < end; i += block_size) {
        float g = fabsf(__half2float(gradients[i]));
        local_max = fmaxf(local_max, g);
    }
    local_max = sneppx_warp_reduce_max(local_max);
    
    if (tid == 0) {
        smem[bid] = local_max;
        scale[bid] = local_max / (float)((1 << (bits - 1)) - 1);
    }
    __syncthreads();
    
    float s = tid == 0 ? scale[bid] : 0.0f;
    if (tid == 0) s = scale[bid];
    __syncthreads();
    s = scale[bid];
    
    float inv_scale = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int i = start + tid; i < end; i += block_size) {
        float g = __half2float(gradients[i]);
        int8_t q = (int8_t)fmaxf(-127.0f, fminf(127.0f, g * inv_scale));
        quantized[i] = q;
    }
}

SNEPPX_CudaError sneppx_cuda_quantize_gradients(
    SNEPPX_CudaStream_t stream,
    const half* gradients, int8_t* quantized,
    float* scale, int numel, int bits
) {
    if (!gradients || !quantized || !scale) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    quantize_gradients_kernel<<<grid, block, grid * sizeof(float), stream>>>(
        gradients, quantized, scale, numel, bits
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}
