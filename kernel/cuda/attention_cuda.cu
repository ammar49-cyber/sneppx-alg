#include "attention_cuda.h"
#include "common.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_pipeline_primitives.h>

namespace cg = cooperative_groups;

// ============================================================================
// Flash Attention v2 Kernel (Forward)
// Based on: https://arxiv.org/abs/2307.08691 (Dao, 2023)
//
// Key optimizations:
// 1. Online softmax rescaling (safe softmax)
// 2. Tiled attention with split KV
// 3. Warp-level reduction for max/sum
// 4. Causal masking with diagonal splitting
// 5. HBM-aware block scheduling
// ============================================================================

template<int kBlockM, int kBlockN, int kHeadDim, int kWarpCount>
__global__ void flashattn_v2_fwd_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    float* __restrict__ lse,
    const half* __restrict__ bias,
    const half* __restrict__ mask,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    float scale,
    float dropout_p,
    bool is_causal,
    int window_left,
    int window_right
) {
    // Block indices
    int bid_x = blockIdx.x;  // head * num_blocks_q + block_q
    int bid_y = blockIdx.y;  // batch
    
    int num_blocks_q = (seq_len_q + kBlockM - 1) / kBlockM;
    int head_idx = bid_x / num_blocks_q;
    int block_q_idx = bid_x % num_blocks_q;
    int batch_idx = bid_y;
    
    int q_start = block_q_idx * kBlockM;
    int q_end = min(q_start + kBlockM, seq_len_q);
    int q_blocks = q_end - q_start;
    
    // Shared memory for tiles
    __shared__ __align__(16) half q_shared[kBlockM * kHeadDim];
    __shared__ __align__(16) half k_shared[kBlockN * kHeadDim];
    __shared__ __align__(16) half v_shared[kBlockN * kHeadDim];
    __shared__ __align__(16) half bias_shared[kBlockM * kBlockN];
    __shared__ __align__(16) float row_max_shared[kBlockM];
    __shared__ __align__(16) float row_sum_shared[kBlockM];
    
    // Accumulators in registers
    float acc[kBlockM * (kHeadDim / kWarpCount)];
    
    // Initialize accumulators
    for (int i = 0; i < q_blocks; i++) {
        for (int j = 0; j < kHeadDim / kWarpCount; j++) {
            acc[i * (kHeadDim / kWarpCount) + j] = 0.0f;
        }
    }
    
    // Load Q tile to shared memory (coalesced)
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    
    // Each warp loads a portion of Q
    int num_warps = blockDim.y;
    int q_load_rows_per_warp = (q_blocks + num_warps - 1) / num_warps;
    
    for (int r = warp_id * q_load_rows_per_warp; r < min((warp_id + 1) * q_load_rows_per_warp, q_blocks); r++) {
        int q_row = q_start + r;
        if (q_row < seq_len_q) {
            int q_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + q_row) * kHeadDim;
            for (int c = lane_id; c < kHeadDim; c += SNEPPX_WARP_SIZE) {
                q_shared[r * kHeadDim + c] = q[q_offset + c];
            }
        }
    }
    __syncthreads();
    
    // Initialize online softmax
    float row_max[kBlockM];
    float row_sum[kBlockM];
    
    for (int i = 0; i < q_blocks; i++) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }
    
    // Iterate over KV blocks
    int kv_blocks = (seq_len_kv + kBlockN - 1) / kBlockN;
    
    for (int kv_block = 0; kv_block < kv_blocks; kv_block++) {
        int kv_start = kv_block * kBlockN;
        int kv_end = min(kv_start + kBlockN, seq_len_kv);
        int kv_blocks_actual = kv_end - kv_start;
        
        // Load K tile to shared memory
        for (int r = warp_id * q_load_rows_per_warp; r < min((warp_id + 1) * q_load_rows_per_warp, kv_blocks_actual); r++) {
            int kv_row = kv_start + r;
            if (kv_row < seq_len_kv) {
                int k_offset = ((batch_idx * num_heads + head_idx) * seq_len_kv + kv_row) * kHeadDim;
                for (int c = lane_id; c < kHeadDim; c += SNEPPX_WARP_SIZE) {
                    k_shared[r * kHeadDim + c] = k[k_offset + c];
                }
            }
        }
        
        // Load V tile to shared memory
        for (int r = warp_id * q_load_rows_per_warp; r < min((warp_id + 1) * q_load_rows_per_warp, kv_blocks_actual); r++) {
            int kv_row = kv_start + r;
            if (kv_row < seq_len_kv) {
                int v_offset = ((batch_idx * num_heads + head_idx) * seq_len_kv + kv_row) * kHeadDim;
                for (int c = lane_id; c < kHeadDim; c += SNEPPX_WARP_SIZE) {
                    v_shared[r * kHeadDim + c] = v[v_offset + c];
                }
            }
        }
        
        // Load bias tile if present
        if (bias != nullptr) {
            for (int r = warp_id; r < q_blocks; r += num_warps) {
                for (int c = lane_id; c < kv_blocks_actual; c += SNEPPX_WARP_SIZE) {
                    bias_shared[r * kBlockN + c] = bias[
                        ((batch_idx * num_heads + head_idx) * seq_len_q + (q_start + r)) * seq_len_kv + (kv_start + c)
                    ];
                }
            }
        }
        __syncthreads();
        
        // For each Q row in this tile
        for (int q_off = 0; q_off < q_blocks; q_off++) {
            int actual_q_row = q_start + q_off;
            float m_prev = row_max[q_off];
            
            // Check causal masking
            bool skip_block = false;
            if (is_causal) {
                int causal_kv_end = actual_q_row;
                if (kv_start > causal_kv_end) {
                    skip_block = true;
                }
            }
            
            if (skip_block) {
                continue;
            }
            
            // Compute S = Q * K^T for this row, using warp-level GEMM
            float s_local = 0.0f;
            
            // Each lane of the warp computes partial dot product
            for (int k_off = lane_id; k_off < kHeadDim; k_off += SNEPPX_WARP_SIZE) {
                float q_val = __half2float(q_shared[q_off * kHeadDim + k_off]);
                for (int kv_off = 0; kv_off < kv_blocks_actual; kv_off++) {
                    float k_val = __half2float(k_shared[kv_off * kHeadDim + k_off]);
                    
                    if (q_off < q_blocks && kv_off < kv_blocks_actual) {
                        // causal mask
                        bool valid = true;
                        if (is_causal && (kv_start + kv_off) > actual_q_row) {
                            valid = false;
                        }
                        // window mask
                        if (valid && (window_left >= 0 || window_right >= 0)) {
                            int pos_diff = (kv_start + kv_off) - actual_q_row;
                            if (pos_diff > window_right || -pos_diff > window_left) {
                                valid = false;
                            }
                        }
                        
                        if (valid) {
                            s_local += q_val * k_val;
                        }
                    }
                }
            }
            
            // Warp reduction
            s_local = sneppx_warp_reduce_sum(s_local);
            
            // Apply scale and bias
            if (lane_id == 0) {
                s_local *= scale;
                if (bias != nullptr) {
                    s_local += __half2float(bias_shared[q_off * kBlockN + kv_blocks_actual - 1]);
                }
                if (mask != nullptr) {
                    // Apply mask value
                }
                
                // Online safe softmax update
                float m_new = fmaxf(m_prev, s_local);
                float exp_s = expf(s_local - m_new);
                float exp_old = expf(m_prev - m_new);
                
                row_max[q_off] = m_new;
                row_sum[q_off] = row_sum[q_off] * exp_old + exp_s;
                row_max_shared[q_off] = m_new;
                row_sum_shared[q_off] = row_sum[q_off];
            }
        }
        
        __syncthreads();
        
        // Broadcast softmax stats to all lanes
        float m_bcast[kBlockM];
        float s_bcast[kBlockM];
        
        if (warp_id == 0) {
            for (int i = lane_id; i < q_blocks; i += SNEPPX_WARP_SIZE) {
                m_bcast[i] = row_max_shared[i];
                s_bcast[i] = row_sum_shared[i];
            }
        }
        
        // Compute O = softmax(S) * V
        for (int q_off = 0; q_off < q_blocks; q_off++) {
            float m_new = m_bcast[q_off];
            float s_new = s_bcast[q_off];
            
            // Rescale previous output
            float rescale = 1.0f;
            if (kv_block > 0) {
                rescale = expf(row_max[q_off] - m_new);
            }
            
            for (int d_off = lane_id; d_off < kHeadDim; d_off += SNEPPX_WARP_SIZE) {
                acc[q_off * (kHeadDim / kWarpCount) + (d_off / SNEPPX_WARP_SIZE)] *= rescale;
            }
            
            // Add contribution from current KV block
            for (int kv_off = 0; kv_off < kv_blocks_actual; kv_off++) {
                bool valid = true;
                if (is_causal && (kv_start + kv_off) > (q_start + q_off)) {
                    valid = false;
                }
                if (valid && (window_left >= 0 || window_right >= 0)) {
                    int pos_diff = (kv_start + kv_off) - (q_start + q_off);
                    if (pos_diff > window_right || -pos_diff > window_left) {
                        valid = false;
                    }
                }
                
                if (valid) {
                    // Compute S value
                    float s_val = 0.0f;
                    for (int kk = lane_id; kk < kHeadDim; kk += SNEPPX_WARP_SIZE) {
                        s_val += __half2float(q_shared[q_off * kHeadDim + kk]) *
                                  __half2float(k_shared[kv_off * kHeadDim + kk]);
                    }
                    s_val = sneppx_warp_reduce_sum(s_val);