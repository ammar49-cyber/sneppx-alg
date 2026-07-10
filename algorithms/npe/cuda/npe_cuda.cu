#include "npe_cuda.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// NPE instruction dispatch kernel
__global__ void npe_execute_kernel(
    const int* opcodes,
    const int* operands,
    half* registers,
    half* memory,
    const half* input,
    half* output,
    int num_instructions,
    int max_operands,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    // PC initialized per batch
    int pc = 0;
    
    while (pc < num_instructions) {
        int op = opcodes[pc];
        int* ops = const_cast<int*>(&operands[pc * max_operands]);
        
        // Decode operands
        int dst = ops[0];
        int src1 = ops[1];
        int src2 = ops[2];
        int imm = ops[3];
        
        // Execute instruction (tid processes one element)
        switch (op) {
            case 0: { // ADD
                float a = __half2float(registers[src1 * blockDim.x + tid]);
                float b = __half2float(registers[src2 * blockDim.x + tid]);
                registers[dst * blockDim.x + tid] = __float2half_rn(a + b);
                break;
            }
            case 1: { // MUL
                float a = __half2float(registers[src1 * blockDim.x + tid]);
                float b = __half2float(registers[src2 * blockDim.x + tid]);
                registers[dst * blockDim.x + tid] = __float2half_rn(a * b);
                break;
            }
            case 2: { // MATMUL (simplified dot product)
                float sum = 0.0f;
                // Thread 0 handles the dot product for this batch
                if (tid == 0) {
                    for (int k = 0; k < imm; k++) {
                        sum += __half2float(registers[src1 * blockDim.x + k]) *
                               __half2float(registers[src2 * blockDim.x + k]);
                    }
                    registers[dst * blockDim.x] = __float2half_rn(sum);
                }
                break;
            }
            case 3: { // RELU
                float x = __half2float(registers[src1 * blockDim.x + tid]);
                registers[dst * blockDim.x + tid] = __float2half_rn(fmaxf(0.0f, x));
                break;
            }
            case 4: { // CONV2D (placeholder)
                registers[dst * blockDim.x + tid] = registers[src1 * blockDim.x + tid];
                break;
            }
            case 5: { // REDUCE_SUM
                float val = __half2float(registers[src1 * blockDim.x + tid]);
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
                }
                if (tid == 0) registers[dst * blockDim.x] = __float2half_rn(val);
                break;
            }
            case 6: { // RESHAPE (no-op, just copy)
                registers[dst * blockDim.x + tid] = registers[src1 * blockDim.x + tid];
                break;
            }
            case 7: { // BRANCH (conditional jump)
                float cond = __half2float(registers[src1 * blockDim.x]);
                if (cond > 0.0f) {