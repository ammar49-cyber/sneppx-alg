#include "bench_common.h"
#include "arix_npe.h"

static void bench_npe_create(void) {
    BENCH_INIT(bs);
    printf("  NPE create:\n");
    ArixNPEConfig cfg = arix_npe_config_default();
    BENCH_START(bs, 10000, 100, {
        ArixNPEVM* vm = arix_npe_vm_create(&cfg);
        arix_npe_vm_destroy(vm);
    });
    bench_print("vm_create", &bs);

    BENCH_START(bs, 10000, 100, {
        ArixNPEProgram* p = arix_npe_program_create(32);
        arix_npe_program_destroy(p);
    });
    bench_print("program_create", &bs);
}

static void bench_npe_basic_ops(void) {
    BENCH_INIT(bs);
    ArixNPEProgram* prog = arix_npe_program_create(16);
    ArixNPEInstruction i1 = {ARIX_LOAD, 1, 0, 0, 0, {64,64}, {0,0}};
    ArixNPEInstruction i2 = {ARIX_RELU, 2, 1, 0, 0, {0,0}, {0,0}};
    ArixNPEInstruction i3 = {ARIX_SOFTMAX, 3, 2, 0, 0, {0,0}, {0,0}};
    ArixNPEInstruction i4 = {ARIX_HALT, 0, 0, 0, 0, {0,0}, {0,0}};
    arix_npe_program_append(prog, i1);
    arix_npe_program_append(prog, i2);
    arix_npe_program_append(prog, i3);
    arix_npe_program_append(prog, i4);

    ArixNPEConfig cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&cfg);
    arix_npe_vm_load(vm, prog);
    size_t sh_in[] = {64, 64};
    ArixTensor* input = arix_tensor_ones(sh_in, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;

    printf("  NPE forward [64,64]:\n");
    BENCH_START(bs, 1000, 50, {
        arix_npe_vm_run(vm, input, &output);
        if (output) arix_tensor_destroy(output);
    });
    bench_print("relu+softmax", &bs);

    arix_tensor_destroy(input);
    arix_npe_vm_destroy(vm);
    arix_npe_program_destroy(prog);
}

static void bench_npe_matmul(void) {
    BENCH_INIT(bs);
    ArixNPEProgram* prog = arix_npe_program_create(16);
    ArixNPEInstruction i1 = {ARIX_LOAD, 1, 0, 0, 0, {128,64}, {0,0}};
    ArixNPEInstruction i2 = {ARIX_LOAD, 2, 0, 0, (int)(64*64), {64,128}, {0,0}};
    ArixNPEInstruction i3 = {ARIX_MATMUL, 3, 1, 2, 0, {0,0}, {0,0}};
    ArixNPEInstruction i4 = {ARIX_HALT, 0, 0, 0, 0, {0,0}, {0,0}};
    arix_npe_program_append(prog, i1);
    arix_npe_program_append(prog, i2);
    arix_npe_program_append(prog, i3);
    arix_npe_program_append(prog, i4);

    ArixNPEConfig cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&cfg);
    arix_npe_vm_load(vm, prog);
    size_t sh_in[] = {128, 64};
    ArixTensor* input = arix_tensor_ones(sh_in, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;

    printf("  NPE matmul [128,64]x[64,128]:\n");
    BENCH_START(bs, 500, 50, {
        arix_npe_vm_run(vm, input, &output);
        if (output) arix_tensor_destroy(output);
    });
    bench_print("matmul", &bs);

    arix_tensor_destroy(input);
    arix_npe_vm_destroy(vm);
    arix_npe_program_destroy(prog);
}

int main(void) {
    printf("=== NPE Benchmarks ===\n");
    BENCH_RUN("Create", bench_npe_create);
    BENCH_RUN("Basic ops", bench_npe_basic_ops);
    BENCH_RUN("Matmul", bench_npe_matmul);
    BENCH_MAIN();
}
