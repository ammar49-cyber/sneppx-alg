#include "gradient_optimization_suite.h"
#include "multidimensional_tensor_engine.h"
#include <stdio.h>
#include <math.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_optimizer_config_default(void) {
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    ASSERT(cfg.learning_rate == 0.01f, "default lr");
    ASSERT(cfg.type == ARIX_OPTIMIZER_SGD, "default type");
}

static void test_optimizer_sgd(void) {
    size_t shape[] = {4};
    ArixTensor* p = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* g = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ((float*)p->data)[0] = 1.0f; ((float*)p->data)[1] = 2.0f;
    ((float*)g->data)[0] = 0.1f; ((float*)g->data)[1] = -0.2f;
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    cfg.learning_rate = 0.5f; cfg.weight_decay = 0.0f; cfg.momentum = 0.0f;
    ArixOptimizer* opt = arix_optimizer_create(&cfg);
    ArixTensor* params[] = {p};
    ArixTensor* grads[] = {g};
    arix_optimizer_step(opt, params, grads, 1);
    float* pd = (float*)p->data;
    ASSERT(fabsf(pd[0] - (1.0f - 0.5f * 0.1f)) < 1e-6f, "sgd p0");
    ASSERT(fabsf(pd[1] - (2.0f - 0.5f * (-0.2f))) < 1e-6f, "sgd p1");
    arix_optimizer_destroy(opt);
    arix_tensor_destroy(p);
    arix_tensor_destroy(g);
}

static void test_optimizer_sgd_momentum(void) {
    size_t shape[] = {2};
    ArixTensor* p = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* g = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ((float*)p->data)[0] = 1.0f; ((float*)p->data)[1] = 2.0f;
    ((float*)g->data)[0] = 0.1f; ((float*)g->data)[1] = 0.1f;
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    cfg.learning_rate = 0.1f; cfg.momentum = 0.9f;
    cfg.type = ARIX_OPTIMIZER_SGD;
    ArixOptimizer* opt = arix_optimizer_create(&cfg);
    ArixTensor* params[] = {p};
    ArixTensor* grads[] = {g};
    arix_optimizer_step(opt, params, grads, 1);
    float* pd = (float*)p->data;
    ASSERT(pd[0] < 1.0f, "momentum p0 < 1.0");
    ASSERT(pd[1] < 2.0f, "momentum p1 < 2.0");
    arix_optimizer_destroy(opt);
    arix_tensor_destroy(p);
    arix_tensor_destroy(g);
}

static void test_optimizer_adam(void) {
    size_t shape[] = {3};
    ArixTensor* p = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* g = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    for (size_t i = 0; i < 3; i++) ((float*)g->data)[i] = 0.5f;
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    cfg.learning_rate = 0.01f; cfg.type = ARIX_OPTIMIZER_ADAM; cfg.weight_decay = 0.0f;
    ArixOptimizer* opt = arix_optimizer_create(&cfg);
    ArixTensor* params[] = {p};
    ArixTensor* grads[] = {g};
    for (int s = 0; s < 20; s++) arix_optimizer_step(opt, params, grads, 1);
    float* pd = (float*)p->data;
    for (size_t i = 0; i < 3; i++) ASSERT(pd[i] < 0.99f, "adam decreased");
    arix_optimizer_destroy(opt);
    arix_tensor_destroy(p);
    arix_tensor_destroy(g);
}

static void test_optimizer_adamw(void) {
    size_t shape[] = {2};
    ArixTensor* p = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* g = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ((float*)g->data)[0] = 0.1f; ((float*)g->data)[1] = 0.1f;
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    cfg.learning_rate = 0.01f; cfg.type = ARIX_OPTIMIZER_ADAMW;
    cfg.weight_decay = 0.01f;
    ArixOptimizer* opt = arix_optimizer_create(&cfg);
    ArixTensor* params[] = {p};
    ArixTensor* grads[] = {g};
    for (int s = 0; s < 20; s++) arix_optimizer_step(opt, params, grads, 1);
    float* pd = (float*)p->data;
    for (size_t i = 0; i < 2; i++) ASSERT(pd[i] < 0.99f, "adamw decreased");
    arix_optimizer_destroy(opt);
    arix_tensor_destroy(p);
    arix_tensor_destroy(g);
}

static void test_optimizer_rmsprop(void) {
    size_t shape[] = {2};
    ArixTensor* p = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* g = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ((float*)g->data)[0] = 0.2f; ((float*)g->data)[1] = 0.2f;
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    cfg.learning_rate = 0.01f; cfg.type = ARIX_OPTIMIZER_RMSPROP;
    ArixOptimizer* opt = arix_optimizer_create(&cfg);
    ArixTensor* params[] = {p};
    ArixTensor* grads[] = {g};
    for (int s = 0; s < 4; s++) arix_optimizer_step(opt, params, grads, 1);
    float* pd = (float*)p->data;
    for (size_t i = 0; i < 2; i++) ASSERT(pd[i] < 1.0f, "rmsprop decreased");
    arix_optimizer_destroy(opt);
    arix_tensor_destroy(p);
    arix_tensor_destroy(g);
}

static void test_optimizer_adagrad(void) {
    size_t shape[] = {2};
    ArixTensor* p = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* g = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ((float*)g->data)[0] = 0.3f; ((float*)g->data)[1] = 0.3f;
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    cfg.learning_rate = 0.1f; cfg.type = ARIX_OPTIMIZER_ADAGRAD;
    ArixOptimizer* opt = arix_optimizer_create(&cfg);
    ArixTensor* params[] = {p};
    ArixTensor* grads[] = {g};
    for (int s = 0; s < 3; s++) arix_optimizer_step(opt, params, grads, 1);
    float* pd = (float*)p->data;
    for (size_t i = 0; i < 2; i++) ASSERT(pd[i] < 1.0f, "adagrad decreased");
    arix_optimizer_destroy(opt);
    arix_tensor_destroy(p);
    arix_tensor_destroy(g);
}

static void test_optimizer_adadelta(void) {
    size_t shape[] = {2};
    ArixTensor* p = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* g = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ((float*)g->data)[0] = 0.1f; ((float*)g->data)[1] = 0.1f;
    ArixOptimizerConfig cfg = arix_optimizer_config_default();
    cfg.type = ARIX_OPTIMIZER_ADADELTA; cfg.learning_rate = 1.0f;
    ArixOptimizer* opt = arix_optimizer_create(&cfg);
    ArixTensor* params[] = {p};
    ArixTensor* grads[] = {g};
    for (int s = 0; s < 3; s++) arix_optimizer_step(opt, params, grads, 1);
    float* pd = (float*)p->data;
    for (size_t i = 0; i < 2; i++) ASSERT(pd[i] < 1.0f, "adadelta decreased");
    arix_optimizer_destroy(opt);
    arix_tensor_destroy(p);
    arix_tensor_destroy(g);
}

static void test_lr_scheduler_step(void) {
    float lr = 0.1f;
    ArixLRScheduler* s = arix_lr_scheduler_step_lr(&lr, 0.1f, 10);
    ASSERT(s != NULL, "sched created");
    for (int i = 0; i < 10; i++) arix_lr_scheduler_step(s, 0.0f);
    ASSERT(fabsf(lr - 0.01f) < 1e-6f, "step lr decayed 10x");
    arix_lr_scheduler_destroy(s);
}

static void test_lr_scheduler_exponential(void) {
    float lr = 0.1f;
    ArixLRScheduler* s = arix_lr_scheduler_exponential(&lr, 0.5f);
    ASSERT(s != NULL, "sched created");
    arix_lr_scheduler_step(s, 0.0f);
    ASSERT(fabsf(lr - 0.05f) < 1e-6f, "exp step 1");
    arix_lr_scheduler_step(s, 0.0f);
    ASSERT(fabsf(lr - 0.025f) < 1e-6f, "exp step 2");
    arix_lr_scheduler_destroy(s);
}

static void test_lr_scheduler_cosine(void) {
    float lr = 0.0f;
    ArixLRScheduler* s = arix_lr_scheduler_cosine(&lr, 0.0f, 1.0f, 100);
    ASSERT(s != NULL, "sched created");
    arix_lr_scheduler_step(s, 0.0f);
    float after = lr;
    ASSERT(after > 0.0f && after < 1.0f, "cosine range");
    ASSERT(fabsf(after - (0.0f + 0.5f * 1.0f * (1.0f + cosf(3.14159265f / 100.0f)))) < 1e-4f, "cosine value");
    arix_lr_scheduler_destroy(s);
}

static void test_lr_scheduler_reduce_on_plateau(void) {
    float lr = 0.1f;
    ArixLRScheduler* s = arix_lr_scheduler_reduce_on_plateau(&lr, 0.5f, 2, 1);
    ASSERT(s != NULL, "sched created");
    for (int i = 0; i < 5; i++) arix_lr_scheduler_step(s, 1.0f);
    ASSERT(fabsf(lr - 0.025f) < 1e-6f, "plateau reduced 2x");
    arix_lr_scheduler_destroy(s);
}

int main(void) {
    run_test("test_optimizer_config_default", test_optimizer_config_default);
    run_test("test_optimizer_sgd", test_optimizer_sgd);
    run_test("test_optimizer_sgd_momentum", test_optimizer_sgd_momentum);
    run_test("test_optimizer_adam", test_optimizer_adam);
    run_test("test_optimizer_adamw", test_optimizer_adamw);
    run_test("test_optimizer_rmsprop", test_optimizer_rmsprop);
    run_test("test_optimizer_adagrad", test_optimizer_adagrad);
    run_test("test_optimizer_adadelta", test_optimizer_adadelta);
    run_test("test_lr_scheduler_step", test_lr_scheduler_step);
    run_test("test_lr_scheduler_exponential", test_lr_scheduler_exponential);
    run_test("test_lr_scheduler_cosine", test_lr_scheduler_cosine);
    run_test("test_lr_scheduler_reduce_on_plateau", test_lr_scheduler_reduce_on_plateau);
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
