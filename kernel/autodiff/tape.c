#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>


ArixTape* arix_tape_create(void) {
    ArixTape* tape = (ArixTape*)arix_malloc(sizeof(ArixTape), 64);
    if (!tape) return NULL;
    memset(tape, 0, sizeof(ArixTape));
    tape->capacity = 64;
    tape->num_vars = 0;
    tape->vars = (ArixVariable**)arix_malloc(tape->capacity * sizeof(ArixVariable*), 64);
    if (!tape->vars) { arix_free(tape, sizeof(ArixTape)); return NULL; }
    memset(tape->vars, 0, tape->capacity * sizeof(ArixVariable*));
    return tape;
}

void arix_tape_destroy(ArixTape* tape) {
    if (!tape) return;
    for (size_t i = 0; i < tape->num_vars; i++) {
        if (tape->vars[i]) arix_variable_destroy(tape->vars[i]);
    }
    if (tape->vars) arix_free(tape->vars, tape->capacity * sizeof(ArixVariable*));
    arix_free(tape, sizeof(ArixTape));
}

void arix_tape_record(ArixTape* tape, ArixVariable* var) {
    if (!tape || !var) return;
    if (tape->num_vars >= tape->capacity) {
        tape->capacity *= 2;
        ArixVariable** new_vars = (ArixVariable**)arix_malloc(tape->capacity * sizeof(ArixVariable*), 64);
        if (!new_vars) return;
        memcpy(new_vars, tape->vars, tape->num_vars * sizeof(ArixVariable*));
        arix_free(tape->vars, tape->capacity / 2 * sizeof(ArixVariable*));
        tape->vars = new_vars;
    }
    tape->vars[tape->num_vars++] = var;
    if (tape->checkpointing && var->backward_ctx && var->recompute_ctx) {
        var->checkpointed = 1;
        var->free_ctx(var->backward_ctx);
        var->backward_ctx = NULL;
    }
}

void arix_tape_checkpoint_begin(ArixTape* tape) {
    if (tape) tape->checkpointing = 1;
}

void arix_tape_checkpoint_end(ArixTape* tape) {
    if (tape) tape->checkpointing = 0;
}

static void tape_topological_sort(ArixTape* tape, ArixVariable** sorted, size_t* num_sorted) {
    if (!tape || !sorted) return;
    size_t n = tape->num_vars;
    int* visited = (int*)arix_malloc(n * sizeof(int), 64);
    if (!visited) return;
    memset(visited, 0, n * sizeof(int));
    *num_sorted = 0;

    /* DFS-based topological sort: for each node, visit all parents first */
    for (size_t i = 0; i < n; i++) {
        if (visited[i]) continue;
        /* Simple stack-based DFS post-order */
        size_t stack_cap = 64;
        size_t* stack = (size_t*)arix_malloc(stack_cap * sizeof(size_t), 64);
        int* on_stack = (int*)arix_malloc(n * sizeof(int), 64);
        if (!stack || !on_stack) { arix_free(stack, stack_cap*sizeof(size_t)); arix_free(on_stack, n*sizeof(int)); break; }
        memset(on_stack, 0, n * sizeof(int));
        size_t sp = 0;
        stack[sp++] = i;
        on_stack[i] = 1;

        while (sp > 0) {
            size_t idx = stack[sp - 1];
            ArixVariable* var = tape->vars[idx];
            int all_parents_visited = 1;
            if (var && var->parents) {
                for (size_t p = 0; p < var->num_parents; p++) {
                    ArixVariable* parent = var->parents[p];
                    if (!parent) continue;
                    /* Find parent index in tape vars */
                    for (size_t pi = 0; pi < n; pi++) {
                        if (tape->vars[pi] == parent && !visited[pi]) {
                            if (!on_stack[pi]) {
                                /* Grow stack if needed */
                                if (sp >= stack_cap) {
                                    stack_cap *= 2;
                                    size_t* ns = (size_t*)arix_malloc(stack_cap * sizeof(size_t), 64);
                                    if (!ns) break;
                                    memcpy(ns, stack, sp * sizeof(size_t));
                                    arix_free(stack, stack_cap/2 * sizeof(size_t));
                                    stack = ns;
                                }
                                stack[sp++] = pi;
                                on_stack[pi] = 1;
                                all_parents_visited = 0;
                            }
                            break;
                        }
                    }
                    if (!all_parents_visited) break;
                }
            }
            if (all_parents_visited) {
                visited[idx] = 1;
                sorted[(*num_sorted)++] = var;
                on_stack[idx] = 0;
                sp--;
            }
        }
        arix_free(on_stack, n * sizeof(int));
        arix_free(stack, stack_cap * sizeof(size_t));
    }
    arix_free(visited, n * sizeof(int));
}

void arix_tape_backward(ArixTape* tape, ArixVariable* loss) {
    if (!tape || !loss) return;

    if (loss->grad) arix_tensor_destroy(loss->grad);
    loss->grad = arix_tensor_ones(loss->data->shape, loss->data->ndim, ARIX_FLOAT32);

    ArixVariable** sorted = NULL;
    size_t n = tape->num_vars;
    size_t num_sorted = 0;
    if (n > 0) {
        sorted = (ArixVariable**)arix_malloc(n * sizeof(ArixVariable*), 64);
        if (sorted) {
            tape_topological_sort(tape, sorted, &num_sorted);
        }
    }

    for (size_t i = num_sorted; i > 0; i--) {
        ArixVariable* var = sorted ? sorted[i - 1] : tape->vars[i - 1];
        if (!var || !var->backward_fn || !var->grad) continue;
        if (!var->backward_ctx && var->checkpointed && var->recompute_ctx) {
            var->backward_ctx = var->recompute_ctx(var, var->params, var->param_count);
        }
        if (!var->backward_ctx) continue;
        var->backward_fn(var->backward_ctx, var->grad);
        if (var->backward_ctx && var->free_ctx) {
            var->free_ctx(var->backward_ctx);
            var->backward_ctx = NULL;
        }
        for (size_t p = 0; p < var->num_parents; p++) {
            ArixVariable* parent = var->parents[p];
            if (!parent) continue;
            parent->ref_count--;
        }
    }

    if (sorted) arix_free(sorted, n * sizeof(ArixVariable*));
}

void arix_tape_zero_grad(ArixTape* tape) {
    if (!tape) return;
    for (size_t i = 0; i < tape->num_vars; i++) {
        if (tape->vars[i] && tape->vars[i]->grad) {
            arix_tensor_destroy(tape->vars[i]->grad);
            tape->vars[i]->grad = NULL;
        }
    }
}

float arix_tape_global_norm(ArixTape* tape) {
    if (!tape) return 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < tape->num_vars; i++) {
        ArixVariable* var = tape->vars[i];
        if (!var || !var->grad) continue;
        float* gd = (float*)var->grad->data;
        size_t sz = var->grad->size;
        for (size_t j = 0; j < sz; j++) sum_sq += gd[j] * gd[j];
    }
    return sqrtf(sum_sq);
}

void arix_tape_clip_grad_norm(ArixTape* tape, float max_norm) {
    if (!tape || max_norm <= 0.0f) return;
    float total_norm = arix_tape_global_norm(tape);
    if (total_norm <= max_norm) return;
    float scale = max_norm / (total_norm + 1e-7f);
    for (size_t i = 0; i < tape->num_vars; i++) {
        ArixVariable* var = tape->vars[i];
        if (!var || !var->grad) continue;
        float* gd = (float*)var->grad->data;
        for (size_t j = 0; j < var->grad->size; j++) gd[j] *= scale;
    }
}
