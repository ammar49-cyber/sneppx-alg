#include "adversarial_robustness_certification.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

int arix_arc_build_train_graph(ArixARCLayer* layer, ArixTape* tape,
                                ArixVariable* input_var,
                                ArixVariable** weight_vars, size_t num_weights,
                                ArixVariable** output_var) {
    if (!layer || !tape || !input_var || !weight_vars || !output_var) return -1;
    size_t n_verifier_layers = layer->output_verifier->num_layers;
    size_t expected = 1 + 2 * n_verifier_layers;
    if (num_weights < expected) return -1;

    size_t wi = 0;
    ArixVariable* proj_w = weight_vars[wi++];

    /* Input guard: project through projection_matrix (always sanitize during training) */
    ArixVariable* proj_w_t = arix_transpose(tape, proj_w, 0, 1);
    ArixVariable* current = arix_matmul(tape, input_var, proj_w_t);

    /* Output verifier: MLP with ReLU */
    for (size_t l = 0; l < n_verifier_layers; l++) {
        ArixVariable* w = weight_vars[wi++];
        ArixVariable* b = weight_vars[wi++];
        ArixVariable* w_t = arix_transpose(tape, w, 0, 1);
        ArixVariable* out = arix_matmul(tape, current, w_t);
        out = arix_add(tape, out, b);
        current = arix_relu(tape, out);
    }

    *output_var = current;
    return 0;
}
