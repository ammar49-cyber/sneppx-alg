#include "adversarial_robustness_certification.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

int SNEPPX_arc_build_train_graph(SNEPPXARCLayer* layer, SNEPPXTape* tape,
                                SNEPPXVariable* input_var,
                                SNEPPXVariable** weight_vars, size_t num_weights,
                                SNEPPXVariable** output_var) {
    if (!layer || !tape || !input_var || !weight_vars || !output_var) return -1;
    size_t n_verifier_layers = layer->output_verifier->num_layers;
    size_t expected = 1 + 2 * n_verifier_layers;
    if (num_weights < expected) return -1;

    size_t wi = 0;
    SNEPPXVariable* proj_w = weight_vars[wi++];

    /* Input guard: project through projection_matrix (always sanitize during training) */
    SNEPPXVariable* proj_w_t = SNEPPX_transpose(tape, proj_w, 0, 1);
    SNEPPXVariable* current = SNEPPX_matmul(tape, input_var, proj_w_t);

    /* Output verifier: MLP with ReLU */
    for (size_t l = 0; l < n_verifier_layers; l++) {
        SNEPPXVariable* w = weight_vars[wi++];
        SNEPPXVariable* b = weight_vars[wi++];
        SNEPPXVariable* w_t = SNEPPX_transpose(tape, w, 0, 1);
        SNEPPXVariable* out = SNEPPX_matmul(tape, current, w_t);
        out = SNEPPX_add(tape, out, b);
        current = SNEPPX_relu(tape, out);
    }

    *output_var = current;
    return 0;
}
