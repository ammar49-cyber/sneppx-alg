#include "fractal_memory_orchestrator.h"
#include "automatic_differentiation_framework.h"

size_t arix_fm_get_params(const ArixFMController* ctrl, ArixTensor** out, size_t max_out) {
    (void)ctrl;
    (void)out;
    (void)max_out;
    return 0;
}

int arix_fm_build_train_graph(ArixFMController* ctrl, ArixTape* tape,
                               ArixVariable* input_var,
                               ArixVariable** weight_vars, size_t num_weights,
                               ArixVariable** output_var) {
    (void)ctrl;
    (void)tape;
    (void)weight_vars;
    (void)num_weights;
    if (!input_var || !output_var) return -1;
    *output_var = input_var;
    return 0;
}
