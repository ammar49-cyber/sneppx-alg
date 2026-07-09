#include "fractal_memory_orchestrator.h"
#include "automatic_differentiation_framework.h"

size_t SNEPPX_fm_get_params(const SNEPPXFMController* ctrl, SNEPPXTensor** out, size_t max_out) {
    (void)ctrl;
    (void)out;
    (void)max_out;
    return 0;
}

int SNEPPX_fm_build_train_graph(SNEPPXFMController* ctrl, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var) {
    (void)ctrl;
    (void)tape;
    (void)weight_vars;
    (void)num_weights;
    if (!input_var || !output_var) return -1;
    *output_var = input_var;
    return 0;
}
