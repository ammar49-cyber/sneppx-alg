#include "differential_privacy.h"
#include <stdio.h>
#include <math.h>

int main() {
    dp_stats_t stats;
    double val = arix_dp_laplace_mech(10.0, 1.0, 1.0);
    printf("Laplace(10.0, 1.0, 1.0) = %.4f\n", val);
    val = arix_dp_gaussian_mech(10.0, 1.0, 1.0, 1e-5);
    printf("Gaussian(10.0, 1.0, 1.0, 1e-5) = %.4f\n", val);
    arix_dp_set_budget(1.0, 1e-5);
    double params[4] = {1.0, 2.0, 3.0, 4.0};
    double grads[4] = {0.1, 0.2, 0.3, 0.4};
    arix_dp_dpsgd_step(params, grads, 4, 0.01, 1.0, 1e-5, 1.0);
    printf("DP-SGD step OK\n");
    if (arix_dp_get_stats(&stats) == 0) {
        printf("Budget remaining: eps=%.4f del=%.4e\n", stats.budget_remaining_epsilon, stats.budget_remaining_delta);
    }
    printf("PASS: Differential privacy OK\n");
    return 0;
}
