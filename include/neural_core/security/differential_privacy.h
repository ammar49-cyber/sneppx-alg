#ifndef ARIX_DIFFERENTIAL_PRIVACY_H
#define ARIX_DIFFERENTIAL_PRIVACY_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    double epsilon;
    double delta;
} dp_budget_t;

typedef struct {
    int id;
    double epsilon;
    double delta;
    double sensitivity;
    char name[64];
} dp_mechanism_t;

typedef struct {
    int id;
    int indices[64];
    int num_mechanisms;
    double total_epsilon;
    double total_delta;
} dp_composition_t;

typedef struct {
    double total_epsilon_consumed;
    double total_delta_consumed;
    int num_mechanisms;
    int num_compositions;
    double budget_remaining_epsilon;
    double budget_remaining_delta;
} dp_stats_t;

double arix_dp_laplace_mech(double value, double sensitivity, double epsilon);
double arix_dp_gaussian_mech(double value, double sensitivity, double epsilon, double delta);
int arix_dp_register_mechanism(dp_mechanism_t *mech);
int arix_dp_compose(dp_composition_t *comp);
int arix_dp_sequential_compose(double *epsilons, double *deltas, int n, double *total_eps, double *total_del);
int arix_dp_advanced_compose(double *epsilons, double *deltas, int n, double delta, double *total_eps, double *total_del);
int arix_dp_set_budget(double epsilon, double delta);
int arix_dp_check_budget(double epsilon, double delta);
int arix_dp_reset_budget(void);
int arix_dp_dpsgd_step(double *params, const double *gradients, int n, double lr, double epsilon, double delta, double clip_norm);
int arix_dp_noise_layer(double *data, int n, double sensitivity, double epsilon, double delta);
int arix_dp_get_stats(dp_stats_t *stats);

#endif
