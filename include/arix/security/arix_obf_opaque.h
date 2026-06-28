#pragma once
#ifndef ARIX_OBF_OPAQUE_H
#define ARIX_OBF_OPAQUE_H

#include "arix_obf_cfg.h"
#include <random>

namespace arix {

struct ArixOpaquePredicate {
    std::string condition_expr;
    bool always_true;
    uint64_t id;
};

class ArixOpaqueEngine {
public:
    ArixOpaqueEngine();
    void set_seed(uint64_t seed);

    ArixOpaquePredicate generate_math_predicate();
    ArixOpaquePredicate generate_pointer_predicate();
    ArixOpaquePredicate generate_loop_predicate();
    void insert_predicate(ArixObfBlock& block, bool desired_outcome, ArixOpaquePredicate& pred);
    void insert_predicates_to_cfg(ArixObfCFG& cfg);

private:
    std::mt19937_64 rng;
    uint64_t next_id;
};

inline bool arix_opaque_always_true() {
    volatile uintptr_t x = 0;
    return (x * x) == 0;
}

inline bool arix_opaque_math_true() {
    volatile int64_t a = 7, b = 3;
    return (a*a + b*b) * 4 == (a+b)*(a+b) + (a-b)*(a-b);
}

inline bool arix_opaque_pointer_true() {
    volatile int x = 0;
    return ((uintptr_t)&x ^ (uintptr_t)&x) == 0;
}

} // namespace arix

#endif // ARIX_OBF_OPAQUE_H
