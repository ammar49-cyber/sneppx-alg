#pragma once
#ifndef SNEPPX_OBF_OPAQUE_H
#define SNEPPX_OBF_OPAQUE_H

#include "control_flow_obfuscation.h"
#include <random>

namespace SNEPPX {

struct SNEPPXOpaquePredicate {
    std::string condition_expr;
    bool always_true;
    uint64_t id;
};

class SNEPPXOpaqueEngine {
public:
    SNEPPXOpaqueEngine();
    void set_seed(uint64_t seed);

    SNEPPXOpaquePredicate generate_math_predicate();
    SNEPPXOpaquePredicate generate_pointer_predicate();
    SNEPPXOpaquePredicate generate_loop_predicate();
    void insert_predicate(SNEPPXObfBlock& block, bool desired_outcome, SNEPPXOpaquePredicate& pred);
    void insert_predicates_to_cfg(SNEPPXObfCFG& cfg);

private:
    std::mt19937_64 rng;
    uint64_t next_id;
};

inline bool SNEPPX_opaque_always_true() {
    volatile uintptr_t x = 0;
    return (x * x) == 0;
}

inline bool SNEPPX_opaque_math_true() {
    volatile int64_t a = 7, b = 3;
    return (a*a + b*b) * 4 == (a+b)*(a+b) + (a-b)*(a-b);
}

inline bool SNEPPX_opaque_pointer_true() {
    volatile int x = 0;
    return ((uintptr_t)&x ^ (uintptr_t)&x) == 0;
}

} // namespace SNEPPX

#endif // SNEPPX_OBF_OPAQUE_H
