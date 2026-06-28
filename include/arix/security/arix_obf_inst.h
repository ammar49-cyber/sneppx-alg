#pragma once
#ifndef ARIX_OBF_INST_H
#define ARIX_OBF_INST_H

#include "arix_obf_cfg.h"
#include <random>

namespace arix {

class ArixObfSubst {
public:
    ArixObfSubst();
    void set_seed(uint64_t seed);
    void substitute_add(ArixObfBlock& block);
    void substitute_logic(ArixObfBlock& block);
    void substitute_compare(ArixObfBlock& block);
    void substitute_all(ArixObfBlock& block);
    void substitute_all_blocks(ArixObfCFG& cfg);

private:
    std::mt19937_64 rng;
    bool choose_substitution();

    ArixObfInstruction make_lea_add(const ArixObfInstruction& inst);
    std::vector<ArixObfInstruction> make_neg_sub_add(const ArixObfInstruction& inst);
    std::vector<ArixObfInstruction> make_mul_shift_add(const ArixObfInstruction& inst);
    std::vector<ArixObfInstruction> make_nand_and(const ArixObfInstruction& inst);
    std::vector<ArixObfInstruction> make_nand_or(const ArixObfInstruction& inst);
    std::vector<ArixObfInstruction> make_nand_xor(const ArixObfInstruction& inst);
    std::vector<ArixObfInstruction> make_sub_cmp(const ArixObfInstruction& inst);
};

} // namespace arix

#endif // ARIX_OBF_INST_H
