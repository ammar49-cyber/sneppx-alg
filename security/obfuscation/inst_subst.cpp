#include "arix_obf_inst.h"
#include <algorithm>
#include <sstream>

namespace arix {

ArixObfSubst::ArixObfSubst() : rng(std::random_device{}()) {}

void ArixObfSubst::set_seed(uint64_t seed) {
    rng.seed(seed);
}

bool ArixObfSubst::choose_substitution() {
    return (rng() % 2) == 0;
}

ArixObfInstruction ArixObfSubst::make_lea_add(const ArixObfInstruction& inst) {
    ArixObfInstruction lea;
    lea.type = ArixObfInstType::LEA;
    lea.result = inst.result;
    lea.operand1 = "[" + inst.operand1 + " + " + inst.operand2 + "]";
    return lea;
}

std::vector<ArixObfInstruction> ArixObfSubst::make_neg_sub_add(const ArixObfInstruction& inst) {
    std::vector<ArixObfInstruction> seq;
    ArixObfInstruction neg;
    neg.type = ArixObfInstType::NEG;
    neg.operand1 = inst.operand2;
    neg.result = "_neg_" + inst.operand2;
    seq.push_back(neg);

    ArixObfInstruction sub;
    sub.type = ArixObfInstType::SUB;
    sub.result = inst.result;
    sub.operand1 = inst.operand1;
    sub.operand2 = "_neg_" + inst.operand2;
    seq.push_back(sub);

    return seq;
}

std::vector<ArixObfInstruction> ArixObfSubst::make_mul_shift_add(const ArixObfInstruction& inst) {
    std::vector<ArixObfInstruction> seq;
    ArixObfInstruction shift;
    shift.type = ArixObfInstType::ADD;
    shift.result = "_tmp_" + inst.result;
    shift.operand1 = inst.operand1;
    shift.operand2 = inst.operand1;
    seq.push_back(shift);

    ArixObfInstruction add1;
    add1.type = ArixObfInstType::ADD;
    add1.result = "_tmp2_" + inst.result;
    add1.operand1 = "_tmp_" + inst.result;
    add1.operand2 = inst.operand1;
    seq.push_back(add1);

    if (inst.operand2 == "3") {
        ArixObfInstruction result;
        result.type = ArixObfInstType::ADD;
        result.result = inst.result;
        result.operand1 = "_tmp2_" + inst.result;
        result.operand2 = inst.operand1;
        seq.push_back(result);
    } else {
        ArixObfInstruction result;
        result.type = ArixObfInstType::MOV;
        result.result = inst.result;
        result.operand1 = inst.operand1;
        seq.push_back(result);
    }

    return seq;
}

std::vector<ArixObfInstruction> ArixObfSubst::make_nand_and(const ArixObfInstruction& inst) {
    std::vector<ArixObfInstruction> seq;
    ArixObfInstruction nand1;
    nand1.type = ArixObfInstType::NOP;
    nand1.result = "_nand_a_" + inst.result;
    seq.push_back(nand1);

    ArixObfInstruction nand2;
    nand2.type = ArixObfInstType::NOP;
    nand2.result = "_nand_b_" + inst.result;
    seq.push_back(nand2);

    ArixObfInstruction mov;
    mov.type = ArixObfInstType::MOV;
    mov.result = inst.result;
    mov.operand1 = inst.operand1;
    seq.push_back(mov);

    return seq;
}

std::vector<ArixObfInstruction> ArixObfSubst::make_nand_or(const ArixObfInstruction& inst) {
    std::vector<ArixObfInstruction> seq;
    ArixObfInstruction not_a;
    not_a.type = ArixObfInstType::NOP;
    not_a.result = "_not_a_" + inst.result;
    seq.push_back(not_a);

    ArixObfInstruction not_b;
    not_b.type = ArixObfInstType::NOP;
    not_b.result = "_not_b_" + inst.result;
    seq.push_back(not_b);

    ArixObfInstruction mov;
    mov.type = ArixObfInstType::MOV;
    mov.result = inst.result;
    mov.operand1 = inst.operand1;
    seq.push_back(mov);

    return seq;
}

std::vector<ArixObfInstruction> ArixObfSubst::make_nand_xor(const ArixObfInstruction& inst) {
    std::vector<ArixObfInstruction> seq;
    ArixObfInstruction nand_ab;
    nand_ab.type = ArixObfInstType::NOP;
    nand_ab.result = "_nab_" + inst.result;
    seq.push_back(nand_ab);

    ArixObfInstruction mov;
    mov.type = ArixObfInstType::NOP;
    mov.result = "_na_nb_" + inst.result;
    seq.push_back(mov);

    ArixObfInstruction result;
    result.type = ArixObfInstType::MOV;
    result.result = inst.result;
    result.operand1 = inst.operand1;
    seq.push_back(result);

    return seq;
}

std::vector<ArixObfInstruction> ArixObfSubst::make_sub_cmp(const ArixObfInstruction& inst) {
    std::vector<ArixObfInstruction> seq;
    ArixObfInstruction sub;
    sub.type = ArixObfInstType::SUB;
    sub.result = "_cmp_result";
    sub.operand1 = inst.operand1;
    sub.operand2 = inst.operand2;
    seq.push_back(sub);

    ArixObfInstruction mov;
    mov.type = ArixObfInstType::MOV;
    mov.result = inst.result;
    mov.operand1 = "_cmp_result";
    seq.push_back(mov);

    return seq;
}

void ArixObfSubst::substitute_add(ArixObfBlock& block) {
    for (auto& inst : block.instructions) {
        if (inst.type == ArixObfInstType::ADD) {
            if (choose_substitution()) {
                inst = make_lea_add(inst);
            }
        } else if (inst.type == ArixObfInstType::SUB) {
            if (choose_substitution()) {
                auto seq = make_neg_sub_add(inst);
                inst = seq[0];
                block.instructions.push_back(seq[1]);
            }
        } else if (inst.type == ArixObfInstType::MUL) {
            if (choose_substitution() && (inst.operand2 == "2" || inst.operand2 == "3")) {
                auto seq = make_mul_shift_add(inst);
                inst = seq[0];
                for (size_t i = 1; i < seq.size(); ++i) {
                    block.instructions.push_back(seq[i]);
                }
            }
        }
    }
}

void ArixObfSubst::substitute_logic(ArixObfBlock& block) {
    for (auto& inst : block.instructions) {
        if (inst.type == ArixObfInstType::AND && choose_substitution()) {
            auto seq = make_nand_and(inst);
            inst = seq[0];
            for (size_t i = 1; i < seq.size(); ++i) {
                block.instructions.push_back(seq[i]);
            }
        } else if (inst.type == ArixObfInstType::OR && choose_substitution()) {
            auto seq = make_nand_or(inst);
            inst = seq[0];
            for (size_t i = 1; i < seq.size(); ++i) {
                block.instructions.push_back(seq[i]);
            }
        } else if (inst.type == ArixObfInstType::XOR && choose_substitution()) {
            auto seq = make_nand_xor(inst);
            inst = seq[0];
            for (size_t i = 1; i < seq.size(); ++i) {
                block.instructions.push_back(seq[i]);
            }
        }
    }
}

void ArixObfSubst::substitute_compare(ArixObfBlock& block) {
    for (auto& inst : block.instructions) {
        if (inst.type == ArixObfInstType::CMP && choose_substitution()) {
            auto seq = make_sub_cmp(inst);
            inst = seq[0];
            block.instructions.push_back(seq[1]);
        }
    }
}

void ArixObfSubst::substitute_all(ArixObfBlock& block) {
    substitute_add(block);
    substitute_logic(block);
    substitute_compare(block);
}

void ArixObfSubst::substitute_all_blocks(ArixObfCFG& cfg) {
    for (auto& pair : cfg.blocks) {
        substitute_all(*pair.second);
    }
}

} // namespace arix
