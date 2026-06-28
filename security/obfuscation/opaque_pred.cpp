#include "arix_obf_opaque.h"
#include <sstream>
#include <algorithm>

namespace arix {

ArixOpaqueEngine::ArixOpaqueEngine() : rng(std::random_device{}()), next_id(1) {}

void ArixOpaqueEngine::set_seed(uint64_t seed) {
    rng.seed(seed);
}

ArixOpaquePredicate ArixOpaqueEngine::generate_math_predicate() {
    ArixOpaquePredicate pred;
    pred.id = next_id++;
    pred.always_true = true;

    int64_t a = static_cast<int64_t>(rng() % 100) + 1;
    int64_t b = static_cast<int64_t>(rng() % 100) + 1;

    std::stringstream ss;
    ss << "(" << a << "*" << a << " + " << b << "*" << b << ") * 4 == "
       << "(" << a << "+" << b << ")*(" << a << "+" << b << ") + "
       << "(" << a << "-" << b << ")*(" << a << "-" << b << ")";
    pred.condition_expr = ss.str();

    return pred;
}

ArixOpaquePredicate ArixOpaqueEngine::generate_pointer_predicate() {
    ArixOpaquePredicate pred;
    pred.id = next_id++;
    pred.always_true = true;
    pred.condition_expr = "(&local_var ^ &local_var) == 0";
    return pred;
}

ArixOpaquePredicate ArixOpaqueEngine::generate_loop_predicate() {
    ArixOpaquePredicate pred;
    pred.id = next_id++;
    pred.always_true = true;
    pred.condition_expr = "complex_loop_always_returns_true()";
    return pred;
}

void ArixOpaqueEngine::insert_predicate(ArixObfBlock& block, bool desired_outcome, ArixOpaquePredicate& pred) {
    ArixObfInstruction cond_inst;
    cond_inst.type = ArixObfInstType::CMP;
    cond_inst.result = "_opaque_" + std::to_string(pred.id);
    cond_inst.operand1 = pred.condition_expr;
    cond_inst.operand2 = desired_outcome ? "1" : "0";

    ArixObfInstruction branch_inst;
    branch_inst.type = desired_outcome ? ArixObfInstType::JZ : ArixObfInstType::JNZ;
    branch_inst.operand1 = "_opaque_skip_" + std::to_string(pred.id);

    block.instructions.insert(block.instructions.begin(), branch_inst);
    block.instructions.insert(block.instructions.begin(), cond_inst);

    ArixObfInstruction opaque_marker;
    opaque_marker.type = ArixObfInstType::NOP;
    opaque_marker.operand1 = "_opaque_end_" + std::to_string(pred.id);
    block.instructions.push_back(opaque_marker);
}

void ArixOpaqueEngine::insert_predicates_to_cfg(ArixObfCFG& cfg) {
    for (auto& pair : cfg.blocks) {
        auto& block = pair.second;
        if (block->is_entry || block->is_exit) continue;

        auto pred = generate_math_predicate();
        insert_predicate(*block, true, pred);

        if (rng() % 2 == 0) {
            auto pred2 = generate_pointer_predicate();
            insert_predicate(*block, true, pred2);
        }
    }
}

} // namespace arix
