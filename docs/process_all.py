import os, re, sys

BASE = r"C:\Users\PC\sneppx-ultra\ARIX_Algo"

TARGET_FILES = [
    r"algorithms\hss\core\discretize.c",
    r"algorithms\hss\core\forward.c",
    r"algorithms\hss\core\forward_train.c",
    r"algorithms\hss\core\hierarchical.c",
    r"algorithms\hss\core\layer.c",
    r"algorithms\hss\core\model.c",
    r"algorithms\hss\core\scan.c",
    r"algorithms\hss\core\step.c",
    r"algorithms\ser\core\expert.c",
    r"algorithms\ser\core\forward_expert.c",
    r"algorithms\ser\core\forward_layer.c",
    r"algorithms\ser\core\forward_train.c",
    r"algorithms\ser\core\gater.c",
    r"algorithms\ser\core\layer.c",
    r"algorithms\ser\core\loss.c",
    r"algorithms\ser\core\model.c",
    r"algorithms\ser\core\route.c",
    r"algorithms\arc\core\attack_sim.c",
    r"algorithms\arc\core\forward.c",
    r"algorithms\arc\core\forward_train.c",
    r"algorithms\arc\core\gradient_obfuscator.c",
    r"algorithms\arc\core\input_guard.c",
    r"algorithms\arc\core\layer.c",
    r"algorithms\arc\core\output_verifier.c",
    r"algorithms\fm\core\controller.c",
    r"algorithms\fm\core\forward.c",
    r"algorithms\fm\core\forward_train.c",
    r"algorithms\fm\core\memory_bank.c",
    r"algorithms\fm\core\node.c",
    r"algorithms\fm\core\sync.c",
    r"algorithms\npe\core\compiler.c",
    r"algorithms\npe\core\forward_train.c",
    r"algorithms\npe\core\jit_pipeline.c",
    r"algorithms\npe\core\program.c",
    r"algorithms\npe\core\verify.c",
    r"algorithms\npe\core\vm.c",
]

def has_layer1_header(content):
    first_lines = content.split("\n")[:15]
    text = "\n".join(first_lines)
    return "WHAT" in text and "CONCEPT" in text and "ROLE" in text and "REFERENCES" in text

def find_insert_point(content):
    lines = content.split("\n")
    last_include = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#include") or s.startswith("#define") or s.startswith("#ifdef") or s.startswith("#ifndef") or s.startswith("#endif") or s.startswith("#pragma"):
            last_include = i
    if last_include >= 0:
        return last_include + 1
    return 0

def has_doc_block_above(content, func_name):
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if func_name in line and "(" in line:
            for j in range(i - 1, -1, -1):
                prev = lines[j].strip()
                if prev == "":
                    continue
                if prev.endswith("*/"):
                    return True
                if prev.startswith("/*") or prev.startswith("//"):
                    return True
                break
    return False

def add_doxygen_brief(content, func_name, ret_type, params):
    lines = content.split("\n")
    result = []
    added = False
    for i, line in enumerate(lines):
        if not added and func_name in line and "(" in line and ret_type in line:
            brief = f" * @brief {func_name} implementation."
            param_lines = []
            for p in params:
                param_lines.append(f" * @param {p}")
            ret_line = f" * @return 0 on success, -1 on error." if ret_type == "int" else f" * @return void."
            result.append("/**")
            result.append(brief)
            for pl in param_lines:
                result.append(pl)
            result.append(ret_line)
            result.append(" */")
            added = True
        result.append(line)
    return "\n".join(result)

def add_header_if_missing(content, rel_path):
    if has_layer1_header(content):
        return content
    header_map = {
        r"algorithms\hss\core\discretize.c": HEADER_HSS_DISCRETIZE,
        r"algorithms\hss\core\forward.c": HEADER_HSS_FORWARD,
        r"algorithms\hss\core\forward_train.c": HEADER_HSS_FORWARD_TRAIN,
        r"algorithms\hss\core\hierarchical.c": HEADER_HSS_HIERARCHICAL,
        r"algorithms\hss\core\layer.c": HEADER_HSS_LAYER,
        r"algorithms\hss\core\model.c": HEADER_HSS_MODEL,
        r"algorithms\hss\core\scan.c": HEADER_HSS_SCAN,
        r"algorithms\hss\core\step.c": HEADER_HSS_STEP,
        r"algorithms\ser\core\expert.c": HEADER_SER_EXPERT,
        r"algorithms\ser\core\forward_expert.c": HEADER_SER_FORWARD_EXPERT,
        r"algorithms\ser\core\forward_layer.c": HEADER_SER_FORWARD_LAYER,
        r"algorithms\ser\core\forward_train.c": HEADER_SER_FORWARD_TRAIN,
        r"algorithms\ser\core\gater.c": HEADER_SER_GATER,
        r"algorithms\ser\core\layer.c": HEADER_SER_LAYER,
        r"algorithms\ser\core\loss.c": HEADER_SER_LOSS,
        r"algorithms\ser\core\model.c": HEADER_SER_MODEL,
        r"algorithms\ser\core\route.c": HEADER_SER_ROUTE,
        r"algorithms\arc\core\attack_sim.c": HEADER_ARC_ATTACK_SIM,
        r"algorithms\arc\core\forward.c": HEADER_ARC_FORWARD,
        r"algorithms\arc\core\forward_train.c": HEADER_ARC_FORWARD_TRAIN,
        r"algorithms\arc\core\gradient_obfuscator.c": HEADER_ARC_GRADIENT_OBFUSCATOR,
        r"algorithms\arc\core\input_guard.c": HEADER_ARC_INPUT_GUARD,
        r"algorithms\arc\core\layer.c": HEADER_ARC_LAYER,
        r"algorithms\arc\core\output_verifier.c": HEADER_ARC_OUTPUT_VERIFIER,
        r"algorithms\fm\core\controller.c": HEADER_FM_CONTROLLER,
        r"algorithms\fm\core\forward.c": HEADER_FM_FORWARD,
        r"algorithms\fm\core\forward_train.c": HEADER_FM_FORWARD_TRAIN,
        r"algorithms\fm\core\memory_bank.c": HEADER_FM_MEMORY_BANK,
        r"algorithms\fm\core\node.c": HEADER_FM_NODE,
        r"algorithms\fm\core\sync.c": HEADER_FM_SYNC,
        r"algorithms\npe\core\compiler.c": HEADER_NPE_COMPILER,
        r"algorithms\npe\core\forward_train.c": HEADER_NPE_FORWARD_TRAIN,
        r"algorithms\npe\core\jit_pipeline.c": HEADER_NPE_JIT_PIPELINE,
        r"algorithms\npe\core\program.c": HEADER_NPE_PROGRAM,
        r"algorithms\npe\core\verify.c": HEADER_NPE_VERIFY,
        r"algorithms\npe\core\vm.c": HEADER_NPE_VM,
    }
    header = header_map.get(rel_path)
    if not header:
        return content
    insert_at = find_insert_point(content)
    lines = content.split("\n")
    return "\n".join(lines[:insert_at]) + "\n" + header + "\n" + "\n".join(lines[insert_at:])

print("Script loaded, ready to process")
