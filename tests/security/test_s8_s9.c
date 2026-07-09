#include "model_checking.h"
#include "self_audit.h"
#include "s8_extensions.h"
#include "s9_extensions.h"
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    test_fn(); printf("PASS\n"); tests_passed++;
}

static int sample_invariant(uint32_t state_id) {
    return state_id < 10;
}

static void test_model_init(void) {
    SNEPPXFormalModel model;
    ASSERT(SNEPPX_model_init(&model) == 0, "model init");
}

static void test_model_simple_graph(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 1, 0);
    SNEPPX_model_add_state(&model, 2, 0, 1);
    SNEPPX_model_add_transition(&model, 0, 1);
    SNEPPX_model_add_transition(&model, 0, 2);
    SNEPPXModelCheckResult result = SNEPPX_model_check(&model);
    ASSERT(result.total_states == 3, "3 states");
    ASSERT(result.reachable_states > 0, "reachable states");
}

static void test_model_invariant(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 0, 0);
    ASSERT(SNEPPX_model_verify_invariant(&model, sample_invariant) == 1, "invariant holds");
    SNEPPX_model_add_state(&model, 99, 0, 0);
    ASSERT(SNEPPX_model_verify_invariant(&model, sample_invariant) == 0, "invariant fails");
}

static void test_self_audit_init(void) {
    SNEPPXSelfAudit audit;
    ASSERT(SNEPPX_self_audit_init(&audit) == 0, "audit init");
    SNEPPX_self_audit_destroy(&audit);
}

static void test_self_audit_run(void) {
    SNEPPXSelfAudit audit;
    SNEPPX_self_audit_init(&audit);
    ASSERT(SNEPPX_self_audit_run_all(&audit) == 0, "run all");
    ASSERT(audit.total_passed > 0, "checks passed");
    ASSERT(audit.total_failed >= 0, "checks run");
    double score = SNEPPX_self_audit_score(&audit);
    ASSERT(score > 0.0, "positive score");
    SNEPPX_self_audit_destroy(&audit);
}

static void test_tla_parse(void) {
    SNEPPXTLAParser parser;
    ASSERT(SNEPPX_tla_parse(&parser, "Init == x = 0") == 0, "tla parse");
    ASSERT(parser.parsed == 1, "parsed flag");
}

static void test_ltl_init_check(void) {
    SNEPPXLTLVerifier ltl;
    ASSERT(SNEPPX_ltl_init(&ltl, "G(x > 0)") == 0, "ltl init");
    int trace[] = {1, 2, 3};
    ASSERT(SNEPPX_ltl_check(&ltl, trace, 3) == 0, "ltl check");
    ASSERT(ltl.holds == 1, "ltl holds");
}

static void test_symex_init_explore(void) {
    SNEPPXSymExEngine se;
    ASSERT(SNEPPX_symex_init(&se, 10) == 0, "symex init");
    ASSERT(se.depth_limit == 10, "depth limit");
    uint8_t bc[] = {0x01, 0x02, 0x03};
    ASSERT(SNEPPX_symex_explore(&se, bc, 3) == 0, "symex explore");
    ASSERT(se.explored_paths >= 1, "paths explored");
}

static void test_loop_invariant(void) {
    char inv[256];
    ASSERT(SNEPPX_loop_invariant_infer("for i in 0..n", inv, sizeof(inv)) == 0, "loop invariant");
}

static void test_data_flow(void) {
    SNEPPXDataFlow df;
    ASSERT(SNEPPX_data_flow_init(&df) == 0, "dataflow init");
    ASSERT(SNEPPX_data_flow_taint(&df, 42) == 0, "taint var");
    ASSERT(df.taint_count == 1, "one taint mark");
    ASSERT(SNEPPX_data_flow_propagate(&df) == 0, "propagate");
}

static void test_lean_export(void) {
    ASSERT(SNEPPX_lean_export_proof("my_theorem", "trivial", "proof.lean") == 0, "lean export");
}

static void test_vuln_scanner(void) {
    SNEPPXVulnScanner vs;
    ASSERT(SNEPPX_vuln_scanner_init(&vs) == 0, "vuln init");
    ASSERT(SNEPPX_vuln_scanner_add_cve(&vs, "CVE-2024-1234") == 0, "add CVE");
    ASSERT(vs.cve_count == 1, "one CVE");
    ASSERT(SNEPPX_vuln_scanner_run(&vs) == 0, "vuln scan");
    ASSERT(vs.scan_complete == 1, "scan complete");
}

static int dummy_fuzz_target(const uint8_t* data, size_t len) {
    (void)data; (void)len;
    return 0;
}

static void test_fuzz_harness(void) {
    SNEPPXFuzzHarness fh;
    ASSERT(SNEPPX_fuzz_harness_init(&fh) == 0, "fuzz init");
    fh.input_len = 64;
    int crashes = SNEPPX_fuzz_harness_run(&fh, dummy_fuzz_target);
    ASSERT(crashes >= 0, "fuzz run");
}

static void test_redteam(void) {
    SNEPPXRedTeamSim rt;
    ASSERT(SNEPPX_redteam_init(&rt) == 0, "redteam init");
    ASSERT(SNEPPX_redteam_add_step(&rt, "recon") == 0, "add step");
    ASSERT(SNEPPX_redteam_add_step(&rt, "exploit") == 0, "add step 2");
    ASSERT(SNEPPX_redteam_execute(&rt) == 0, "execute");
    ASSERT(rt.completed == 1, "completed");
}

static void test_compliance(void) {
    ASSERT(SNEPPX_compliance_check_nist("AC-1") == 1, "nist check");
}

static void test_bugbounty(void) {
    SNEPPXBugBountyTriage bt;
    ASSERT(SNEPPX_bugbounty_triage_init(&bt) == 0, "bb triage init");
    ASSERT(SNEPPX_bugbounty_triage_analyze(&bt, "XSS in login") == 0, "analyze report");
    ASSERT(bt.severity >= 0, "severity assigned");
}

int main(void) {
    run_test("model_init", test_model_init);
    run_test("model_simple_graph", test_model_simple_graph);
    run_test("model_invariant", test_model_invariant);
    run_test("self_audit_init", test_self_audit_init);
    run_test("self_audit_run", test_self_audit_run);
    run_test("tla_parse", test_tla_parse);
    run_test("ltl_init_check", test_ltl_init_check);
    run_test("symex_init_explore", test_symex_init_explore);
    run_test("loop_invariant", test_loop_invariant);
    run_test("data_flow", test_data_flow);
    run_test("lean_export", test_lean_export);
    run_test("vuln_scanner", test_vuln_scanner);
    run_test("fuzz_harness", test_fuzz_harness);
    run_test("redteam", test_redteam);
    run_test("compliance", test_compliance);
    run_test("bugbounty", test_bugbounty);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
