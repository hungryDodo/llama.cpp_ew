#ifndef LLAMA_API
#define LLAMA_API
#endif

#include <cstddef>
#include <llamaedge/hooks.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
        printf("  PASS: %s\n", msg); \
    } else { \
        printf("  FAIL: %s\n", msg); \
    } \
} while(0)

void test_b4_profiles() {
    printf("\n=== Test: B4 Profiles ===\n");

    ASSERT(LLAMAEDGE_PROFILE_CONSERVATIVE == 0, "CONSERVATIVE = 0");
    ASSERT(LLAMAEDGE_PROFILE_BALANCED == 1, "BALANCED = 1");
    ASSERT(LLAMAEDGE_PROFILE_AGGRESSIVE == 2, "AGGRESSIVE = 2");
}

void test_b4_thinning_policy() {
    printf("\n=== Test: B4 Thinning Policy ===\n");

    ASSERT(LLAMAEDGE_THIN_NONE == 0, "THIN_NONE = 0");
    ASSERT(LLAMAEDGE_THIN_L2_NORM == 1, "THIN_L2_NORM = 1");
}

void test_b4_quant_bitplan() {
    printf("\n=== Test: B4 Quant Bitplan ===\n");

    ASSERT(LLAMAEDGE_Q3_K4V3 == 0, "Q3_K4V3 = 0");
    ASSERT(LLAMAEDGE_Q4_K4V4 == 1, "Q4_K4V4 = 1");
    ASSERT(LLAMAEDGE_Q4_K4V3 == 2, "Q4_K4V3 = 2");
}

void test_b4_config_creation() {
    printf("\n=== Test: B4 Config Creation ===\n");

    llamaedge_b4_config config = {};
    config.profile = LLAMAEDGE_PROFILE_BALANCED;
    config.thinning.enabled = true;
    config.thinning.policy = LLAMAEDGE_THIN_L2_NORM;
    config.thinning.top_k = 128;
    config.thinning.top_ratio = 0.0f;
    config.thinning.protected_token_count = 5;
    config.quant.enabled = true;
    config.quant.bitplan = LLAMAEDGE_Q3_K4V3;
    config.quant.use_wht = true;
    config.warm.n_warm_layers = 4;
    config.warm.n_warm_chunks = 2;
    config.warm.chunk_size_tokens = 256;
    config.chunk_size_tokens = 512;

    ASSERT(config.profile == LLAMAEDGE_PROFILE_BALANCED, "Profile is BALANCED");
    ASSERT(config.thinning.enabled == true, "Thinning is enabled");
    ASSERT(config.thinning.policy == LLAMAEDGE_THIN_L2_NORM, "Thinning policy is L2_NORM");
    ASSERT(config.thinning.top_k == 128, "Thinning top_k = 128");
    ASSERT(config.thinning.protected_token_count == 5, "Protected token count = 5");
    ASSERT(config.quant.enabled == true, "Quantization is enabled");
    ASSERT(config.quant.bitplan == LLAMAEDGE_Q3_K4V3, "Quant bitplan is Q3_K4V3");
    ASSERT(config.quant.use_wht == true, "WHT is enabled");
    ASSERT(config.warm.n_warm_layers == 4, "Warm layers = 4");
    ASSERT(config.warm.n_warm_chunks == 2, "Warm chunks = 2");
    ASSERT(config.warm.chunk_size_tokens == 256, "Warm chunk size = 256");
    ASSERT(config.chunk_size_tokens == 512, "Chunk size = 512");
}

void test_b4_thinning_config() {
    printf("\n=== Test: B4 Thinning Config ===\n");

    llamaedge_thinning_config thin_cfg = {};
    thin_cfg.enabled = true;
    thin_cfg.policy = LLAMAEDGE_THIN_L2_NORM;
    thin_cfg.top_k = 64;
    thin_cfg.top_ratio = 0.0f;
    thin_cfg.protected_token_count = 10;

    ASSERT(thin_cfg.enabled == true, "Thinning enabled");
    ASSERT(thin_cfg.policy == LLAMAEDGE_THIN_L2_NORM, "Policy is L2_NORM");
    ASSERT(thin_cfg.top_k == 64, "Top K is 64");
    ASSERT(thin_cfg.protected_token_count == 10, "Protected count is 10");
}

void test_b4_quant_config() {
    printf("\n=== Test: B4 Quant Config ===\n");

    llamaedge_quant_config quant_cfg = {};
    quant_cfg.enabled = true;
    quant_cfg.bitplan = LLAMAEDGE_Q4_K4V4;
    quant_cfg.use_wht = false;

    ASSERT(quant_cfg.enabled == true, "Quant enabled");
    ASSERT(quant_cfg.bitplan == LLAMAEDGE_Q4_K4V4, "Bitplan is Q4_K4V4");
    ASSERT(quant_cfg.use_wht == false, "WHT disabled");
}

void test_b4_warm_config() {
    printf("\n=== Test: B4 Warm Config ===\n");

    llamaedge_warm_config warm_cfg = {};
    warm_cfg.n_warm_layers = 8;
    warm_cfg.n_warm_chunks = 3;
    warm_cfg.chunk_size_tokens = 128;

    ASSERT(warm_cfg.n_warm_layers == 8, "Warm layers = 8");
    ASSERT(warm_cfg.n_warm_chunks == 3, "Warm chunks = 3");
    ASSERT(warm_cfg.chunk_size_tokens == 128, "Chunk size = 128");
}

int main() {
    printf("========================================\n");
    printf("B4 EdgeWeaver Test Suite\n");
    printf("========================================\n");

    test_b4_profiles();
    test_b4_thinning_policy();
    test_b4_quant_bitplan();
    test_b4_config_creation();
    test_b4_thinning_config();
    test_b4_quant_config();
    test_b4_warm_config();

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("========================================\n");

    return tests_passed == tests_run ? 0 : 1;
}
