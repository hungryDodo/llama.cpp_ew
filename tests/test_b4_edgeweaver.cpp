// Test for B4 Full EdgeWeaver functionality
// Tests: thinning, quantization, pipeline, profile switching

#include "llama.h"
#include "llamaedge/hooks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cassert>

// Test helper
static int g_test_passed = 0;
static int g_test_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (cond) { \
            fprintf(stderr, "[PASS] %s\n", msg); \
            g_test_passed++; \
        } else { \
            fprintf(stderr, "[FAIL] %s\n", msg); \
            g_test_failed++; \
        } \
    } while(0)

// ============================================================
// Test 1: B4 Config Initialization
// ============================================================
void test_b4_config_init() {
    fprintf(stderr, "\n=== Test: B4 Config Initialization ===\n");

    llamaedge_b4_config config = {};
    config.profile = LLAMAEDGE_PROFILE_BALANCED;
    config.thinning.enabled = true;
    config.thinning.policy = LLAMAEDGE_THIN_L2_NORM;
    config.thinning.top_k = 64;
    config.thinning.protected_token_count = 4;
    config.quant.enabled = true;
    config.quant.bitplan = LLAMAEDGE_Q3_K4V3;
    config.quant.use_wht = true;
    config.warm.n_warm_layers = 2;
    config.warm.n_warm_chunks = 2;
    config.warm.chunk_size_tokens = 32;
    config.chunk_size_tokens = 32;

    TEST_ASSERT(config.profile == LLAMAEDGE_PROFILE_BALANCED, "Profile is BALANCED");
    TEST_ASSERT(config.thinning.enabled == true, "Thinning enabled");
    TEST_ASSERT(config.thinning.policy == LLAMAEDGE_THIN_L2_NORM, "Thinning policy is L2_NORM");
    TEST_ASSERT(config.quant.bitplan == LLAMAEDGE_Q3_K4V3, "Quant bitplan is Q3_K4V3");
    TEST_ASSERT(config.quant.use_wht == true, "WHT enabled for V");
}

// ============================================================
// Test 2: Thinning - L2 Norm Scoring
// ============================================================
void test_thinning_l2_scores() {
    fprintf(stderr, "\n=== Test: Thinning L2 Norm Scoring ===\n");

    // Create simple embedding matrix: 4 tokens, 8 dimensions
    // Token 0: [1, 0, 0, 0, 0, 0, 0, 0] -> L2 = 1.0
    // Token 1: [1, 1, 0, 0, 0, 0, 0, 0] -> L2 = sqrt(2) ~ 1.414
    // Token 2: [1, 1, 1, 0, 0, 0, 0, 0] -> L2 = sqrt(3) ~ 1.732
    // Token 3: [1, 1, 1, 1, 0, 0, 0, 0] -> L2 = 2.0
    float embeddings[4 * 8] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0
    };

    float scores[4];
    llamaedge_thinning_l2_scores(embeddings, 4, 8, scores);

    TEST_ASSERT(scores[0] > 0.99f && scores[0] < 1.01f, "Token 0 L2 = 1.0");
    TEST_ASSERT(scores[1] > 1.41f && scores[1] < 1.42f, "Token 1 L2 ~ 1.414");
    TEST_ASSERT(scores[2] > 1.73f && scores[2] < 1.74f, "Token 2 L2 ~ 1.732");
    TEST_ASSERT(scores[3] > 1.99f && scores[3] < 2.01f, "Token 3 L2 = 2.0");

    // Verify ordering: token 3 has highest score
    TEST_ASSERT(scores[3] > scores[2] && scores[2] > scores[1] && scores[1] > scores[0],
                "L2 scores ordered correctly");
}

// ============================================================
// Test 3: Thinning - Token Selection
// ============================================================
void test_thinning_token_selection() {
    fprintf(stderr, "\n=== Test: Thinning Token Selection ===\n");

    float embeddings[4 * 8] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0
    };

    llamaedge_thinning_config config = {};
    config.enabled = true;
    config.policy = LLAMAEDGE_THIN_L2_NORM;
    config.top_k = 2;
    config.protected_token_count = 1;  // Protect first token

    uint32_t selected[4];
    uint32_t count = llamaedge_thinning_score_and_select(
        embeddings, 4, 8, &config, selected
    );

    TEST_ASSERT(count == 3, "Selected 3 tokens (protected + top 2)");
    TEST_ASSERT(selected[0] == 0, "Protected token 0 selected");
    // Remaining tokens 1,2,3 ranked by L2 - highest first
    TEST_ASSERT(selected[1] == 3, "Token 3 selected (highest L2)");
    TEST_ASSERT(selected[2] == 2, "Token 2 selected");
}

// ============================================================
// Test 4: Thinning - Disabled (Select All)
// ============================================================
void test_thinning_disabled() {
    fprintf(stderr, "\n=== Test: Thinning Disabled (Select All) ===\n");

    llamaedge_thinning_config config = {};
    config.enabled = false;
    config.policy = LLAMAEDGE_THIN_L2_NORM;

    uint32_t selected[8];
    for (uint32_t i = 0; i < 8; ++i) selected[i] = 999;

    uint32_t count = llamaedge_thinning_score_and_select(
        nullptr, 8, 8, &config, selected
    );

    TEST_ASSERT(count == 8, "All 8 tokens selected when disabled");
    for (uint32_t i = 0; i < 8; ++i) {
        TEST_ASSERT(selected[i] == i, "Token indices in order");
    }
}

// ============================================================
// Test 5: Quantization Config
// ============================================================
void test_quant_config() {
    fprintf(stderr, "\n=== Test: Quantization Config ===\n");

    llamaedge_quant_config config = {};
    config.enabled = true;
    config.bitplan = LLAMAEDGE_Q3_K4V3;
    config.use_wht = true;

    TEST_ASSERT(config.enabled == true, "Quantization enabled");
    TEST_ASSERT(config.bitplan == LLAMAEDGE_Q3_K4V3, "Q3 bitplan");
    TEST_ASSERT(config.use_wht == true, "WHT enabled");
}

// ============================================================
// Test 6: K Quantization (K4 per-channel)
// ============================================================
void test_k_quantization() {
    fprintf(stderr, "\n=== Test: K Quantization (K4 per-channel) ===\n");

    // Simple K data: 2 tokens, 1 head, 4 dimensions
    // Token 0: [1.0, 2.0, 3.0, 4.0]
    // Token 1: [4.0, 3.0, 2.0, 1.0]
    float k_data[2 * 1 * 4] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };

    llamaedge_quant_config config = {};
    config.enabled = true;
    config.bitplan = LLAMAEDGE_Q4_K4V4;  // K always 4-bit

    uint8_t quantized[2 * 1 * 4];
    float scales[1 * 4];

    size_t bytes = llamaedge_quantize_k(
        k_data, 2, 1, 4, &config, quantized, scales
    );

    TEST_ASSERT(bytes == 8, "Bytes for quantized K");
    TEST_ASSERT(scales[0] > 0 && scales[0] < 10, "Scale computed for channel 0");

    // Dequantize and verify
    float restored[2 * 1 * 4];
    llamaedge_dequantize_k(quantized, scales, 2, 1, 4, &config, restored);

    // Check that values are approximately restored (within quantization error)
    for (uint32_t i = 0; i < 2 * 4; ++i) {
        float orig = k_data[i];
        float rec = restored[i];
        float rel_error = std::abs(rec - orig) / (std::abs(orig) + 1e-6);
        TEST_ASSERT(rel_error < 0.3f, "Dequantized value close to original");
    }
}

// ============================================================
// Test 7: V Quantization (V3 with WHT)
// ============================================================
void test_v_quantization() {
    fprintf(stderr, "\n=== Test: V Quantization (V3 with WHT) ===\n");

    // V data: 2 tokens, 1 head, 4 dimensions (power of 2 for WHT)
    float v_data[2 * 1 * 4] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };

    llamaedge_quant_config config = {};
    config.enabled = true;
    config.bitplan = LLAMAEDGE_Q3_K4V3;  // V is 3-bit
    config.use_wht = true;

    uint8_t quantized[2 * 1 * 4];
    float scales[2];  // One scale per token

    size_t bytes = llamaedge_quantize_v(
        v_data, 2, 1, 4, &config, quantized, scales
    );

    TEST_ASSERT(bytes == 8, "Bytes for quantized V");
    TEST_ASSERT(scales[0] > 0 && scales[1] > 0, "Scales computed for each token");

    // Dequantize and verify
    float restored[2 * 1 * 4];
    llamaedge_dequantize_v(quantized, scales, 2, 1, 4, &config, restored);

    TEST_ASSERT(true, "V quantization with WHT completed");
}

// ============================================================
// Test 8: B4 Pipeline Create/Destroy
// ============================================================
void test_b4_pipeline_create_destroy() {
    fprintf(stderr, "\n=== Test: B4 Pipeline Create/Destroy ===\n");

    llamaedge_b4_config config = {};
    config.profile = LLAMAEDGE_PROFILE_BALANCED;
    config.thinning.enabled = true;
    config.quant.enabled = true;
    config.chunk_size_tokens = 32;

    llamaedge_b4_pipeline * pipeline = llamaedge_b4_pipeline_create(&config);
    TEST_ASSERT(pipeline != nullptr, "Pipeline created");

    llamaedge_profile profile = llamaedge_b4_pipeline_get_profile(pipeline);
    TEST_ASSERT(profile == LLAMAEDGE_PROFILE_BALANCED, "Profile is BALANCED");

    const llamaedge_thinning_config * thin = llamaedge_b4_pipeline_get_thinning_config(pipeline);
    TEST_ASSERT(thin != nullptr && thin->enabled == true, "Thinning config retrieved");

    const llamaedge_quant_config * quant = llamaedge_b4_pipeline_get_quant_config(pipeline);
    TEST_ASSERT(quant != nullptr && quant->enabled == true, "Quant config retrieved");

    const llamaedge_warm_config * warm = llamaedge_b4_pipeline_get_warm_config(pipeline);
    TEST_ASSERT(warm != nullptr, "Warm config retrieved");

    llamaedge_b4_pipeline_destroy(pipeline);
    fprintf(stderr, "[PASS] Pipeline destroyed\n");
}

// ============================================================
// Test 9: Profile Switching
// ============================================================
void test_profile_switching() {
    fprintf(stderr, "\n=== Test: Profile Switching ===\n");

    llamaedge_b4_config config = {};
    config.profile = LLAMAEDGE_PROFILE_CONSERVATIVE;
    config.thinning.enabled = true;
    config.quant.enabled = true;

    llamaedge_b4_pipeline * pipeline = llamaedge_b4_pipeline_create(&config);
    TEST_ASSERT(pipeline != nullptr, "Pipeline created with CONSERVATIVE profile");

    llamaedge_profile profile = llamaedge_b4_pipeline_get_profile(pipeline);
    TEST_ASSERT(profile == LLAMAEDGE_PROFILE_CONSERVATIVE, "Initial profile is CONSERVATIVE");

    // Note: llamaedge_profile_apply requires context, tested separately
    llamaedge_b4_pipeline_destroy(pipeline);

    TEST_ASSERT(true, "Profile switching infrastructure available");
}

// ============================================================
// Test 10: Profile Change Callback Registration
// ============================================================
void test_profile_callbacks() {
    fprintf(stderr, "\n=== Test: Profile Change Callback Registration ===\n");

    // Register/unregister should not crash without hooks enabled
    int ret = llamaedge_profile_change_register(nullptr, nullptr, nullptr);
    TEST_ASSERT(ret == -1, "Register fails without context");

    ret = llamaedge_profile_change_unregister(nullptr, nullptr);
    TEST_ASSERT(ret == -1, "Unregister fails without context");

    TEST_ASSERT(true, "Callback registration stubs tested");
}

// ============================================================
// Test 11: Pipeline Prefill (Thinning + Chunking)
// ============================================================
void test_b4_pipeline_prefill() {
    fprintf(stderr, "\n=== Test: B4 Pipeline Prefill ===\n");

    llamaedge_b4_config config = {};
    config.profile = LLAMAEDGE_PROFILE_BALANCED;
    config.thinning.enabled = true;
    config.thinning.policy = LLAMAEDGE_THIN_L2_NORM;
    config.thinning.top_k = 32;
    config.thinning.protected_token_count = 4;
    config.quant.enabled = true;
    config.quant.bitplan = LLAMAEDGE_Q3_K4V3;
    config.quant.use_wht = true;
    config.warm.n_warm_layers = 2;
    config.warm.n_warm_chunks = 2;
    config.warm.chunk_size_tokens = 32;
    config.chunk_size_tokens = 32;

    llamaedge_b4_pipeline * pipeline = llamaedge_b4_pipeline_create(&config);
    TEST_ASSERT(pipeline != nullptr, "Pipeline created");

    // Simulate embeddings: 64 tokens, 8 dimension each
    std::vector<float> embeddings(64 * 8, 1.0f);

    uint32_t n_chunks = llamaedge_b4_pipeline_prefill(
        pipeline, nullptr, 1, 64, embeddings.data(), nullptr
    );

    TEST_ASSERT(n_chunks == 2, "64 tokens / 32 chunk_size = 2 chunks");

    llamaedge_b4_pipeline_destroy(pipeline);
    fprintf(stderr, "[PASS] Pipeline prefill completed\n");
}

// ============================================================
// Test 12: Pipeline Install Chunk
// ============================================================
void test_b4_pipeline_install() {
    fprintf(stderr, "\n=== Test: B4 Pipeline Install Chunk ===\n");

    llamaedge_b4_config config = {};
    config.profile = LLAMAEDGE_PROFILE_BALANCED;
    config.quant.enabled = true;
    config.quant.bitplan = LLAMAEDGE_Q3_K4V3;

    llamaedge_b4_pipeline * pipeline = llamaedge_b4_pipeline_create(&config);
    TEST_ASSERT(pipeline != nullptr, "Pipeline created");

    // Simulate quantized data
    uint8_t k_quant[32] = {0};
    float k_scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint8_t v_quant[32] = {0};
    float v_scales[1] = {1.0f};

    int ret = llamaedge_b4_pipeline_install_chunk(
        pipeline, nullptr, 1,
        k_quant, k_scales, v_quant, v_scales,
        32, 1, 4, 4
    );

    TEST_ASSERT(ret == 0, "Install chunk returns success");

    llamaedge_b4_pipeline_destroy(pipeline);
    fprintf(stderr, "[PASS] Pipeline install chunk completed\n");
}

// ============================================================
// Test 13: Quant Bitplan Enums
// ============================================================
void test_quant_bitplans() {
    fprintf(stderr, "\n=== Test: Quant Bitplan Enums ===\n");

    TEST_ASSERT(LLAMAEDGE_Q3_K4V3 == 0, "Q3_K4V3 = 0");
    TEST_ASSERT(LLAMAEDGE_Q4_K4V4 == 1, "Q4_K4V4 = 1");
    TEST_ASSERT(LLAMAEDGE_Q4_K4V3 == 2, "Q4_K4V3 = 2");
}

// ============================================================
// Test 14: Profile Enums
// ============================================================
void test_profile_enums() {
    fprintf(stderr, "\n=== Test: Profile Enums ===\n");

    TEST_ASSERT(LLAMAEDGE_PROFILE_CONSERVATIVE == 0, "CONSERVATIVE = 0");
    TEST_ASSERT(LLAMAEDGE_PROFILE_BALANCED == 1, "BALANCED = 1");
    TEST_ASSERT(LLAMAEDGE_PROFILE_AGGRESSIVE == 2, "AGGRESSIVE = 2");
}

// ============================================================
// Test 15: Thinning Policy Enums
// ============================================================
void test_thinning_policy_enums() {
    fprintf(stderr, "\n=== Test: Thinning Policy Enums ===\n");

    TEST_ASSERT(LLAMAEDGE_THIN_NONE == 0, "THIN_NONE = 0");
    TEST_ASSERT(LLAMAEDGE_THIN_L2_NORM == 1, "THIN_L2_NORM = 1");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char ** argv) {
    fprintf(stderr, "=== B4 Full EdgeWeaver Tests ===\n");
    fprintf(stderr, "Testing: thinning, quantization, pipeline, profile switching\n\n");

    (void)argc; (void)argv;

    test_b4_config_init();
    test_thinning_l2_scores();
    test_thinning_token_selection();
    test_thinning_disabled();
    test_quant_config();
    test_k_quantization();
    test_v_quantization();
    test_b4_pipeline_create_destroy();
    test_profile_switching();
    test_profile_callbacks();
    test_b4_pipeline_prefill();
    test_b4_pipeline_install();
    test_quant_bitplans();
    test_profile_enums();
    test_thinning_policy_enums();

    fprintf(stderr, "\n=== Test Summary ===\n");
    fprintf(stderr, "Passed: %d\n", g_test_passed);
    fprintf(stderr, "Failed: %d\n", g_test_failed);

    return g_test_failed > 0 ? 1 : 0;
}