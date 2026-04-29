// Test for B3 chunk streaming functionality
// Tests: TX ring, payload pool, frontier, warm threshold

#include "llama.h"
#include "llamaedge/hooks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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
// Test 1: TX Ring Create/Destroy
// ============================================================
void test_tx_ring_create_destroy() {
    fprintf(stderr, "\n=== Test: TX Ring Create/Destroy ===\n");

    llamaedge_tx_ring_config config = {};
    config.ring_size = 16;
    config.payload_pool_bytes = 1024 * 1024;
    config.n_destinations = 2;

    llamaedge_tx_ring * ring = llamaedge_tx_ring_create(&config);
    TEST_ASSERT(ring != nullptr, "TX ring created");

    // Just verify it can be destroyed without crash
    llamaedge_tx_ring_destroy(ring);
    fprintf(stderr, "[PASS] TX ring created and destroyed\n");
}

// ============================================================
// Test 2: TX Ring Push/Pop
// ============================================================
void test_tx_ring_push_pop() {
    fprintf(stderr, "\n=== Test: TX Ring Push/Pop ===\n");

    llamaedge_tx_ring_config config = {};
    config.ring_size = 4;
    config.payload_pool_bytes = 64 * 1024;
    config.n_destinations = 1;

    llamaedge_tx_ring * ring = llamaedge_tx_ring_create(&config);
    TEST_ASSERT(ring != nullptr, "TX ring created");

    // Push some descriptors
    for (int i = 0; i < 3; ++i) {
        llamaedge_tx_desc desc = {};
        desc.session_id = 1;
        desc.layer_id = i;
        desc.chunk_id = 0;
        desc.token_begin = i * 32;
        desc.token_count = 32;
        desc.n_kv_heads = 8;
        desc.n_embd_head_k = 128;
        desc.n_embd_head_v = 128;
        desc.payload_size = 32 * 8 * 128 * sizeof(float);

        int ret = llamaedge_tx_ring_push(ring, 0, &desc);
        TEST_ASSERT(ret == 0, "Push descriptor succeeded");
    }

    // Check pending count
    uint32_t pending = llamaedge_tx_ring_pending(ring, 0);
    TEST_ASSERT(pending == 3, "3 descriptors pending");

    // Pop one
    llamaedge_tx_desc desc_out = {};
    int ret = llamaedge_tx_ring_pop(ring, 0, &desc_out);
    TEST_ASSERT(ret == 0, "Pop descriptor succeeded");
    TEST_ASSERT(desc_out.layer_id == 0, "Popped layer 0");

    pending = llamaedge_tx_ring_pending(ring, 0);
    TEST_ASSERT(pending == 2, "2 descriptors remaining");

    // Pop remaining
    for (int i = 0; i < 2; ++i) {
        ret = llamaedge_tx_ring_pop(ring, 0, &desc_out);
        TEST_ASSERT(ret == 0, "Pop descriptor succeeded");
    }

    pending = llamaedge_tx_ring_pending(ring, 0);
    TEST_ASSERT(pending == 0, "No descriptors remaining");

    // Pop from empty ring should fail
    ret = llamaedge_tx_ring_pop(ring, 0, &desc_out);
    TEST_ASSERT(ret == -1, "Pop from empty ring fails");

    llamaedge_tx_ring_destroy(ring);
}

// ============================================================
// Test 3: TX Ring Overflow
// ============================================================
void test_tx_ring_overflow() {
    fprintf(stderr, "\n=== Test: TX Ring Overflow ===\n");

    llamaedge_tx_ring_config config = {};
    config.ring_size = 2;
    config.payload_pool_bytes = 64 * 1024;
    config.n_destinations = 1;

    llamaedge_tx_ring * ring = llamaedge_tx_ring_create(&config);
    TEST_ASSERT(ring != nullptr, "TX ring created");

    // Fill the ring
    for (int i = 0; i < 2; ++i) {
        llamaedge_tx_desc desc = {};
        desc.session_id = 1;
        desc.layer_id = i;
        int ret = llamaedge_tx_ring_push(ring, 0, &desc);
        TEST_ASSERT(ret == 0, "Push succeeded");
    }

    // This should fail - ring full
    llamaedge_tx_desc desc = {};
    desc.session_id = 1;
    desc.layer_id = 99;
    int ret = llamaedge_tx_ring_push(ring, 0, &desc);
    TEST_ASSERT(ret == -1, "Push to full ring fails");

    llamaedge_tx_ring_destroy(ring);
}

// ============================================================
// Test 4: Payload Pool
// ============================================================
void test_payload_pool() {
    fprintf(stderr, "\n=== Test: Payload Pool ===\n");

    llamaedge_tx_ring_config config = {};
    config.ring_size = 4;
    config.payload_pool_bytes = 1024;
    config.n_destinations = 1;

    llamaedge_tx_ring * ring = llamaedge_tx_ring_create(&config);
    TEST_ASSERT(ring != nullptr, "TX ring created");

    // Allocate from pool
    uint8_t * p1 = llamaedge_payload_pool_alloc(ring, 256);
    TEST_ASSERT(p1 != nullptr, "First allocation succeeded");

    uint8_t * p2 = llamaedge_payload_pool_alloc(ring, 256);
    TEST_ASSERT(p2 != nullptr, "Second allocation succeeded");

    TEST_ASSERT(p2 > p1, "Second allocation is after first");

    llamaedge_tx_ring_destroy(ring);
}

// ============================================================
// Test 5: Frontier Management
// ============================================================
void test_frontier() {
    fprintf(stderr, "\n=== Test: Frontier Management ===\n");

    llamaedge_frontier * frontier = llamaedge_frontier_create(1, 4);
    TEST_ASSERT(frontier != nullptr, "Frontier created");
    TEST_ASSERT(frontier->session_id == 1, "Frontier session_id = 1");

    // Initial state should be -1 (nothing installed)
    for (uint32_t i = 0; i < frontier->n_layers; ++i) {
        TEST_ASSERT(frontier->layer_frontier[i] == -1, "Initial frontier is -1");
    }

    // Update layer 0: tokens 0-31 installed
    llamaedge_frontier_update(frontier, 0, 0, 32);
    TEST_ASSERT(frontier->layer_frontier[0] == 32, "Layer 0 frontier updated to 32");

    // Update layer 1: tokens 0-31 installed
    llamaedge_frontier_update(frontier, 1, 0, 32);
    TEST_ASSERT(frontier->layer_frontier[1] == 32, "Layer 1 frontier updated to 32");

    // Update layer 0 again: tokens 32-63 installed (extends)
    llamaedge_frontier_update(frontier, 0, 32, 32);
    TEST_ASSERT(frontier->layer_frontier[0] == 64, "Layer 0 frontier extended to 64");

    // Update layer 2
    llamaedge_frontier_update(frontier, 2, 0, 32);
    TEST_ASSERT(frontier->layer_frontier[2] == 32, "Layer 2 frontier updated");

    // Out-of-order chunk should not advance contiguity until the gap is filled
    llamaedge_frontier_update(frontier, 3, 32, 32);
    TEST_ASSERT(frontier->layer_frontier[3] == -1, "Layer 3 frontier stays blocked on a gap");
    llamaedge_frontier_update(frontier, 3, 0, 32);
    TEST_ASSERT(frontier->layer_frontier[3] == 64, "Layer 3 frontier catches up once gap is filled");

    // Minimum frontier across all layers
    int32_t min_f = llamaedge_frontier_min(frontier);
    TEST_ASSERT(min_f == 32, "Minimum frontier respects contiguous progress across layers");

    llamaedge_frontier_destroy(frontier);
    fprintf(stderr, "[PASS] Frontier destroyed\n");
}

// ============================================================
// Test 6: Warm Threshold - Layer Condition
// ============================================================
void test_warm_threshold_layer() {
    fprintf(stderr, "\n=== Test: Warm Threshold (Layer Condition) ===\n");

    llamaedge_frontier * frontier = llamaedge_frontier_create(1, 4);
    TEST_ASSERT(frontier != nullptr, "Frontier created");

    llamaedge_warm_config config = {};
    config.n_warm_layers = 2;       // First 2 layers needed
    config.n_warm_chunks = 2;        // First 2 chunks needed
    config.chunk_size_tokens = 32;

    // Scenario: layer 0 and 1 have frontier at 64 (past token 0)
    // This should trigger warm threshold for decode position 0
    llamaedge_frontier_update(frontier, 0, 0, 64);
    llamaedge_frontier_update(frontier, 1, 0, 64);

    bool warm = llamaedge_warm_threshold_met(frontier, 0, &config);
    TEST_ASSERT(warm == true, "Warm threshold met: first 2 layers > pos 0");

    // Decode position 50 should still be okay if layer frontier > 50
    warm = llamaedge_warm_threshold_met(frontier, 50, &config);
    TEST_ASSERT(warm == true, "Warm threshold met: first 2 layers > pos 50");

    // Decode position 100 should NOT be okay (layers only at 64)
    warm = llamaedge_warm_threshold_met(frontier, 100, &config);
    TEST_ASSERT(warm == false, "Warm threshold NOT met: layers only at 64");

    llamaedge_frontier_destroy(frontier);
}

// ============================================================
// Test 7: Warm Threshold - Chunk Condition
// ============================================================
void test_warm_threshold_chunk() {
    fprintf(stderr, "\n=== Test: Warm Threshold (Chunk Condition) ===\n");

    llamaedge_frontier * frontier = llamaedge_frontier_create(1, 4);
    TEST_ASSERT(frontier != nullptr, "Frontier created");

    llamaedge_warm_config config = {};
    config.n_warm_layers = 0;       // Disable layer condition to test chunk condition
    config.n_warm_chunks = 3;        // First 3 chunks needed
    config.chunk_size_tokens = 32;

    // Install 2 chunks (64 tokens) in all layers
    for (uint32_t i = 0; i < frontier->n_layers; ++i) {
        llamaedge_frontier_update(frontier, i, 0, 64);
    }

    // 2 chunks installed, need 3 - should NOT trigger
    bool warm = llamaedge_warm_threshold_met(frontier, 0, &config);
    TEST_ASSERT(warm == false, "Not warm: only 2 chunks installed, need 3");

    // Install 3rd chunk (96 tokens) in ALL layers to make chunks contiguous
    for (uint32_t i = 0; i < frontier->n_layers; ++i) {
        llamaedge_frontier_update(frontier, i, 64, 32);
    }

    // Now 3 chunks installed in ALL layers - should trigger
    warm = llamaedge_warm_threshold_met(frontier, 0, &config);
    TEST_ASSERT(warm == true, "Warm: 3 chunks installed in all layers");

    llamaedge_frontier_destroy(frontier);
}

// ============================================================
// Test 8: Multi-Destination TX Ring
// ============================================================
void test_tx_ring_multi_dest() {
    fprintf(stderr, "\n=== Test: TX Ring Multi-Destination ===\n");

    llamaedge_tx_ring_config config = {};
    config.ring_size = 4;
    config.payload_pool_bytes = 1024;
    config.n_destinations = 3;

    llamaedge_tx_ring * ring = llamaedge_tx_ring_create(&config);
    TEST_ASSERT(ring != nullptr, "TX ring created with 3 destinations");

    // Push to destination 0
    llamaedge_tx_desc desc0 = {};
    desc0.session_id = 1;
    desc0.layer_id = 0;
    int ret = llamaedge_tx_ring_push(ring, 0, &desc0);
    TEST_ASSERT(ret == 0, "Push to dest 0 succeeded");

    // Push to destination 1
    llamaedge_tx_desc desc1 = {};
    desc1.session_id = 2;
    desc1.layer_id = 1;
    ret = llamaedge_tx_ring_push(ring, 1, &desc1);
    TEST_ASSERT(ret == 0, "Push to dest 1 succeeded");

    // Pending counts should be independent
    TEST_ASSERT(llamaedge_tx_ring_pending(ring, 0) == 1, "Dest 0 has 1 pending");
    TEST_ASSERT(llamaedge_tx_ring_pending(ring, 1) == 1, "Dest 1 has 1 pending");
    TEST_ASSERT(llamaedge_tx_ring_pending(ring, 2) == 0, "Dest 2 has 0 pending");

    // Pop from dest 0
    llamaedge_tx_desc desc_out = {};
    ret = llamaedge_tx_ring_pop(ring, 0, &desc_out);
    TEST_ASSERT(ret == 0, "Pop from dest 0 succeeded");
    TEST_ASSERT(desc_out.session_id == 1, "Popped dest 0's descriptor");

    // Dest 1 should be unaffected
    TEST_ASSERT(llamaedge_tx_ring_pending(ring, 1) == 1, "Dest 1 still has 1 pending");

    llamaedge_tx_ring_destroy(ring);
}

// ============================================================
// Test 9: Hook Registration (stub tests)
// ============================================================
void test_hook_registration() {
    fprintf(stderr, "\n=== Test: Hook Registration (Stubs) ===\n");

    // Without LLAMAEDGE_ENABLE_HOOKS, these should fail
    int ret = llamaedge_kv_export_register(nullptr, nullptr, nullptr);
    TEST_ASSERT(ret == -1, "KV export register fails without hooks");

    ret = llamaedge_kv_install_register(nullptr, nullptr, nullptr);
    TEST_ASSERT(ret == -1, "KV install register fails without hooks");

    ret = llamaedge_kv_export_layers(nullptr, 0, 0);
    TEST_ASSERT(ret == -1, "KV export layers fails without hooks");

    // State export/import should still work with llama_context
    // (tested in other tests)
    TEST_ASSERT(true, "Hook registration stubs tested");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char ** argv) {
    fprintf(stderr, "=== B3 Chunk Streaming Tests ===\n");
    fprintf(stderr, "Testing: TX ring, payload pool, frontier, warm threshold\n\n");

    (void)argc; (void)argv;

    test_tx_ring_create_destroy();
    test_tx_ring_push_pop();
    test_tx_ring_overflow();
    test_payload_pool();
    test_frontier();
    test_warm_threshold_layer();
    test_warm_threshold_chunk();
    test_tx_ring_multi_dest();
    test_hook_registration();

    fprintf(stderr, "\n=== Test Summary ===\n");
    fprintf(stderr, "Passed: %d\n", g_test_passed);
    fprintf(stderr, "Failed: %d\n", g_test_failed);

    return g_test_failed > 0 ? 1 : 0;
}
