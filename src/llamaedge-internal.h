#ifndef LLAMAEDGE_INTERNAL_H
#define LLAMAEDGE_INTERNAL_H

#include "llama.h"
#include "llamaedge/hooks.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Internal Structures (moved from hooks.h / hooks.cpp)
// ============================================================

// Per-layer KV export hook entry
struct llamaedge_kv_export_entry {
    llamaedge_kv_export_fn  fn;
    void *                  user_data;
    llamaedge_kv_export_entry * next;
};

// Per-layer KV install hook entry
struct llamaedge_kv_install_entry {
    llamaedge_kv_install_fn fn;
    void *                  user_data;
    llamaedge_kv_install_entry * next;
};

// Synchronization primitives for TX ring destination
struct llamaedge_tx_ring_sync {
    std::mutex mutex;
    std::condition_variable cv;
};

// TX descriptor ring for one destination
struct llamaedge_tx_ring_dest {
    std::vector<llamaedge_tx_desc> descriptors;
    std::vector<uint32_t> in_use;
    uint32_t head = 0;
    uint32_t tail = 0;
    uint32_t count = 0;
    llamaedge_tx_ring_sync * sync;
};

// Payload-pool allocation block. The pool is a real ring-local allocator,
// not a process-global demo bump pointer. Blocks are split on allocation,
// merged on free, and protected by pool_mutex.
struct llamaedge_payload_block {
    size_t offset = 0;
    size_t size = 0;
    bool free = true;
};

// TX ring state
struct llamaedge_tx_ring {
    uint32_t ring_size;
    uint32_t payload_pool_bytes;
    uint32_t n_destinations;
    std::vector<llamaedge_tx_ring_dest> destinations;
    std::vector<uint8_t> payload_pool;
    std::vector<llamaedge_payload_block> payload_blocks;
    std::mutex pool_mutex;
};

// Hook registry stored in context extension
struct llamaedge_hook_registry {
    llamaedge_kv_export_entry  * kv_export_hooks = nullptr;
    llamaedge_kv_install_entry * kv_install_hooks = nullptr;
    llamaedge_hook             * general_hooks = nullptr;
    uint32_t                    active_hooks = 0;
};

// Profile state
struct llamaedge_profile_change_entry {
    llamaedge_profile_change_fn fn;
    void * user_data;
    llamaedge_profile_change_entry * next;
};

struct llamaedge_profile_state {
    llamaedge_profile current_profile = LLAMAEDGE_PROFILE_BALANCED;
    llamaedge_profile_change_entry * callbacks = nullptr;
};

// B4 pipeline state
struct llamaedge_b4_pipeline {
    llamaedge_b4_config config;
    llamaedge_profile current_profile;
    std::vector<uint8_t> k_quant_buf;
    std::vector<float> k_scales_buf;
    std::vector<uint8_t> v_quant_buf;
    std::vector<float> v_scales_buf;
};

// Pending span types for frontier
using llamaedge_pending_span = std::pair<int32_t, int32_t>;
using llamaedge_pending_span_table = std::vector<std::vector<llamaedge_pending_span>>;

// ============================================================
// Internal helper declarations
// ============================================================

// Hook registry
llamaedge_hook_registry * llamaedge_get_registry(struct llama_context * ctx);
void llamaedge_recompute_active_hooks(llamaedge_hook_registry * reg);
void llamaedge_clear_registry(llamaedge_hook_registry * reg);

// KV cache metadata access
void llamaedge_get_model_hparams(struct llama_context * ctx,
    uint32_t * n_layers_out, uint32_t * n_kv_heads_out,
    uint32_t * n_embd_head_k_out, uint32_t * n_embd_head_v_out);

// Tensor extraction
size_t llamaedge_extract_kv_tensor_f32(const struct ggml_tensor * src,
    float * dst, uint32_t n_tokens, uint32_t n_heads, uint32_t n_embd);
size_t llamaedge_deserialize_kv_tensor(const uint8_t * src,
    int32_t type, uint64_t size_row, uint32_t n_tokens, uint32_t n_heads, uint32_t n_embd,
    float * dst);

// Pending span management
llamaedge_pending_span_table * llamaedge_frontier_pending(llamaedge_frontier * frontier);
void llamaedge_frontier_add_pending(llamaedge_frontier * frontier,
    int32_t layer_id, int32_t token_begin, int32_t token_end);
void llamaedge_frontier_merge_pending(llamaedge_frontier * frontier, int32_t layer_id);

// Wire format helpers
const uint8_t * llamaedge_read_u32(const uint8_t * src, size_t size, size_t * offset, uint32_t * out);
const uint8_t * llamaedge_read_u64(const uint8_t * src, size_t size, size_t * offset, uint64_t * out);
const uint8_t * llamaedge_read_i32(const uint8_t * src, size_t size, size_t * offset, int32_t * out);

// Quantization helpers
int32_t llamaedge_qrange(int32_t bits);
void llamaedge_walsh_hadamard_transform_inplace(float * data, uint32_t n);

// Thinning helpers
uint32_t llamaedge_select_token_indices_by_l2(
    const float * scores, uint32_t n_tokens,
    uint32_t top_k, float top_ratio,
    uint32_t protected_token_count,
    uint32_t * selected_indices);

// Profile helpers
llamaedge_profile_state * llamaedge_get_profile_state(const struct llama_context * ctx);
void llamaedge_profile_to_config(llamaedge_profile profile,
    llamaedge_thinning_config * thinning_out, llamaedge_quant_config * quant_out);

// Chunk header serialization
size_t llamaedge_chunk_header_json_size(const llamaedge_chunk_header * hdr);
size_t llamaedge_chunk_header_to_json(const llamaedge_chunk_header * hdr, char * json_out, size_t json_size);
bool llamaedge_chunk_header_from_json(const char * json_str, llamaedge_chunk_header * hdr_out);

#ifdef __cplusplus
}
#endif

#endif // LLAMAEDGE_INTERNAL_H
