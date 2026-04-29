#ifndef LLAMAEDGE_HOOKS_H
#define LLAMAEDGE_HOOKS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Forward declaration
struct llama_context;
struct llama_batch;

// ============================================================
// Phase 2 B3 Chunk Streaming - Hook Infrastructure
// ============================================================
//
// B3 adds on B2:
//   - Per-layer KV export callback
//   - Per-layer KV install callback
//   - Chunk-based streaming with TX descriptor ring
//   - Warm threshold for early decode start
//   - Frontier correctness model
//
// Usage:
//   1. Register KV export hook to get called after each layer
//   2. Register KV install hook to get called when installing chunks
//   3. Use chunk API for streaming data transfer
//   4. Check warm threshold before starting decode

#define LLAMAEDGE_HOOK_PRE_PREFILL    (1 << 0)
#define LLAMAEDGE_HOOK_POST_PREFILL   (1 << 1)
#define LLAMAEDGE_HOOK_KV_EXPORT      (1 << 2)
#define LLAMAEDGE_HOOK_KV_INSTALL     (1 << 3)

// Per-layer KV export hook - called after each layer completes
// Returns 0 on success, negative on error
typedef int (*llamaedge_kv_export_fn)(
    struct llama_context * ctx,
    void * user_data,
    int32_t layer_id,           // layer index being exported
    int32_t seq_id,             // sequence ID
    const float * k_data,       // key tensor data (can be nullptr if empty)
    const float * v_data,       // value tensor data (can be nullptr if empty)
    uint32_t n_tokens,          // number of tokens in this layer's KV
    uint32_t n_kv_heads,        // number of KV heads
    uint32_t n_embd_head_k,     // key embedding dimension
    uint32_t n_embd_head_v      // value embedding dimension
);

// Per-layer KV install hook - called when installing chunks
// Returns 0 on success, negative on error
typedef int (*llamaedge_kv_install_fn)(
    struct llama_context * ctx,
    void * user_data,
    int32_t layer_id,
    int32_t seq_id,
    const float * k_data,
    const float * v_data,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_k,
    uint32_t n_embd_head_v
);

// Hook function signature (legacy)
typedef int (*llamaedge_hook_fn)(struct llama_context * ctx, void * user_data);

// Hook registration structure
typedef struct llamaedge_hook {
    uint32_t               hook_type;    // combination of LLAMAEDGE_HOOK_* flags
    llamaedge_hook_fn      fn;            // hook function
    void *                 user_data;     // user-provided context
    struct llamaedge_hook * next;          // linked list for multiple hooks
} llamaedge_hook;

// ============================================================
// B3: Chunk Streaming Structures
// ============================================================

// Chunk configuration
typedef struct llamaedge_chunk_config {
    uint32_t chunk_size_tokens;   // number of tokens per chunk
    uint32_t n_layers;            // total number of layers
    uint32_t n_kv_heads;          // number of KV heads
    uint32_t n_embd_head_k;       // key head dimension
    uint32_t n_embd_head_v;       // value head dimension
} llamaedge_chunk_config;

// TX descriptor - describes one chunk to be sent
typedef struct llamaedge_tx_desc {
    int32_t     session_id;       // session identifier
    int32_t     layer_id;         // layer index
    int32_t     chunk_id;         // chunk index within layer
    uint32_t    token_begin;      // first token position in this chunk
    uint32_t    token_count;      // number of tokens in chunk
    uint32_t    n_kv_heads;       // KV head count
    uint32_t    n_embd_head_k;    // key head dim
    uint32_t    n_embd_head_v;    // value head dim
    size_t      payload_size;     // total bytes for this chunk's KV
    const uint8_t * payload;      // pointer to payload in shared pool (set by sender)
    bool        in_use;           // descriptor slot in use
} llamaedge_tx_desc;

// Frontier tracking structure
// Tracks installed contiguous frontier per session per layer
typedef struct llamaedge_frontier {
    int32_t * layer_frontier;      // [n_layers] - max contiguous token installed per layer
    int32_t   session_id;          // session this frontier belongs to
    uint32_t  n_layers;            // number of layers
    void *    pending_spans;       // internal C++ state for out-of-order chunk tracking
} llamaedge_frontier;

// ============================================================
// B3: TX Descriptor Ring (per destination)
// ============================================================

// TX ring configuration
typedef struct llamaedge_tx_ring_config {
    uint32_t ring_size;           // number of descriptor slots
    uint32_t payload_pool_bytes;   // shared payload pool size
    uint32_t n_destinations;      // number of destinations (workers)
} llamaedge_tx_ring_config;

// TX ring handle
typedef struct llamaedge_tx_ring llamaedge_tx_ring;

// Create/destroy TX ring
LLAMA_API llamaedge_tx_ring * llamaedge_tx_ring_create(
    const llamaedge_tx_ring_config * config
);
LLAMA_API void llamaedge_tx_ring_destroy(llamaedge_tx_ring * ring);

// Push a chunk descriptor to TX ring for a destination
// Returns 0 on success, -1 if ring full
LLAMA_API int llamaedge_tx_ring_push(
    llamaedge_tx_ring * ring,
    uint32_t destination,         // destination index (0-based)
    const llamaedge_tx_desc * desc
);

// Pop completed descriptor from TX ring
// Returns 0 on success, -1 if ring empty
LLAMA_API int llamaedge_tx_ring_pop(
    llamaedge_tx_ring * ring,
    uint32_t destination,
    llamaedge_tx_desc * desc
);

// Get ring status
LLAMA_API uint32_t llamaedge_tx_ring_pending(
    const llamaedge_tx_ring * ring,
    uint32_t destination
);

// Payload pool management
LLAMA_API uint8_t * llamaedge_payload_pool_alloc(
    llamaedge_tx_ring * ring,
    size_t size
);
LLAMA_API void llamaedge_payload_pool_free(
    llamaedge_tx_ring * ring,
    uint8_t * payload
);

// ============================================================
// B3: Per-Layer KV Export/Install API
// ============================================================

// Register per-layer KV export hook
LLAMA_API int llamaedge_kv_export_register(
    struct llama_context * ctx,
    llamaedge_kv_export_fn fn,
    void * user_data
);

// Unregister per-layer KV export hook
LLAMA_API int llamaedge_kv_export_unregister(
    struct llama_context * ctx,
    llamaedge_kv_export_fn fn
);

// Register per-layer KV install hook
LLAMA_API int llamaedge_kv_install_register(
    struct llama_context * ctx,
    llamaedge_kv_install_fn fn,
    void * user_data
);

// Unregister per-layer KV install hook
LLAMA_API int llamaedge_kv_install_unregister(
    struct llama_context * ctx,
    llamaedge_kv_install_fn fn
);

// Layer iteration: export all pending layers for a sequence
// Returns number of layers exported
LLAMA_API int llamaedge_kv_export_layers(
    struct llama_context * ctx,
    int32_t seq_id,
    uint32_t n_tokens
);

// Layer iteration: install a chunk for a specific layer
LLAMA_API int llamaedge_kv_install_chunk(
    struct llama_context * ctx,
    int32_t seq_id,
    int32_t layer_id,
    const float * k_data,
    const float * v_data,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_k,
    uint32_t n_embd_head_v
);

// ============================================================
// B3: Warm Threshold
// ============================================================

// Warm threshold configuration
typedef struct llamaedge_warm_config {
    uint32_t n_warm_layers;       // L_warm: layers needed for warm decode
    uint32_t n_warm_chunks;        // K_warm: contiguous chunks needed for warm decode
    uint32_t chunk_size_tokens;    // tokens per chunk
} llamaedge_warm_config;

// Frontier management
LLAMA_API llamaedge_frontier * llamaedge_frontier_create(
    int32_t session_id,
    uint32_t n_layers
);
LLAMA_API void llamaedge_frontier_destroy(llamaedge_frontier * frontier);

// Update frontier after installing a chunk
LLAMA_API void llamaedge_frontier_update(
    llamaedge_frontier * frontier,
    int32_t layer_id,
    uint32_t token_begin,
    uint32_t token_count
);

// Check if warm threshold is met
// Returns true if decode can start
LLAMA_API bool llamaedge_warm_threshold_met(
    const llamaedge_frontier * frontier,
    int32_t current_decode_pos,
    const llamaedge_warm_config * config
);

// Get minimum frontier across all layers
LLAMA_API int32_t llamaedge_frontier_min(
    const llamaedge_frontier * frontier
);

// ============================================================
// B4: Full EdgeWeaver - Thinning + Quantization + Control Plane
// ============================================================
//
// B4 adds on B3:
//   - Prompt-side thinning (L2-Norm based token selection)
//   - K/V quantization (Q3: K4/V3 + WHT)
//   - Mode-aware runtime (conservative/balanced/aggressive profiles)
//   - Full control plane integration

// Thinning policy types
typedef enum llamaedge_thinning_policy {
    LLAMAEDGE_THIN_NONE = 0,      // No thinning
    LLAMAEDGE_THIN_L2_NORM = 1,   // L2-Norm based thinning
} llamaedge_thinning_policy;

// Quantization bitplan for K and V
typedef enum llamaedge_quant_bitplan {
    LLAMAEDGE_Q3_K4V3 = 0,   // Q3: K=4bit, V=3bit with WHT
    LLAMAEDGE_Q4_K4V4 = 1,   // Q4: K=4bit, V=4bit
    LLAMAEDGE_Q4_K4V3 = 2,   // Q4: K=4bit, V=3bit with WHT
    LLAMAEDGE_Q3_K3V2 = 3,   // Legacy experimental K=3bit, V=2bit with WHT
} llamaedge_quant_bitplan;

// Profile types for mode-aware runtime
typedef enum llamaedge_profile {
    LLAMAEDGE_PROFILE_CONSERVATIVE = 0,  // Minimal compression
    LLAMAEDGE_PROFILE_BALANCED = 1,       // Default profile
    LLAMAEDGE_PROFILE_AGGRESSIVE = 2,     // Maximum compression
} llamaedge_profile;

// B4 thinning configuration
typedef struct llamaedge_thinning_config {
    bool        enabled;                    // Enable prompt-side thinning
    llamaedge_thinning_policy policy;        // Thinning policy
    uint32_t    top_k;                      // Keep top K tokens (0 = use ratio)
    float       top_ratio;                  // Keep ratio of tokens (0 = use k)
    uint32_t    protected_token_count;      // Number of first tokens to always keep
} llamaedge_thinning_config;

// B4 quantization configuration
typedef struct llamaedge_quant_config {
    bool        enabled;                    // Enable K/V quantization
    llamaedge_quant_bitplan bitplan;        // Quantization bitplan
    bool        use_wht;                    // Use Walsh-Hadamard transform for V
} llamaedge_quant_config;

// B4 complete configuration
typedef struct llamaedge_b4_config {
    llamaedge_profile         profile;       // Active profile selection
    llamaedge_thinning_config thinning;     // Thinning settings
    llamaedge_quant_config    quant;        // Quantization settings
    llamaedge_warm_config     warm;         // Warm threshold settings
    uint32_t                  chunk_size_tokens; // Tokens per chunk
} llamaedge_b4_config;

// ============================================================
// B4: Thinning API
// ============================================================

// Score tokens using L2-Norm and select tokens to keep
// Returns number of tokens selected
LLAMA_API uint32_t llamaedge_thinning_score_and_select(
    const float * token_embeddings,     // [n_tokens][n_embd] row-major
    uint32_t n_tokens,
    uint32_t n_embd,
    const llamaedge_thinning_config * config,
    uint32_t * selected_indices          // [n_tokens] output indices
);

// Compute L2-Norm scores for tokens
// Returns normalized scores in-place
LLAMA_API void llamaedge_thinning_l2_scores(
    const float * token_embeddings,
    uint32_t n_tokens,
    uint32_t n_embd,
    float * scores_out                   // [n_tokens] output scores
);

// ============================================================
// B4: Quantization API
// ============================================================

// Quantize K tensor (per-channel scaling)
// Returns bytes needed for quantized data
LLAMA_API size_t llamaedge_quantize_k(
    const float * k_data,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_k,
    const llamaedge_quant_config * config,
    uint8_t * quantized_out,
    float * scales_out
);

// Dequantize K tensor
LLAMA_API void llamaedge_dequantize_k(
    const uint8_t * quantized,
    const float * scales,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_k,
    const llamaedge_quant_config * config,
    float * k_data_out
);

// Quantize V tensor (per-token scaling, optional WHT)
// Returns bytes needed for quantized data
LLAMA_API size_t llamaedge_quantize_v(
    const float * v_data,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_v,
    const llamaedge_quant_config * config,
    uint8_t * quantized_out,
    float * scales_out
);

// Dequantize V tensor
LLAMA_API void llamaedge_dequantize_v(
    const uint8_t * quantized,
    const float * scales,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_v,
    const llamaedge_quant_config * config,
    float * v_data_out
);

// ============================================================
// B4: Control Plane Integration
// ============================================================

// Control plane mode/profile update callback type
typedef void (*llamaedge_profile_change_fn)(
    llamaedge_profile old_profile,
    llamaedge_profile new_profile,
    void * user_data
);

// Register profile change callback
LLAMA_API int llamaedge_profile_change_register(
    struct llama_context * ctx,
    llamaedge_profile_change_fn fn,
    void * user_data
);

// Unregister profile change callback
LLAMA_API int llamaedge_profile_change_unregister(
    struct llama_context * ctx,
    llamaedge_profile_change_fn fn
);

// Apply new profile configuration
LLAMA_API int llamaedge_profile_apply(
    struct llama_context * ctx,
    llamaedge_profile profile
);

// Get current active profile
LLAMA_API llamaedge_profile llamaedge_profile_get(
    const struct llama_context * ctx
);

// ============================================================
// B4: Complete Pipeline State
// ============================================================

// B4 pipeline state handle
typedef struct llamaedge_b4_pipeline llamaedge_b4_pipeline;

// Create B4 pipeline with configuration
LLAMA_API llamaedge_b4_pipeline * llamaedge_b4_pipeline_create(
    const llamaedge_b4_config * config
);

// Destroy B4 pipeline
LLAMA_API void llamaedge_b4_pipeline_destroy(llamaedge_b4_pipeline * pipeline);

// Get active profile
LLAMA_API llamaedge_profile llamaedge_b4_pipeline_get_profile(
    const llamaedge_b4_pipeline * pipeline
);

// Update pipeline profile and apply corresponding thinning/quant configs
// This mirrors Python's COMPRESSION_PROFILES mapping
LLAMA_API int llamaedge_b4_pipeline_update_profile(
    llamaedge_b4_pipeline * pipeline,
    llamaedge_profile profile
);

// Get thinning config
LLAMA_API const llamaedge_thinning_config * llamaedge_b4_pipeline_get_thinning_config(
    const llamaedge_b4_pipeline * pipeline
);

// Get quant config
LLAMA_API const llamaedge_quant_config * llamaedge_b4_pipeline_get_quant_config(
    const llamaedge_b4_pipeline * pipeline
);

// Get warm config
LLAMA_API const llamaedge_warm_config * llamaedge_b4_pipeline_get_warm_config(
    const llamaedge_b4_pipeline * pipeline
);

// Process prefill with thinning and quantization
// Returns number of chunks produced
LLAMA_API uint32_t llamaedge_b4_pipeline_prefill(
    llamaedge_b4_pipeline * pipeline,
    struct llama_context * ctx,
    int32_t seq_id,
    uint32_t n_tokens,
    const float * embeddings,           // [n_tokens][n_embd] prompt embeddings
    llamaedge_tx_ring * tx_ring
);

// Install received chunk (on decode side)
// Returns 0 on success
LLAMA_API int llamaedge_b4_pipeline_install_chunk(
    llamaedge_b4_pipeline * pipeline,
    struct llama_context * ctx,
    int32_t seq_id,
    const uint8_t * quantized_k,
    const float * k_scales,
    const uint8_t * quantized_v,
    const float * v_scales,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_k,
    uint32_t n_embd_head_v
);

// Chunk header structure for wire transmission
typedef struct llamaedge_chunk_header {
    int32_t     chunk_id;
    int32_t     layer_id;
    uint32_t    token_begin;
    uint32_t    token_count;
    uint8_t     k_bits;
    uint8_t     v_bits;
    bool        use_wht;
    uint32_t    n_kv_heads;
    uint32_t    n_embd_head_k;
    uint32_t    n_embd_head_v;
} llamaedge_chunk_header;

// ============================================================
// Phase 2 Option 1+4: KV State Wire Format Parser
// ============================================================
//
// Parses serialized KV state from llama_state_seq_get_data() to extract
// per-layer K/V tensors without requiring llama.cpp internal API changes.
//
// Wire format (from llama_kv_cache::state_write_data):
//   - uint32_t v_trans
//   - uint32_t n_layer
//   - For each layer K:
//     - int32_t k_type_i
//     - uint64_t k_size_row
//     - For each cell range: k tensor data
//   - For each layer V (if !v_trans):
//     - int32_t v_type_i
//     - uint64_t v_size_row
//     - For each cell range: v tensor data
//   - For each layer V (if v_trans):
//     - int32_t v_type_i
//     - uint32_t v_size_el
//     - uint32_t n_embd_v_gqa
//     - For each dim: For each cell range: v tensor data

// Per-layer K/V data extracted from serialized state
typedef struct llamaedge_kv_layer_data {
    int32_t layer_id;

    // K tensor info
    int32_t  k_type;
    uint64_t k_size_row;
    uint32_t k_n_tokens;       // total tokens across all ranges
    const uint8_t * k_data;    // pointer into serialized buffer

    // V tensor info
    int32_t  v_type;
    uint64_t v_size_row;
    uint32_t v_n_tokens;
    const uint8_t * v_data;    // pointer into serialized buffer

    // For transposed V
    bool     v_trans;
    uint32_t v_size_el;
    uint32_t n_embd_v_gqa;
    const uint8_t * v_data_trans;  // pointer to transposed V data
} llamaedge_kv_layer_data;

// Parsed KV state result
typedef struct llamaedge_kv_parsed_state {
    bool v_trans;
    uint32_t n_layer;
    uint32_t n_tokens;              // total tokens
    llamaedge_kv_layer_data * layers;  // [n_layer] array
    const uint8_t * payload_end;   // pointer past end of parsed data
} llamaedge_kv_parsed_state;

// Parse serialized KV state to extract per-layer tensors
// Returns parsed state on success, nullptr on error
LLAMA_API llamaedge_kv_parsed_state * llamaedge_kv_parse_state(
    const uint8_t * serialized,
    size_t size
);

// Free parsed state
LLAMA_API void llamaedge_kv_parse_state_free(llamaedge_kv_parsed_state * state);

// Serialize chunk header to wire format (JSON)
// Returns bytes written
LLAMA_API size_t llamaedge_b4_serialize_chunk_header(
    const llamaedge_chunk_header * hdr,
    uint8_t * dst,
    size_t dst_size
);

// Parse chunk header from wire format (JSON)
// Returns true on success
LLAMA_API bool llamaedge_b4_parse_chunk_header(
    const uint8_t * src,
    size_t src_size,
    llamaedge_chunk_header * hdr_out
);

// Get the exact size needed for full sequence state
LLAMA_API size_t llamaedge_state_get_size(struct llama_context * ctx, int32_t seq_id);

// Export full sequence state to buffer
LLAMA_API size_t llamaedge_state_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    int32_t seq_id
);

// Import sequence state from buffer
LLAMA_API size_t llamaedge_state_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    int32_t seq_id
);

// ============================================================
// Hook Management API (from B2)
// ============================================================

// Initialize hook registry for a context
LLAMA_API void llamaedge_hook_init(struct llama_context * ctx);

// Register a hook
LLAMA_API int llamaedge_hook_register(struct llama_context * ctx, const llamaedge_hook * hook);

// Unregister a hook
LLAMA_API int llamaedge_hook_unregister(struct llama_context * ctx, const llamaedge_hook * hook);

// Check if a hook type is active
LLAMA_API bool llamaedge_hook_is_active(struct llama_context * ctx, uint32_t hook_type);

// ============================================================
// Compile-time Flags
// ============================================================

// LLAMAEDGE_ENABLE_HOOKS: Enable hook infrastructure (default: off)
// LLAMAEDGE_HOOK_VERBOSE: Verbose hook logging (default: off)
// LLAMAEDGE_B3_STREAMING: Enable B3 chunk streaming (default: off)
// LLAMAEDGE_B4_EDGEWEAVER: Enable B4 Full EdgeWeaver (thinning + quant + control plane)

#ifdef __cplusplus
}
#endif

#endif // LLAMAEDGE_HOOKS_H
