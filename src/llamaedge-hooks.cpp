// llamaedge hooks implementation
// Hook infrastructure for EdgeWeaver Phase 2 B3 chunk streaming
// Adds on B2: per-layer KV export, TX ring, payload pool, frontier, warm threshold

#include "llama.h"
#include "llamaedge/hooks.h"
#include "ggml-backend.h"

#include <algorithm>  // for std::min
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

// ============================================================
// Internal Structures
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
// Using raw pointer to avoid vector resize issues with non-movable types
struct llamaedge_tx_ring_sync {
    std::mutex mutex;
    std::condition_variable cv;
};

// TX descriptor ring for one destination
struct llamaedge_tx_ring_dest {
    std::vector<llamaedge_tx_desc> descriptors;  // ring buffer
    std::vector<uint32_t> in_use;               // in-use flags
    uint32_t head = 0;                          // pop position
    uint32_t tail = 0;                          // push position
    uint32_t count = 0;                         // current count
    llamaedge_tx_ring_sync * sync;              // pointer to sync (dynamically allocated)
};

// TX ring state
struct llamaedge_tx_ring {
    uint32_t ring_size;
    uint32_t payload_pool_bytes;
    uint32_t n_destinations;
    std::vector<llamaedge_tx_ring_dest> destinations;
    std::vector<uint8_t> payload_pool;          // shared payload pool
    std::mutex pool_mutex;
};

// Hook registry stored in context extension
struct llamaedge_hook_registry {
    llamaedge_kv_export_entry  * kv_export_hooks = nullptr;
    llamaedge_kv_install_entry * kv_install_hooks = nullptr;
    llamaedge_hook             * general_hooks = nullptr;
    uint32_t                    active_hooks = 0;
};

using llamaedge_pending_span = std::pair<int32_t, int32_t>;
using llamaedge_pending_span_table = std::vector<std::vector<llamaedge_pending_span>>;

static llamaedge_pending_span_table * llamaedge_frontier_pending(llamaedge_frontier * frontier) {
    return reinterpret_cast<llamaedge_pending_span_table *>(frontier->pending_spans);
}

static void llamaedge_frontier_add_pending(
    llamaedge_frontier * frontier,
    int32_t layer_id,
    int32_t token_begin,
    int32_t token_end
) {
    auto & spans = (*llamaedge_frontier_pending(frontier))[layer_id];
    spans.emplace_back(token_begin, token_end);
    std::sort(spans.begin(), spans.end(), [](const auto & left, const auto & right) {
        if (left.first != right.first) {
            return left.first < right.first;
        }
        return left.second < right.second;
    });

    std::vector<llamaedge_pending_span> merged;
    for (const auto & span : spans) {
        if (merged.empty() || span.first > merged.back().second) {
            merged.push_back(span);
        } else {
            merged.back().second = std::max(merged.back().second, span.second);
        }
    }
    spans.swap(merged);
}

static void llamaedge_frontier_merge_pending(llamaedge_frontier * frontier, int32_t layer_id) {
    auto & spans = (*llamaedge_frontier_pending(frontier))[layer_id];
    int32_t & contiguous_frontier = frontier->layer_frontier[layer_id];
    bool progressed = true;

    while (progressed) {
        progressed = false;
        for (auto it = spans.begin(); it != spans.end(); ) {
            if (it->first <= contiguous_frontier) {
                contiguous_frontier = std::max(contiguous_frontier, it->second);
                it = spans.erase(it);
                progressed = true;
            } else {
                ++it;
            }
        }
    }
}

// ============================================================
// Hook Registry (stored in static for now, should be in context)
// ============================================================

#ifdef LLAMAEDGE_ENABLE_HOOKS

// Per-context hook registry - for simplicity using a static pointer approach
// In production, this should be stored in context extension
static thread_local llamaedge_hook_registry * g_registry = nullptr;

static llamaedge_hook_registry * get_registry(struct llama_context * ctx) {
    (void)ctx;
    if (!g_registry) {
        g_registry = new llamaedge_hook_registry();
    }
    return g_registry;
}

#else

static inline llamaedge_hook_registry * get_registry(struct llama_context * ctx) {
    (void)ctx;
    return nullptr;
}

#endif

// ============================================================
// B3: TX Ring Implementation
// ============================================================

LLAMA_API llamaedge_tx_ring * llamaedge_tx_ring_create(
    const llamaedge_tx_ring_config * config
) {
    if (!config || config->ring_size == 0 || config->n_destinations == 0) {
        return nullptr;
    }

    llamaedge_tx_ring * ring = new llamaedge_tx_ring();
    ring->ring_size = config->ring_size;
    ring->payload_pool_bytes = config->payload_pool_bytes;
    ring->n_destinations = config->n_destinations;
    ring->payload_pool.resize(config->payload_pool_bytes);

    ring->destinations.resize(config->n_destinations);
    for (uint32_t i = 0; i < config->n_destinations; ++i) {
        ring->destinations[i].descriptors.resize(config->ring_size);
        ring->destinations[i].in_use.resize(config->ring_size, 0);
        ring->destinations[i].sync = new llamaedge_tx_ring_sync();
    }

    return ring;
}

LLAMA_API void llamaedge_tx_ring_destroy(llamaedge_tx_ring * ring) {
    if (ring) {
        // Clean up sync objects
        for (uint32_t i = 0; i < ring->n_destinations; ++i) {
            delete ring->destinations[i].sync;
        }
        delete ring;
    }
}

LLAMA_API int llamaedge_tx_ring_push(
    llamaedge_tx_ring * ring,
    uint32_t destination,
    const llamaedge_tx_desc * desc
) {
    if (!ring || !desc || destination >= ring->n_destinations) {
        return -1;
    }

    auto & dest = ring->destinations[destination];
    std::lock_guard<std::mutex> lock(dest.sync->mutex);

    if (dest.count >= ring->ring_size) {
        return -1;  // ring full
    }

    uint32_t idx = dest.tail % ring->ring_size;
    dest.descriptors[idx] = *desc;
    dest.descriptors[idx].in_use = true;
    dest.tail++;
    dest.count++;

    return 0;
}

LLAMA_API int llamaedge_tx_ring_pop(
    llamaedge_tx_ring * ring,
    uint32_t destination,
    llamaedge_tx_desc * desc
) {
    if (!ring || !desc || destination >= ring->n_destinations) {
        return -1;
    }

    auto & dest = ring->destinations[destination];
    std::lock_guard<std::mutex> lock(dest.sync->mutex);

    if (dest.count == 0) {
        return -1;  // ring empty
    }

    uint32_t idx = dest.head % ring->ring_size;
    *desc = dest.descriptors[idx];
    dest.descriptors[idx].in_use = false;
    dest.head++;
    dest.count--;

    return 0;
}

LLAMA_API uint32_t llamaedge_tx_ring_pending(
    const llamaedge_tx_ring * ring,
    uint32_t destination
) {
    if (!ring || destination >= ring->n_destinations) {
        return 0;
    }
    return ring->destinations[destination].count;
}

LLAMA_API uint8_t * llamaedge_payload_pool_alloc(
    llamaedge_tx_ring * ring,
    size_t size
) {
    if (!ring) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(ring->pool_mutex);

    // Simple first-fit allocation
    // In production, should use a better allocator
    static size_t pool_offset = 0;

    if (pool_offset + size > ring->payload_pool_bytes) {
        pool_offset = 0;  // wrap around - should track allocations properly
    }

    uint8_t * result = ring->payload_pool.data() + pool_offset;
    pool_offset += size;

    return result;
}

LLAMA_API void llamaedge_payload_pool_free(
    llamaedge_tx_ring * ring,
    uint8_t * payload
) {
    (void)ring;
    (void)payload;
    // Shared pool - free is a no-op for now
    // In production, should track and reclaim
}

// ============================================================
// B3: Per-Layer KV Export Hooks
// ============================================================

LLAMA_API int llamaedge_kv_export_register(
    struct llama_context * ctx,
    llamaedge_kv_export_fn fn,
    void * user_data
) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx || !fn) {
        return -1;
    }

    auto * reg = get_registry(ctx);

    auto * entry = new llamaedge_kv_export_entry();
    entry->fn = fn;
    entry->user_data = user_data;
    entry->next = reg->kv_export_hooks;
    reg->kv_export_hooks = entry;
    reg->active_hooks |= LLAMAEDGE_HOOK_KV_EXPORT;

    return 0;
#else
    (void)ctx; (void)fn; (void)user_data;
    return -1;
#endif
}

LLAMA_API int llamaedge_kv_export_unregister(
    struct llama_context * ctx,
    llamaedge_kv_export_fn fn
) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx || !fn) {
        return -1;
    }

    auto * reg = get_registry(ctx);
    auto ** prev = &reg->kv_export_hooks;
    auto * curr = reg->kv_export_hooks;

    while (curr) {
        if (curr->fn == fn) {
            *prev = curr->next;
            delete curr;
            return 0;
        }
        prev = &curr->next;
        curr = curr->next;
    }

    return -1;
#else
    (void)ctx; (void)fn;
    return -1;
#endif
}

LLAMA_API int llamaedge_kv_install_register(
    struct llama_context * ctx,
    llamaedge_kv_install_fn fn,
    void * user_data
) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx || !fn) {
        return -1;
    }

    auto * reg = get_registry(ctx);

    auto * entry = new llamaedge_kv_install_entry();
    entry->fn = fn;
    entry->user_data = user_data;
    entry->next = reg->kv_install_hooks;
    reg->kv_install_hooks = entry;
    reg->active_hooks |= LLAMAEDGE_HOOK_KV_INSTALL;

    return 0;
#else
    (void)ctx; (void)fn; (void)user_data;
    return -1;
#endif
}

LLAMA_API int llamaedge_kv_install_unregister(
    struct llama_context * ctx,
    llamaedge_kv_install_fn fn
) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx || !fn) {
        return -1;
    }

    auto * reg = get_registry(ctx);
    auto ** prev = &reg->kv_install_hooks;
    auto * curr = reg->kv_install_hooks;

    while (curr) {
        if (curr->fn == fn) {
            *prev = curr->next;
            delete curr;
            return 0;
        }
        prev = &curr->next;
        curr = curr->next;
    }

    return -1;
#else
    (void)ctx; (void)fn;
    return -1;
#endif
}

// ============================================================
// B3: Layer Iteration API
// ============================================================

// ============================================================
// B3: Per-Layer KV Export Hooks - Internal Access
// ============================================================

// Forward declarations for llama.cpp internal types
struct llama_kv_cache;
struct llama_kv_cache_context;

// Access the internal KV cache from context's memory
// Returns nullptr if memory is not a standard KV cache
static struct llama_kv_cache * get_kv_cache(struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    // Get the memory interface from context
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return nullptr;
    }

    // The memory is actually a llama_kv_cache, but it's internal
    // For now, we'll work with what we can access through public API
    // In a full implementation, we'd need internal API access or friend access
    return nullptr;  // Placeholder - full access requires llama.cpp internal API
}

// Get model hyperparameters for KV export
static void get_model_hparams(struct llama_context * ctx,
                              int32_t * n_layers_out,
                              int32_t * n_kv_heads_out,
                              int32_t * n_embd_head_k_out,
                              int32_t * n_embd_head_v_out) {
    if (!ctx || !n_layers_out || !n_kv_heads_out || !n_embd_head_k_out || !n_embd_head_v_out) {
        return;
    }

    const struct llama_model * model = llama_get_model(ctx);
    if (!model) {
        return;
    }

    *n_layers_out = llama_model_n_layer(model);
    *n_kv_heads_out = llama_model_n_head_kv(model);
    // Note: head_dim access would need additional API
    // Use common default or query differently
    *n_embd_head_k_out = llama_model_n_embd(model) / llama_model_n_head(model);
    *n_embd_head_v_out = llama_model_n_embd(model) / llama_model_n_head(model);
}

// Copy K/V data from a ggml tensor to float buffer
// Handles different quantization types by using ggml backend tensor get
static size_t extract_kv_tensor_f32(const struct ggml_tensor * src,
                                   float * dst,
                                   size_t n_elements) {
    if (!src || !dst || n_elements == 0) {
        return 0;
    }

    // Use ggml_backend_tensor_get which works for all backends
    // This reads from device memory to host memory
    std::vector<float> temp(n_elements > 16384 ? 16384 : n_elements);

    size_t copied = 0;
    size_t offset = 0;

    while (copied < n_elements) {
        size_t batch = std::min(n_elements - copied, temp.size());
        ggml_backend_tensor_get(src, temp.data(), offset * sizeof(float), batch * sizeof(float));

        // Copy to destination (handles non-contiguous if needed)
        memcpy(dst + copied, temp.data(), batch * sizeof(float));
        offset += batch;
        copied += batch;
    }

    return copied * sizeof(float);
}

// Copy K/V data from a serialized buffer to float buffer
// Handles different quantization types
static size_t deserialize_kv_tensor(const uint8_t * src,
                                   float * dst,
                                   size_t n_elements,
                                   int32_t tensor_type) {
    if (!src || !dst || n_elements == 0) {
        return 0;
    }

    // The tensor data from state_write_data is raw ggml tensor data
    // For float type (GGML_TYPE_F32 = 0), directly copy
    if (tensor_type == GGML_TYPE_F32 || tensor_type == 0) {
        memcpy(dst, src, n_elements * sizeof(float));
        return n_elements * sizeof(float);
    }

    // For other types, would need dequantization - not implemented yet
    // For now, just return 0 to indicate no data
    (void)dst;
    return 0;
}

LLAMA_API int llamaedge_kv_export_layers(
    struct llama_context * ctx,
    int32_t seq_id,
    uint32_t n_tokens
) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx) {
        return -1;
    }

    auto * reg = get_registry(ctx);
    if (!reg->kv_export_hooks) {
        return 0;
    }

    // Get model hyperparameters
    int32_t n_layers = 1;
    int32_t n_kv_heads = 1;
    int32_t n_embd_head_k = 128;
    int32_t n_embd_head_v = 128;

    get_model_hparams(ctx, &n_layers, &n_kv_heads, &n_embd_head_k, &n_embd_head_v);

    // Get serialized KV state size
    size_t state_size = llama_state_seq_get_size(ctx, seq_id);
    if (state_size == 0) {
        return 0;
    }

    // Allocate buffer for serialized state
    std::vector<uint8_t> state_buffer(state_size);

    // Get serialized state
    size_t actual_size = llama_state_seq_get_data(ctx, state_buffer.data(), state_size, seq_id);
    if (actual_size == 0 || actual_size > state_size) {
        return 0;
    }

    // Parse the wire format to extract per-layer tensors
    llamaedge_kv_parsed_state * parsed = llamaedge_kv_parse_state(state_buffer.data(), actual_size);
    if (!parsed) {
        return 0;
    }

    // Allocate buffers for K and V tensors per layer
    size_t k_buf_size = (size_t)n_tokens * n_kv_heads * n_embd_head_k * sizeof(float);
    size_t v_buf_size = (size_t)n_tokens * n_kv_heads * n_embd_head_v * sizeof(float);
    std::vector<float> k_buf(k_buf_size / sizeof(float));
    std::vector<float> v_buf(v_buf_size / sizeof(float));

    // Iterate all layers and call registered hooks with parsed data
    int exported = 0;
    uint32_t actual_n_tokens = n_tokens > 0 ? n_tokens : parsed->n_tokens;
    if (actual_n_tokens == 0) actual_n_tokens = 32; // fallback

    for (int32_t layer = 0; layer < n_layers && layer < (int32_t)parsed->n_layer; layer++) {
        llamaedge_kv_layer_data * layer_data = &parsed->layers[layer];

        // Extract K tensor
        if (layer_data->k_data && layer_data->k_size_row > 0) {
            // Compute expected K row bytes: n_kv_heads * n_embd_head_k * sizeof(float)
            uint64_t expected_row_bytes = (uint64_t)n_kv_heads * n_embd_head_k * sizeof(float);
            if (layer_data->k_size_row == expected_row_bytes) {
                // Data layout in serialized buffer: contiguously stored per token
                // Copy the K data for n_tokens
                size_t k_copy_size = std::min(
                    (size_t)layer_data->k_size_row * actual_n_tokens,
                    k_buf_size
                );
                deserialize_kv_tensor(layer_data->k_data, k_buf.data(),
                                      k_copy_size / sizeof(float),
                                      layer_data->k_type);
            }
        }

        // Extract V tensor
        if (!layer_data->v_trans && layer_data->v_data && layer_data->v_size_row > 0) {
            uint64_t expected_row_bytes = (uint64_t)n_kv_heads * n_embd_head_v * sizeof(float);
            if (layer_data->v_size_row == expected_row_bytes) {
                size_t v_copy_size = std::min(
                    (size_t)layer_data->v_size_row * actual_n_tokens,
                    v_buf_size
                );
                deserialize_kv_tensor(layer_data->v_data, v_buf.data(),
                                      v_copy_size / sizeof(float),
                                      layer_data->v_type);
            }
        } else if (layer_data->v_trans && layer_data->v_data_trans) {
            // Transposed V: layout is [n_embd_v_gqa][n_tokens][n_kv_heads] elements
            // but stored as [n_embd_v_gqa][n_tokens * n_kv_heads * elem_size]
            // The Python side will handle de-transpose
            // For now, we just pass the pointer
        }

        // Call hook for this layer with real model dimensions
        auto * hook = reg->kv_export_hooks;
        while (hook) {
            hook->fn(ctx, hook->user_data, layer, seq_id,
                    k_buf.data(), v_buf.data(), actual_n_tokens,
                    (uint32_t)n_kv_heads,
                    (uint32_t)n_embd_head_k,
                    (uint32_t)n_embd_head_v);
            exported++;
            hook = hook->next;
        }
    }

    llamaedge_kv_parse_state_free(parsed);
    return exported;
#else
    (void)ctx; (void)seq_id; (void)n_tokens;
    return -1;
#endif
}

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
) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx) {
        return -1;
    }

    auto * reg = get_registry(ctx);
    auto * hook = reg->kv_install_hooks;

    while (hook) {
        int ret = hook->fn(ctx, hook->user_data, layer_id, seq_id,
                          k_data, v_data, n_tokens,
                          n_kv_heads, n_embd_head_k, n_embd_head_v);
        if (ret != 0) {
            return ret;
        }
        hook = hook->next;
    }

    return 0;
#else
    (void)ctx; (void)seq_id; (void)layer_id;
    (void)k_data; (void)v_data; (void)n_tokens;
    (void)n_kv_heads; (void)n_embd_head_k; (void)n_embd_head_v;
    return -1;
#endif
}

// ============================================================
// B3: Frontier Management
// ============================================================

LLAMA_API llamaedge_frontier * llamaedge_frontier_create(
    int32_t session_id,
    uint32_t n_layers
) {
    llamaedge_frontier * frontier = new llamaedge_frontier();
    frontier->session_id = session_id;
    frontier->n_layers = n_layers;
    frontier->layer_frontier = new int32_t[n_layers];
    frontier->pending_spans = new llamaedge_pending_span_table(n_layers);
    for (uint32_t i = 0; i < n_layers; ++i) {
        frontier->layer_frontier[i] = -1;  // -1 means nothing installed
    }
    return frontier;
}

LLAMA_API void llamaedge_frontier_destroy(llamaedge_frontier * frontier) {
    if (frontier) {
        delete llamaedge_frontier_pending(frontier);
        delete[] frontier->layer_frontier;
        delete frontier;
    }
}

LLAMA_API void llamaedge_frontier_update(
    llamaedge_frontier * frontier,
    int32_t layer_id,
    uint32_t token_begin,
    uint32_t token_count
) {
    if (!frontier || layer_id < 0 || layer_id >= (int32_t)frontier->n_layers) {
        return;
    }

    int32_t current = frontier->layer_frontier[layer_id];
    if (token_count == 0) {
        return;
    }

    int32_t token_end = static_cast<int32_t>(token_begin + token_count);
    bool extends_contiguous = current < 0 ? token_begin == 0 : token_begin <= static_cast<uint32_t>(current);

    if (extends_contiguous) {
        frontier->layer_frontier[layer_id] = std::max(current < 0 ? 0 : current, token_end);
        llamaedge_frontier_merge_pending(frontier, layer_id);
    } else {
        llamaedge_frontier_add_pending(frontier, layer_id, static_cast<int32_t>(token_begin), token_end);
    }
}

LLAMA_API int32_t llamaedge_frontier_min(
    const llamaedge_frontier * frontier
) {
    if (!frontier || frontier->n_layers == 0) {
        return -1;
    }

    int32_t min_frontier = frontier->layer_frontier[0];
    for (uint32_t i = 1; i < frontier->n_layers; ++i) {
        if (frontier->layer_frontier[i] < min_frontier) {
            min_frontier = frontier->layer_frontier[i];
        }
    }
    return min_frontier;
}

// ============================================================
// B3: Warm Threshold Check
// ============================================================

LLAMA_API bool llamaedge_warm_threshold_met(
    const llamaedge_frontier * frontier,
    int32_t current_decode_pos,
    const llamaedge_warm_config * config
) {
    if (!frontier || !config) {
        return false;
    }

    if (current_decode_pos < 0) {
        return false;
    }

    uint32_t K_warm = config->n_warm_chunks;
    uint32_t chunk_size = config->chunk_size_tokens > 0 ? config->chunk_size_tokens : 32;
    uint32_t L_warm = config->n_warm_layers;

    // Clamp L_warm to number of layers
    if (L_warm > frontier->n_layers) {
        L_warm = frontier->n_layers;
    }

    // Calculate required tokens for K_warm chunks
    uint32_t required_tokens = K_warm * chunk_size;

    // Check K_warm contiguous chunks condition: ALL layers must have at least K_warm chunks
    // This ensures contiguity from position 0 across all layers
    bool chunks_condition_met = true;
    for (uint32_t i = 0; i < frontier->n_layers; ++i) {
        int32_t f = frontier->layer_frontier[i];
        if (f < 0 || (uint32_t)f < required_tokens) {
            chunks_condition_met = false;
            break;
        }
    }

    if (chunks_condition_met) {
        return true;  // K_warm chunks met - warm threshold achieved
    }

    // Check L_warm layers condition: first L_warm layers must have frontier > current_decode_pos
    // This is an alternative path to warm, independent of chunk count
    if (L_warm > 0) {
        bool layer_condition_met = true;
        for (uint32_t i = 0; i < L_warm; ++i) {
            int32_t f = frontier->layer_frontier[i];
            if (f < 0 || (int32_t)f <= current_decode_pos) {
                layer_condition_met = false;
                break;
            }
        }
        if (layer_condition_met) {
            return true;
        }
    }

    return false;
}

// ============================================================
// B2 Baseline: State Export/Import (unchanged from B2)
// ============================================================

LLAMA_API size_t llamaedge_state_get_size(struct llama_context * ctx, int32_t seq_id) {
    if (!ctx) {
        return 0;
    }
    return llama_state_seq_get_size(ctx, seq_id);
}

LLAMA_API size_t llamaedge_state_get_data(
        struct llama_context * ctx,
        uint8_t * dst,
        size_t   size,
        int32_t  seq_id) {
    if (!ctx || !dst) {
        return 0;
    }
    return llama_state_seq_get_data(ctx, dst, size, seq_id);
}

LLAMA_API size_t llamaedge_state_set_data(
        struct llama_context * ctx,
        const uint8_t * src,
        size_t   size,
        int32_t  seq_id) {
    if (!ctx || !src) {
        return 0;
    }
    return llama_state_seq_set_data(ctx, src, size, seq_id);
}

// ============================================================
// Phase 2 Option 1+4: KV State Wire Format Parser
// ============================================================

// Helper: read uint32_t from buffer
static const uint8_t * read_u32(const uint8_t * src, size_t size, size_t * offset, uint32_t * out) {
    if (*offset + sizeof(uint32_t) > size) return nullptr;
    memcpy(out, src + *offset, sizeof(uint32_t));
    *offset += sizeof(uint32_t);
    return src + *offset;
}

// Helper: read uint64_t from buffer
static const uint8_t * read_u64(const uint8_t * src, size_t size, size_t * offset, uint64_t * out) {
    if (*offset + sizeof(uint64_t) > size) return nullptr;
    memcpy(out, src + *offset, sizeof(uint64_t));
    *offset += sizeof(uint64_t);
    return src + *offset;
}

// Helper: read int32_t from buffer
static const uint8_t * read_i32(const uint8_t * src, size_t size, size_t * offset, int32_t * out) {
    if (*offset + sizeof(int32_t) > size) return nullptr;
    memcpy(out, src + *offset, sizeof(int32_t));
    *offset += sizeof(int32_t);
    return src + *offset;
}

LLAMA_API llamaedge_kv_parsed_state * llamaedge_kv_parse_state(
    const uint8_t * serialized,
    size_t size
) {
    if (!serialized || size == 0) {
        return nullptr;
    }

    size_t offset = 0;

    // Read header: v_trans and n_layer
    uint32_t v_trans;
    uint32_t n_layer;
    if (!read_u32(serialized, size, &offset, &v_trans)) return nullptr;
    if (!read_u32(serialized, size, &offset, &n_layer)) return nullptr;
    if (n_layer == 0) return nullptr;

    // Allocate result
    llamaedge_kv_parsed_state * result = new llamaedge_kv_parsed_state();
    result->v_trans = (v_trans != 0);
    result->n_layer = n_layer;
    result->n_tokens = 0;
    result->layers = new llamaedge_kv_layer_data[n_layer];

    // Track positions and sizes for K layers
    struct TensorInfo {
        const uint8_t * data;
        uint64_t row_size;
        size_t data_size;
    };
    std::vector<TensorInfo> k_info(n_layer);
    std::vector<TensorInfo> v_info(n_layer);

    // Parse K layer headers
    for (uint32_t il = 0; il < n_layer; il++) {
        int32_t k_type_i;
        if (!read_i32(serialized, size, &offset, &k_type_i)) break;
        k_info[il].row_size = 0;
        if (!read_u64(serialized, size, &offset, &k_info[il].row_size)) break;

        k_info[il].data = serialized + offset;
        k_info[il].data_size = 0; // will be computed after we know n_tokens
    }

    // Parse V layer headers (don't skip data yet)
    for (uint32_t il = 0; il < n_layer; il++) {
        int32_t v_type_i;
        if (!read_i32(serialized, size, &offset, &v_type_i)) break;
        v_info[il].data = serialized + offset;
        v_info[il].row_size = 0;

        if (!result->v_trans) {
            if (!read_u64(serialized, size, &offset, &v_info[il].row_size)) break;
            v_info[il].data_size = 0;
        } else {
            uint32_t v_size_el;
            uint32_t n_embd_v_gqa;
            if (!read_u32(serialized, size, &offset, &v_size_el)) break;
            if (!read_u32(serialized, size, &offset, &n_embd_v_gqa)) break;

            // For transposed V, store element size in row_size field as marker
            v_info[il].row_size = v_size_el;
            v_info[il].data_size = 0; // will be set after we know n_tokens
        }
    }

    // Now we can compute the token count from the structure
    // Total bytes for K data = sum over layers of k_info[il].data_size
    // But we don't know data_size yet either

    // Instead, use remaining buffer to estimate
    // After K+V headers, remaining bytes should be K+V tensor data
    size_t header_end = offset;
    size_t tensor_data_begin = header_end;
    size_t remaining = size - header_end;

    // Rough estimate: assume equal K and V data, and divide by n_layer
    // This assumes one range per layer covering all tokens
    // The Python side will refine based on model dimensions
    size_t per_layer_bytes = remaining / (2 * n_layer);

    // Back-calculate estimated tokens per layer from K row size
    // row_size should be n_kv_heads * n_embd_head * element_size
    // But we don't know n_kv_heads/n_embd_head here. Use a reasonable minimum.
    uint32_t est_tokens = 32; // default assumption
    for (uint32_t il = 0; il < n_layer; il++) {
        if (k_info[il].row_size > 0) {
            // Estimate: row_size / 4 / 32 = n_kv_heads * n_embd_head / 4 / 32
            // For now, just use placeholder - Python side knows real dims
            k_info[il].data_size = (size_t)k_info[il].row_size * est_tokens;
        }
        if (v_info[il].row_size > 0) {
            if (!result->v_trans) {
                v_info[il].data_size = (size_t)v_info[il].row_size * est_tokens;
            } else {
                // Transposed: data_size = v_size_el * n_embd_v_gqa * est_tokens
                v_info[il].data_size = v_info[il].row_size * est_tokens; // row_size stores v_size_el here
            }
        }
    }

    // Now skip past K and V data
    offset = tensor_data_begin;
    for (uint32_t il = 0; il < n_layer; il++) {
        offset += k_info[il].data_size;
    }
    for (uint32_t il = 0; il < n_layer; il++) {
        offset += v_info[il].data_size;
    }

    result->payload_end = serialized + offset;

    // Fill in layer data
    for (uint32_t il = 0; il < n_layer; il++) {
        llamaedge_kv_layer_data * layer = &result->layers[il];
        layer->layer_id = (int32_t)il;
        layer->k_type = GGML_TYPE_F32; // assume float (actual type would need lookup)
        layer->k_size_row = k_info[il].row_size;
        layer->k_n_tokens = est_tokens;
        layer->k_data = k_info[il].data;

        layer->v_type = GGML_TYPE_F32;
        layer->v_size_row = v_info[il].row_size;
        layer->v_n_tokens = est_tokens;
        layer->v_trans = result->v_trans;

        if (!result->v_trans) {
            layer->v_data = v_info[il].data;
            layer->v_data_trans = nullptr;
            layer->v_size_el = 0;
            layer->n_embd_v_gqa = 0;
        } else {
            layer->v_data = nullptr;
            layer->v_data_trans = v_info[il].data;
            layer->v_size_el = (uint32_t)v_info[il].row_size; // row_size stores v_size_el for transposed
            // n_embd_v_gqa is unknown without model dims - placeholder
            layer->n_embd_v_gqa = 128; // placeholder - Python knows real value
        }
    }

    return result;
}

LLAMA_API void llamaedge_kv_parse_state_free(llamaedge_kv_parsed_state * state) {
    if (state) {
        delete[] state->layers;
        delete state;
    }
}

LLAMA_API void llamaedge_hook_init(struct llama_context * ctx) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    (void)ctx;
    // Registry is allocated lazily
#else
    (void)ctx;
#endif
}

LLAMA_API int llamaedge_hook_register(struct llama_context * ctx, const llamaedge_hook * hook) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx || !hook) {
        return -1;
    }

    auto * reg = get_registry(ctx);

    auto * entry = new llamaedge_hook();
    *entry = *hook;
    entry->next = reg->general_hooks;
    reg->general_hooks = entry;
    reg->active_hooks |= hook->hook_type;

    return 0;
#else
    (void)ctx; (void)hook;
    return -1;
#endif
}

LLAMA_API int llamaedge_hook_unregister(struct llama_context * ctx, const llamaedge_hook * hook) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx || !hook) {
        return -1;
    }

    auto * reg = get_registry(ctx);
    auto ** prev = &reg->general_hooks;
    auto * curr = reg->general_hooks;

    while (curr) {
        if (curr->fn == hook->fn) {
            *prev = curr->next;
            delete curr;
            return 0;
        }
        prev = &curr->next;
        curr = curr->next;
    }

    return -1;
#else
    (void)ctx; (void)hook;
    return -1;
#endif
}

LLAMA_API bool llamaedge_hook_is_active(struct llama_context * ctx, uint32_t hook_type) {
#ifdef LLAMAEDGE_ENABLE_HOOKS
    if (!ctx) {
        return false;
    }
    auto * reg = get_registry(ctx);
    return (reg && (reg->active_hooks & hook_type) != 0);
#else
    (void)ctx; (void)hook_type;
    return false;
#endif
}

// ============================================================
// B4: Thinning Implementation
// ============================================================

#include <cmath>
#include <algorithm>

LLAMA_API void llamaedge_thinning_l2_scores(
    const float * token_embeddings,
    uint32_t n_tokens,
    uint32_t n_embd,
    float * scores_out
) {
    if (!token_embeddings || !scores_out || n_tokens == 0 || n_embd == 0) {
        return;
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const float * emb = token_embeddings + i * n_embd;
        float sum_sq = 0.0f;
        for (uint32_t j = 0; j < n_embd; ++j) {
            float val = emb[j];
            sum_sq += val * val;
        }
        scores_out[i] = std::sqrt(sum_sq);
    }
}

static uint32_t select_token_indices_by_l2(
    const float * scores,
    uint32_t n_tokens,
    uint32_t top_k,
    float top_ratio,
    uint32_t protected_count,
    uint32_t * selected_indices
) {
    if (n_tokens == 0 || !selected_indices) {
        return 0;
    }

    uint32_t keep_count = 0;
    if (top_ratio > 0.0f) {
        keep_count = std::max<uint32_t>(1, (uint32_t)(n_tokens * top_ratio));
    } else if (top_k > 0) {
        keep_count = std::min(top_k, n_tokens);
    } else {
        keep_count = n_tokens;
    }

    // Always protect first protected_count tokens
    uint32_t n_selected = 0;
    for (uint32_t i = 0; i < protected_count && i < n_tokens; ++i) {
        selected_indices[n_selected++] = i;
    }

    // Create index list for remaining tokens
    std::vector<uint32_t> remaining;
    remaining.reserve(n_tokens - protected_count);
    for (uint32_t i = protected_count; i < n_tokens; ++i) {
        remaining.push_back(i);
    }

    // Sort by score descending (highest L2 norm first)
    std::sort(remaining.begin(), remaining.end(),
        [&scores](uint32_t a, uint32_t b) {
            if (scores[a] != scores[b]) {
                return scores[a] > scores[b];
            }
            return a < b;
        });

    // Select top keep_count - protected_count from remaining
    uint32_t to_select = (keep_count > protected_count) ? (keep_count - protected_count) : 0;
    for (uint32_t i = 0; i < to_select && i < remaining.size(); ++i) {
        selected_indices[n_selected++] = remaining[i];
    }

    std::sort(selected_indices, selected_indices + n_selected);
    return n_selected;
}

LLAMA_API uint32_t llamaedge_thinning_score_and_select(
    const float * token_embeddings,
    uint32_t n_tokens,
    uint32_t n_embd,
    const llamaedge_thinning_config * config,
    uint32_t * selected_indices
) {
    if (!config || !selected_indices) {
        return 0;
    }

    if (!config->enabled || config->policy == LLAMAEDGE_THIN_NONE) {
        // No thinning - select all tokens
        for (uint32_t i = 0; i < n_tokens; ++i) {
            selected_indices[i] = i;
        }
        return n_tokens;
    }

    if (config->policy == LLAMAEDGE_THIN_L2_NORM) {
        // Compute L2 scores
        std::vector<float> scores(n_tokens);
        llamaedge_thinning_l2_scores(token_embeddings, n_tokens, n_embd, scores.data());

        // Normalize scores
        float min_s = scores[0], max_s = scores[0];
        for (uint32_t i = 1; i < n_tokens; ++i) {
            if (scores[i] < min_s) min_s = scores[i];
            if (scores[i] > max_s) max_s = scores[i];
        }
        float range = max_s - min_s;
        if (range > 0) {
            for (uint32_t i = 0; i < n_tokens; ++i) {
                scores[i] = (scores[i] - min_s) / range;
            }
        }

        return select_token_indices_by_l2(
            scores.data(), n_tokens,
            config->top_k, config->top_ratio,
            config->protected_token_count,
            selected_indices
        );
    }

    return 0;
}

// ============================================================
// B4: Quantization Implementation (Q3: K4/V3 + WHT)
// ============================================================

static int32_t qrange(int32_t bits) {
    return (1 << (bits - 1)) - 1;
}

static void walsh_hadamard_transform_inplace(float * data, uint32_t n) {
    // Check power of two
    if ((n & (n - 1)) != 0) return;

    uint32_t step = 1;
    while (step < n) {
        uint32_t span = step * 2;
        for (uint32_t start = 0; start < n; start += span) {
            for (uint32_t offset = 0; offset < step; ++offset) {
                uint32_t left = start + offset;
                uint32_t right = left + step;
                float a = data[left];
                float b = data[right];
                data[left] = a + b;
                data[right] = a - b;
            }
        }
        step = span;
    }
}

LLAMA_API size_t llamaedge_quantize_k(
    const float * k_data,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_k,
    const llamaedge_quant_config * config,
    uint8_t * quantized_out,
    float * scales_out
) {
    if (!k_data || !config || !quantized_out || !scales_out) {
        return 0;
    }
    if (!config->enabled) {
        return 0;
    }

    // K is quantized per-channel (column-wise)
    // Layout: [n_kv_heads][n_embd_head_k][n_tokens]
    // For simplicity, assume [n_tokens][n_kv_heads][n_embd_head_k] transposed
    // Determine K bits from bitplan
    uint32_t k_bits = (config->bitplan == LLAMAEDGE_Q3_K3V2) ? 3 : 4;
    uint32_t qmax = qrange(k_bits);
    uint32_t total_scales = n_kv_heads * n_embd_head_k;

    for (uint32_t head = 0; head < n_kv_heads; ++head) {
        for (uint32_t dim = 0; dim < n_embd_head_k; ++dim) {
            // Find scale for this channel
            float max_abs = 0.0f;
            for (uint32_t tok = 0; tok < n_tokens; ++tok) {
                uint32_t idx = tok * n_kv_heads * n_embd_head_k + head * n_embd_head_k + dim;
                float abs_val = std::abs(k_data[idx]);
                if (abs_val > max_abs) max_abs = abs_val;
            }
            float scale = (max_abs > 0) ? (max_abs / qmax) : 1.0f;
            scales_out[head * n_embd_head_k + dim] = scale;

            // Quantize
            for (uint32_t tok = 0; tok < n_tokens; ++tok) {
                uint32_t src_idx = tok * n_kv_heads * n_embd_head_k + head * n_embd_head_k + dim;
                uint32_t dst_idx = tok * total_scales + head * n_embd_head_k + dim;
                int32_t q = (int32_t)std::round(k_data[src_idx] / scale);
                if (q > qmax) q = qmax;
                else if (q < -qmax) q = -qmax;
                quantized_out[dst_idx] = (uint8_t)(q + qmax);  // offset to positive
            }
        }
    }

    return n_tokens * total_scales;
}

LLAMA_API void llamaedge_dequantize_k(
    const uint8_t * quantized,
    const float * scales,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_k,
    const llamaedge_quant_config * config,
    float * k_data_out
) {
    if (!quantized || !scales || !k_data_out || !config) {
        return;
    }

    uint32_t k_bits = (config->bitplan == LLAMAEDGE_Q3_K3V2) ? 3 : 4;
    uint32_t qmax = qrange(k_bits);
    uint32_t total_scales = n_kv_heads * n_embd_head_k;

    for (uint32_t head = 0; head < n_kv_heads; ++head) {
        for (uint32_t dim = 0; dim < n_embd_head_k; ++dim) {
            float scale = scales[head * n_embd_head_k + dim];
            for (uint32_t tok = 0; tok < n_tokens; ++tok) {
                uint32_t src_idx = tok * total_scales + head * n_embd_head_k + dim;
                uint32_t dst_idx = tok * n_kv_heads * n_embd_head_k + head * n_embd_head_k + dim;
                int32_t q = (int32_t)quantized[src_idx] - qmax;
                k_data_out[dst_idx] = (float)q * scale;
            }
        }
    }
}

LLAMA_API size_t llamaedge_quantize_v(
    const float * v_data,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_v,
    const llamaedge_quant_config * config,
    uint8_t * quantized_out,
    float * scales_out
) {
    if (!v_data || !config || !quantized_out || !scales_out) {
        return 0;
    }
    if (!config->enabled) {
        return 0;
    }

    // V is quantized per-token (row-wise), optional WHT
    int32_t v_bits;
    switch (config->bitplan) {
        case LLAMAEDGE_Q3_K3V2:
            v_bits = 2;
            break;
        case LLAMAEDGE_Q4_K4V4:
            v_bits = 4;
            break;
        case LLAMAEDGE_Q3_K4V3:
        case LLAMAEDGE_Q4_K4V3:
        default:
            v_bits = 3;
            break;
    }
    uint32_t qmax = qrange(v_bits);

    for (uint32_t tok = 0; tok < n_tokens; ++tok) {
        // Extract row: [n_kv_heads][n_embd_head_v]
        std::vector<float> row(n_kv_heads * n_embd_head_v);
        for (uint32_t head = 0; head < n_kv_heads; ++head) {
            for (uint32_t dim = 0; dim < n_embd_head_v; ++dim) {
                uint32_t src_idx = tok * n_kv_heads * n_embd_head_v + head * n_embd_head_v + dim;
                row[head * n_embd_head_v + dim] = v_data[src_idx];
            }
        }

        // Apply WHT if enabled
        if (config->use_wht && (n_embd_head_v & (n_embd_head_v - 1)) == 0) {
            walsh_hadamard_transform_inplace(row.data(), n_embd_head_v);
        }

        // Find scale for this token
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < row.size(); ++i) {
            if (std::abs(row[i]) > max_abs) max_abs = std::abs(row[i]);
        }
        float scale = (max_abs > 0) ? (max_abs / qmax) : 1.0f;
        scales_out[tok] = scale;

        // Quantize
        for (uint32_t i = 0; i < row.size(); ++i) {
            int32_t q = (int32_t)std::round(row[i] / scale);
            if (q > qmax) q = qmax;
            else if (q < -qmax) q = -qmax;
            quantized_out[tok * n_kv_heads * n_embd_head_v + i] = (uint8_t)(q + qmax);
        }
    }

    return n_tokens * n_kv_heads * n_embd_head_v;
}

LLAMA_API void llamaedge_dequantize_v(
    const uint8_t * quantized,
    const float * scales,
    uint32_t n_tokens,
    uint32_t n_kv_heads,
    uint32_t n_embd_head_v,
    const llamaedge_quant_config * config,
    float * v_data_out
) {
    if (!quantized || !scales || !v_data_out || !config) {
        return;
    }

    int32_t v_bits;
    switch (config->bitplan) {
        case LLAMAEDGE_Q3_K3V2:
            v_bits = 2;
            break;
        case LLAMAEDGE_Q4_K4V4:
            v_bits = 4;
            break;
        case LLAMAEDGE_Q3_K4V3:
        case LLAMAEDGE_Q4_K4V3:
        default:
            v_bits = 3;
            break;
    }
    uint32_t qmax = qrange(v_bits);

    for (uint32_t tok = 0; tok < n_tokens; ++tok) {
        float scale = scales[tok];
        std::vector<float> row(n_kv_heads * n_embd_head_v);

        // Dequantize
        for (uint32_t i = 0; i < row.size(); ++i) {
            int32_t q = (int32_t)quantized[tok * n_kv_heads * n_embd_head_v + i] - qmax;
            row[i] = (float)q * scale;
        }

        // Apply inverse WHT if enabled
        if (config->use_wht && (n_embd_head_v & (n_embd_head_v - 1)) == 0) {
            walsh_hadamard_transform_inplace(row.data(), n_embd_head_v);
            float inv_n = 1.0f / n_embd_head_v;
            for (float & f : row) f *= inv_n;
        }

        // Copy back to output
        for (uint32_t head = 0; head < n_kv_heads; ++head) {
            for (uint32_t dim = 0; dim < n_embd_head_v; ++dim) {
                uint32_t dst_idx = tok * n_kv_heads * n_embd_head_v + head * n_embd_head_v + dim;
                v_data_out[dst_idx] = row[head * n_embd_head_v + dim];
            }
        }
    }
}

// ============================================================
// B4: Profile Change Callbacks
// ============================================================

struct llamaedge_profile_change_entry {
    llamaedge_profile_change_fn fn;
    void * user_data;
    llamaedge_profile_change_entry * next;
};

// Profile state stored per context
struct llamaedge_profile_state {
    llamaedge_profile current_profile = LLAMAEDGE_PROFILE_BALANCED;
    llamaedge_profile_change_entry * callbacks = nullptr;
};

#ifdef LLAMAEDGE_ENABLE_HOOKS
static thread_local llamaedge_profile_state * g_profile_state = nullptr;

static llamaedge_profile_state * get_profile_state(const struct llama_context * ctx) {
    (void)ctx;
    if (!g_profile_state) {
        g_profile_state = new llamaedge_profile_state();
    }
    return g_profile_state;
}
#else
static inline llamaedge_profile_state * get_profile_state(const struct llama_context * ctx) {
    (void)ctx;
    return nullptr;
}
#endif

LLAMA_API int llamaedge_profile_change_register(
    struct llama_context * ctx,
    llamaedge_profile_change_fn fn,
    void * user_data
) {
    if (!ctx || !fn) {
        return -1;
    }

    auto * state = get_profile_state(ctx);
    if (!state) return -1;

    auto * entry = new llamaedge_profile_change_entry();
    entry->fn = fn;
    entry->user_data = user_data;
    entry->next = state->callbacks;
    state->callbacks = entry;

    return 0;
}

LLAMA_API int llamaedge_profile_change_unregister(
    struct llama_context * ctx,
    llamaedge_profile_change_fn fn
) {
    if (!ctx || !fn) {
        return -1;
    }

    auto * state = get_profile_state(ctx);
    if (!state) return -1;

    auto ** prev = &state->callbacks;
    auto * curr = state->callbacks;

    while (curr) {
        if (curr->fn == fn) {
            *prev = curr->next;
            delete curr;
            return 0;
        }
        prev = &curr->next;
        curr = curr->next;
    }

    return -1;
}

LLAMA_API int llamaedge_profile_apply(
    struct llama_context * ctx,
    llamaedge_profile profile
) {
    if (!ctx) {
        return -1;
    }

    auto * state = get_profile_state(ctx);
    if (!state) return -1;

    llamaedge_profile old = state->current_profile;
    if (old == profile) {
        return 0;
    }

    state->current_profile = profile;

    // Notify callbacks
    auto * cb = state->callbacks;
    while (cb) {
        cb->fn(old, profile, cb->user_data);
        cb = cb->next;
    }

    return 0;
}

LLAMA_API llamaedge_profile llamaedge_profile_get(
    const struct llama_context * ctx
) {
    auto * state = get_profile_state(ctx);
    if (!state) {
        return LLAMAEDGE_PROFILE_BALANCED;
    }
    return state->current_profile;
}

// ============================================================
// B4: Wire Format for Chunk Transmission
// ============================================================

// Compute size needed for a chunk header in JSON format
static size_t chunk_header_json_size(const llamaedge_chunk_header * hdr) {
    (void)hdr;
    // Approximate JSON size
    return 256;
}

// Serialize chunk header to JSON
// Returns bytes written, 0 on error
static size_t chunk_header_to_json(
    const llamaedge_chunk_header * hdr,
    char * json_out,
    size_t json_size
) {
    if (!hdr || !json_out || json_size < 256) {
        return 0;
    }

    int written = snprintf(json_out, json_size,
        "{"
        "\"chunk_id\":%d,"
        "\"layer_id\":%d,"
        "\"token_begin\":%u,"
        "\"token_count\":%u,"
        "\"k_bits\":%u,"
        "\"v_bits\":%u,"
        "\"use_wht\":%s,"
        "\"n_kv_heads\":%u,"
        "\"n_embd_head_k\":%u,"
        "\"n_embd_head_v\":%u"
        "}",
        hdr->chunk_id,
        hdr->layer_id,
        hdr->token_begin,
        hdr->token_count,
        hdr->k_bits,
        hdr->v_bits,
        hdr->use_wht ? "true" : "false",
        hdr->n_kv_heads,
        hdr->n_embd_head_k,
        hdr->n_embd_head_v
    );

    return (written > 0 && (size_t)written < json_size) ? (size_t)written : 0;
}

// Parse chunk header from JSON
// Returns true on success, false on error
static bool chunk_header_from_json(
    const char * json_str,
    llamaedge_chunk_header * hdr_out
) {
    if (!json_str || !hdr_out) {
        return false;
    }

    llamaedge_chunk_header hdr = {};

    // Simple field-by-field parsing
    const char * p = json_str;
    while (*p && p - json_str < 1024) {
        if (strncmp(p, "\"chunk_id\":", 11) == 0) {
            sscanf(p + 11, "%d", &hdr.chunk_id);
        } else if (strncmp(p, "\"layer_id\":", 11) == 0) {
            sscanf(p + 11, "%d", &hdr.layer_id);
        } else if (strncmp(p, "\"token_begin\":", 13) == 0) {
            sscanf(p + 13, "%u", &hdr.token_begin);
        } else if (strncmp(p, "\"token_count\":", 14) == 0) {
            sscanf(p + 14, "%u", &hdr.token_count);
        } else if (strncmp(p, "\"k_bits\":", 9) == 0) {
            unsigned int tmp;
            sscanf(p + 9, "%u", &tmp);
            hdr.k_bits = (uint8_t)tmp;
        } else if (strncmp(p, "\"v_bits\":", 9) == 0) {
            unsigned int tmp;
            sscanf(p + 9, "%u", &tmp);
            hdr.v_bits = (uint8_t)tmp;
        } else if (strncmp(p, "\"use_wht\":", 10) == 0) {
            if (strncmp(p + 10, "true", 4) == 0) {
                hdr.use_wht = true;
            }
        } else if (strncmp(p, "\"n_kv_heads\":", 13) == 0) {
            sscanf(p + 13, "%u", &hdr.n_kv_heads);
        } else if (strncmp(p, "\"n_embd_head_k\":", 16) == 0) {
            sscanf(p + 16, "%u", &hdr.n_embd_head_k);
        } else if (strncmp(p, "\"n_embd_head_v\":", 16) == 0) {
            sscanf(p + 16, "%u", &hdr.n_embd_head_v);
        }
        p++;
    }

    *hdr_out = hdr;
    return true;
}

// ============================================================
// B4: Pipeline State
// ============================================================

struct llamaedge_b4_pipeline {
    llamaedge_b4_config config;
    llamaedge_profile current_profile;
    // Working buffers for chunk processing
    std::vector<uint8_t> k_quant_buf;
    std::vector<float> k_scales_buf;
    std::vector<uint8_t> v_quant_buf;
    std::vector<float> v_scales_buf;
};

LLAMA_API llamaedge_b4_pipeline * llamaedge_b4_pipeline_create(
    const llamaedge_b4_config * config
) {
    if (!config) {
        return nullptr;
    }

    auto * pipeline = new llamaedge_b4_pipeline();
    pipeline->config = *config;
    pipeline->current_profile = config->profile;

    // Pre-allocate working buffers based on max chunk size
    uint32_t max_tokens = config->chunk_size_tokens > 0 ? config->chunk_size_tokens * 2 : 128;
    uint32_t max_kv_heads = 32;
    uint32_t max_embd = 128;

    size_t k_size = max_tokens * max_kv_heads * max_embd;
    size_t v_size = max_tokens * max_kv_heads * max_embd;
    size_t scales_size = max_kv_heads * max_embd;

    pipeline->k_quant_buf.resize(k_size);
    pipeline->k_scales_buf.resize(scales_size);
    pipeline->v_quant_buf.resize(v_size);
    pipeline->v_scales_buf.resize(max_tokens);

    return pipeline;
}

LLAMA_API void llamaedge_b4_pipeline_destroy(llamaedge_b4_pipeline * pipeline) {
    if (pipeline) {
        delete pipeline;
    }
}

LLAMA_API llamaedge_profile llamaedge_b4_pipeline_get_profile(
    const llamaedge_b4_pipeline * pipeline
) {
    if (!pipeline) {
        return LLAMAEDGE_PROFILE_BALANCED;
    }
    return pipeline->current_profile;
}

// Map profile enum to thinning and quant configuration
// This mirrors Python's COMPRESSION_PROFILES in control_plane.py
static void profile_to_config(
    llamaedge_profile profile,
    llamaedge_thinning_config * thinning_out,
    llamaedge_quant_config * quant_out
) {
    if (!thinning_out || !quant_out) return;

    // Default: enable thinning and quantization
    thinning_out->enabled = true;
    thinning_out->policy = LLAMAEDGE_THIN_L2_NORM;
    thinning_out->top_k = 0;  // use ratio
    thinning_out->protected_token_count = 4;

    quant_out->enabled = true;
    quant_out->use_wht = true;

    switch (profile) {
        case LLAMAEDGE_PROFILE_CONSERVATIVE:
            // keep_ratio=0.8, k_bits=4, v_bits=4
            thinning_out->top_ratio = 0.8f;
            quant_out->bitplan = LLAMAEDGE_Q4_K4V4;
            break;
        case LLAMAEDGE_PROFILE_AGGRESSIVE:
            // keep_ratio=0.4, k_bits=4, v_bits=3 (V1 frozen aggressive mode)
            thinning_out->top_ratio = 0.4f;
            quant_out->bitplan = LLAMAEDGE_Q4_K4V3;
            break;
        case LLAMAEDGE_PROFILE_BALANCED:
        default:
            // keep_ratio=0.6, k_bits=4, v_bits=3
            thinning_out->top_ratio = 0.6f;
            quant_out->bitplan = LLAMAEDGE_Q3_K4V3;
            break;
    }
}

LLAMA_API int llamaedge_b4_pipeline_update_profile(
    llamaedge_b4_pipeline * pipeline,
    llamaedge_profile profile
) {
    if (!pipeline) {
        return -1;
    }

    llamaedge_profile old = pipeline->current_profile;
    if (old == profile) {
        return 0;  // No change needed
    }

    pipeline->current_profile = profile;

    // Map profile to thinning and quant configs
    profile_to_config(profile, &pipeline->config.thinning, &pipeline->config.quant);

    return 0;
}

LLAMA_API const llamaedge_thinning_config * llamaedge_b4_pipeline_get_thinning_config(
    const llamaedge_b4_pipeline * pipeline
) {
    if (!pipeline) {
        return nullptr;
    }
    return &pipeline->config.thinning;
}

LLAMA_API const llamaedge_quant_config * llamaedge_b4_pipeline_get_quant_config(
    const llamaedge_b4_pipeline * pipeline
) {
    if (!pipeline) {
        return nullptr;
    }
    return &pipeline->config.quant;
}

LLAMA_API const llamaedge_warm_config * llamaedge_b4_pipeline_get_warm_config(
    const llamaedge_b4_pipeline * pipeline
) {
    if (!pipeline) {
        return nullptr;
    }
    return &pipeline->config.warm;
}

LLAMA_API uint32_t llamaedge_b4_pipeline_prefill(
    llamaedge_b4_pipeline * pipeline,
    struct llama_context * ctx,
    int32_t seq_id,
    uint32_t n_tokens,
    const float * embeddings,
    llamaedge_tx_ring * tx_ring
) {
    if (!pipeline || n_tokens == 0) {
        return 0;
    }

    uint32_t chunk_size = pipeline->config.chunk_size_tokens > 0 ?
        pipeline->config.chunk_size_tokens : 32;

    // Step 1: Apply thinning if enabled
    std::vector<uint32_t> selected_indices(n_tokens);
    uint32_t n_selected = n_tokens;

    if (pipeline->config.thinning.enabled &&
        pipeline->config.thinning.policy == LLAMAEDGE_THIN_L2_NORM) {
        n_selected = llamaedge_thinning_score_and_select(
            embeddings,
            n_tokens,
            0,  // n_embd - would need to be passed or derived
            &pipeline->config.thinning,
            selected_indices.data()
        );
    } else {
        // No thinning - select all tokens in order
        for (uint32_t i = 0; i < n_tokens; ++i) {
            selected_indices[i] = i;
        }
        n_selected = n_tokens;
    }

    // Step 2: For each chunk, quantize K/V and push to TX ring
    uint32_t n_chunks = (n_selected + chunk_size - 1) / chunk_size;

    if (tx_ring && pipeline->config.quant.enabled) {
        const llamaedge_quant_config * qcfg = &pipeline->config.quant;

        for (uint32_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
            uint32_t chunk_begin = chunk_id * chunk_size;
            uint32_t chunk_end = std::min(chunk_begin + chunk_size, n_selected);
            uint32_t chunk_tokens = chunk_end - chunk_begin;

            if (chunk_tokens == 0) continue;

            // Build chunk header for this chunk
            llamaedge_chunk_header hdr = {};
            hdr.chunk_id = (int32_t)chunk_id;
            hdr.layer_id = 0;  // Would be set per-layer in real implementation
            hdr.token_begin = chunk_begin;
            hdr.token_count = chunk_tokens;
            // Map bitplan to actual k/v bits
            switch (qcfg->bitplan) {
                case LLAMAEDGE_Q4_K4V4:
                    hdr.k_bits = 4;
                    hdr.v_bits = 4;
                    break;
                case LLAMAEDGE_Q3_K3V2:
                    hdr.k_bits = 3;
                    hdr.v_bits = 2;
                    break;
                case LLAMAEDGE_Q3_K4V3:
                case LLAMAEDGE_Q4_K4V3:
                default:
                    hdr.k_bits = 4;
                    hdr.v_bits = 3;
                    break;
            }
            hdr.use_wht = qcfg->use_wht;
            hdr.n_kv_heads = 0;  // Would be set from model config
            hdr.n_embd_head_k = 0;
            hdr.n_embd_head_v = 0;

            // Serialize header to JSON
            char json_hdr[256];
            size_t json_len = chunk_header_to_json(&hdr, json_hdr, sizeof(json_hdr));

            // Allocate payload in TX ring pool
            size_t est_payload_size = json_len + 512;  // Placeholder for actual quantized data
            uint8_t * payload = llamaedge_payload_pool_alloc(tx_ring, est_payload_size);

            if (payload) {
                // Copy JSON header
                memcpy(payload, json_hdr, json_len);

                // In real implementation:
                // 1. Extract K/V data from layer's attention cache
                // 2. Run llamaedge_quantize_k() and llamaedge_quantize_v()
                // 3. Copy quantized data into payload after JSON header

                llamaedge_tx_desc desc = {};
                desc.session_id = seq_id;
                desc.layer_id = hdr.layer_id;
                desc.chunk_id = hdr.chunk_id;
                desc.token_begin = hdr.token_begin;
                desc.token_count = hdr.token_count;
                desc.payload_size = est_payload_size;
                desc.payload = payload;

                // Push to destination 0 (first decode node)
                llamaedge_tx_ring_push(tx_ring, 0, &desc);
            }
        }
    }

    return n_chunks;
}

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
) {
    if (!pipeline || !ctx) {
        return -1;
    }

    // Step 1: Dequantize K and V
    std::vector<float> k_dequant(n_tokens * n_kv_heads * n_embd_head_k);
    std::vector<float> v_dequant(n_tokens * n_kv_heads * n_embd_head_v);

    llamaedge_quant_config qcfg = pipeline->config.quant;

    if (quantized_k && k_scales) {
        llamaedge_dequantize_k(
            quantized_k,
            k_scales,
            n_tokens,
            n_kv_heads,
            n_embd_head_k,
            &qcfg,
            k_dequant.data()
        );
    }

    if (quantized_v && v_scales) {
        llamaedge_dequantize_v(
            quantized_v,
            v_scales,
            n_tokens,
            n_kv_heads,
            n_embd_head_v,
            &qcfg,
            v_dequant.data()
        );
    }

    // Step 2: Install into KV cache via registered hooks
    int32_t layer_id = 0;  // Would be set based on chunk's layer_id

    int ret = llamaedge_kv_install_chunk(
        ctx,
        seq_id,
        layer_id,
        k_dequant.data(),
        v_dequant.data(),
        n_tokens,
        n_kv_heads,
        n_embd_head_k,
        n_embd_head_v
    );

    return ret;
}

LLAMA_API size_t llamaedge_b4_serialize_chunk_header(
    const llamaedge_chunk_header * hdr,
    uint8_t * dst,
    size_t dst_size
) {
    if (!hdr || !dst || dst_size < 256) {
        return 0;
    }
    return chunk_header_to_json(hdr, (char *)dst, dst_size);
}

LLAMA_API bool llamaedge_b4_parse_chunk_header(
    const uint8_t * src,
    size_t src_size,
    llamaedge_chunk_header * hdr_out
) {
    if (!src || !hdr_out) {
        return false;
    }
    // Ensure null terminator for string parsing
    char json_str[1024];
    size_t copy_size = std::min(src_size, sizeof(json_str) - 1);
    memcpy(json_str, src, copy_size);
    json_str[copy_size] = '\0';
    return chunk_header_from_json(json_str, hdr_out);
}
