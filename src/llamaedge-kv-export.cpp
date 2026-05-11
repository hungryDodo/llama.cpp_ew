#include "llamaedge-internal.h"


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

    auto * reg = llamaedge_get_registry(ctx);

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

    auto * reg = llamaedge_get_registry(ctx);
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

    auto * reg = llamaedge_get_registry(ctx);

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

    auto * reg = llamaedge_get_registry(ctx);
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
struct llama_kv_cache * llamaedge_get_kv_cache(struct llama_context * ctx) {
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
void llamaedge_get_model_hparams(struct llama_context * ctx,
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
size_t llamaedge_extract_kv_tensor_f32(const struct ggml_tensor * src,
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
size_t llamaedge_deserialize_kv_tensor(const uint8_t * src,
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

    auto * reg = llamaedge_get_registry(ctx);
    if (!reg->kv_export_hooks) {
        return 0;
    }

    // Get model hyperparameters
    int32_t n_layers = 1;
    int32_t n_kv_heads = 1;
    int32_t n_embd_head_k = 128;
    int32_t n_embd_head_v = 128;

    llamaedge_get_model_hparams(ctx, &n_layers, &n_kv_heads, &n_embd_head_k, &n_embd_head_v);

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
                llamaedge_deserialize_kv_tensor(layer_data->k_data, k_buf.data(),
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
                llamaedge_deserialize_kv_tensor(layer_data->v_data, v_buf.data(),
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

    auto * reg = llamaedge_get_registry(ctx);
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

