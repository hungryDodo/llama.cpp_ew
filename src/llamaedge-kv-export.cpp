#include "llamaedge-internal.h"

// ============================================================
// B3 Streaming Full-KV Baseline: per-layer KV export hooks
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
    if (!reg) {
        return -1;
    }

    auto * entry = new llamaedge_kv_export_entry();
    entry->fn = fn;
    entry->user_data = user_data;
    entry->next = reg->kv_export_hooks;
    reg->kv_export_hooks = entry;
    llamaedge_recompute_active_hooks(reg);

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
    if (!reg) {
        return -1;
    }
    auto ** prev = &reg->kv_export_hooks;
    auto * curr = reg->kv_export_hooks;

    while (curr) {
        if (curr->fn == fn) {
            *prev = curr->next;
            delete curr;
            llamaedge_recompute_active_hooks(reg);
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
    if (!reg) {
        return -1;
    }

    auto * entry = new llamaedge_kv_install_entry();
    entry->fn = fn;
    entry->user_data = user_data;
    entry->next = reg->kv_install_hooks;
    reg->kv_install_hooks = entry;
    llamaedge_recompute_active_hooks(reg);

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
    if (!reg) {
        return -1;
    }
    auto ** prev = &reg->kv_install_hooks;
    auto * curr = reg->kv_install_hooks;

    while (curr) {
        if (curr->fn == fn) {
            *prev = curr->next;
            delete curr;
            llamaedge_recompute_active_hooks(reg);
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
// B3 Streaming Full-KV Baseline: per-layer KV export hooks - internal access
// ============================================================

namespace {

size_t llamaedge_kv_row_elems(uint32_t n_heads, uint32_t n_embd) {
    return static_cast<size_t>(n_heads) * static_cast<size_t>(n_embd);
}

bool llamaedge_supported_export_type(int32_t tensor_type) {
    return tensor_type == GGML_TYPE_F32 || tensor_type == GGML_TYPE_F16;
}

size_t llamaedge_expected_row_bytes(int32_t tensor_type, uint32_t n_heads, uint32_t n_embd) {
    const size_t elems = llamaedge_kv_row_elems(n_heads, n_embd);
    if (tensor_type == GGML_TYPE_F32) {
        return elems * sizeof(float);
    }
    if (tensor_type == GGML_TYPE_F16) {
        return elems * sizeof(ggml_fp16_t);
    }
    return 0;
}

} // namespace

void llamaedge_get_model_hparams(struct llama_context * ctx,
                                 uint32_t * n_layers_out,
                                 uint32_t * n_kv_heads_out,
                                 uint32_t * n_embd_head_k_out,
                                 uint32_t * n_embd_head_v_out) {
    if (!ctx || !n_layers_out || !n_kv_heads_out || !n_embd_head_k_out || !n_embd_head_v_out) {
        return;
    }

    const struct llama_model * model = llama_get_model(ctx);
    if (!model) {
        return;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_head = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    const int32_t n_embd = llama_model_n_embd(model);
    if (n_layer <= 0 || n_head <= 0 || n_head_kv <= 0 || n_embd <= 0) {
        return;
    }

    *n_layers_out = static_cast<uint32_t>(n_layer);
    *n_kv_heads_out = static_cast<uint32_t>(n_head_kv);
    *n_embd_head_k_out = static_cast<uint32_t>(n_embd / n_head);
    *n_embd_head_v_out = static_cast<uint32_t>(n_embd / n_head);
}

size_t llamaedge_extract_kv_tensor_f32(const struct ggml_tensor * src,
                                       float * dst,
                                       uint32_t n_tokens,
                                       uint32_t n_heads,
                                       uint32_t n_embd) {
    if (!src || !dst || n_tokens == 0 || n_heads == 0 || n_embd == 0) {
        return 0;
    }

    const int32_t tensor_type = static_cast<int32_t>(src->type);
    const size_t row_elems = llamaedge_kv_row_elems(n_heads, n_embd);
    const size_t row_bytes = ggml_row_size(src->type, row_elems);
    if (!llamaedge_supported_export_type(tensor_type) || row_bytes == 0) {
        return 0;
    }

    std::vector<uint8_t> row(row_bytes);
    size_t copied = 0;
    for (uint32_t tok = 0; tok < n_tokens; ++tok) {
        ggml_backend_tensor_get(src, row.data(), static_cast<size_t>(tok) * row_bytes, row_bytes);
        const size_t converted = llamaedge_deserialize_kv_tensor(
            row.data(), tensor_type, row_bytes, 1, n_heads, n_embd,
            dst + static_cast<size_t>(tok) * row_elems);
        if (converted == 0) {
            return copied;
        }
        copied += converted;
    }
    return copied;
}

size_t llamaedge_deserialize_kv_tensor(const uint8_t * src,
                                       int32_t tensor_type,
                                       uint64_t size_row,
                                       uint32_t n_tokens,
                                       uint32_t n_heads,
                                       uint32_t n_embd,
                                       float * dst) {
    if (!src || !dst || n_tokens == 0 || n_heads == 0 || n_embd == 0) {
        return 0;
    }

    const size_t row_elems = llamaedge_kv_row_elems(n_heads, n_embd);
    const size_t expected_row_bytes = llamaedge_expected_row_bytes(tensor_type, n_heads, n_embd);
    if (expected_row_bytes == 0 || size_row < expected_row_bytes) {
        return 0;
    }

    for (uint32_t tok = 0; tok < n_tokens; ++tok) {
        const uint8_t * row = src + static_cast<size_t>(tok) * static_cast<size_t>(size_row);
        float * out = dst + static_cast<size_t>(tok) * row_elems;
        if (tensor_type == GGML_TYPE_F32) {
            memcpy(out, row, expected_row_bytes);
        } else if (tensor_type == GGML_TYPE_F16) {
            const auto * fp16 = reinterpret_cast<const ggml_fp16_t *>(row);
            ggml_fp16_to_fp32_row(fp16, out, row_elems);
        } else {
            return 0;
        }
    }

    return static_cast<size_t>(n_tokens) * static_cast<size_t>(size_row);
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
    if (!reg || !reg->kv_export_hooks) {
        return 0;
    }

    uint32_t n_layers = 1;
    uint32_t n_kv_heads = 1;
    uint32_t n_embd_head_k = 128;
    uint32_t n_embd_head_v = 128;
    llamaedge_get_model_hparams(ctx, &n_layers, &n_kv_heads, &n_embd_head_k, &n_embd_head_v);
    if (n_layers == 0 || n_kv_heads == 0 || n_embd_head_k == 0 || n_embd_head_v == 0) {
        return 0;
    }

    const size_t state_size = llama_state_seq_get_size(ctx, seq_id);
    if (state_size == 0) {
        return 0;
    }
    std::vector<uint8_t> state_buffer(state_size);
    const size_t actual_size = llama_state_seq_get_data(ctx, state_buffer.data(), state_size, seq_id);
    if (actual_size == 0 || actual_size > state_size) {
        return 0;
    }

    llamaedge_kv_parsed_state * parsed = llamaedge_kv_parse_state(state_buffer.data(), actual_size);
    if (!parsed) {
        return 0;
    }

    const uint32_t available_tokens = parsed->n_tokens;
    const uint32_t actual_n_tokens = n_tokens > 0 ? std::min(n_tokens, available_tokens) : available_tokens;
    if (actual_n_tokens == 0) {
        llamaedge_kv_parse_state_free(parsed);
        return 0;
    }

    const size_t k_elements = static_cast<size_t>(actual_n_tokens) * n_kv_heads * n_embd_head_k;
    const size_t v_elements = static_cast<size_t>(actual_n_tokens) * n_kv_heads * n_embd_head_v;
    std::vector<float> k_buf(k_elements);
    std::vector<float> v_buf(v_elements);

    int exported = 0;
    const uint32_t layer_limit = std::min(n_layers, parsed->n_layer);
    for (uint32_t layer = 0; layer < layer_limit; ++layer) {
        llamaedge_kv_layer_data * layer_data = &parsed->layers[layer];
        std::fill(k_buf.begin(), k_buf.end(), 0.0f);
        std::fill(v_buf.begin(), v_buf.end(), 0.0f);

        const size_t k_bytes = llamaedge_deserialize_kv_tensor(
            layer_data->k_data, layer_data->k_type, layer_data->k_size_row,
            actual_n_tokens, n_kv_heads, n_embd_head_k, k_buf.data());
        size_t v_bytes = 0;
        if (!layer_data->v_trans) {
            v_bytes = llamaedge_deserialize_kv_tensor(
                layer_data->v_data, layer_data->v_type, layer_data->v_size_row,
                actual_n_tokens, n_kv_heads, n_embd_head_v, v_buf.data());
        }

        // Fail closed for unsupported/transposed rows; do not call hooks with
        // stale or all-zero buffers from a previous layer.
        if (k_bytes == 0 || v_bytes == 0) {
            continue;
        }

        for (auto * hook = reg->kv_export_hooks; hook; hook = hook->next) {
            hook->fn(ctx, hook->user_data, static_cast<int32_t>(layer), seq_id,
                     k_buf.data(), v_buf.data(), actual_n_tokens,
                     n_kv_heads, n_embd_head_k, n_embd_head_v);
            exported++;
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
    if (!reg) {
        return -1;
    }

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
