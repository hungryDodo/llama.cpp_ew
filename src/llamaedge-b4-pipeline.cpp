#include "llamaedge-internal.h"



// ============================================================
// B4: Pipeline State
// ============================================================


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
void llamaedge_profile_to_config(
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
    llamaedge_profile_to_config(profile, &pipeline->config.thinning, &pipeline->config.quant);

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
            size_t json_len = llamaedge_chunk_header_to_json(&hdr, json_hdr, sizeof(json_hdr));

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
    return llamaedge_chunk_header_to_json(hdr, (char *)dst, dst_size);
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
    return llamaedge_chunk_header_from_json(json_str, hdr_out);
}
