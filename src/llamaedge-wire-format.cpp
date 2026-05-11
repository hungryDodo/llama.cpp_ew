#include "llamaedge-internal.h"


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
const uint8_t * llamaedge_read_u32(const uint8_t * src, size_t size, size_t * offset, uint32_t * out) {
    if (*offset + sizeof(uint32_t) > size) return nullptr;
    memcpy(out, src + *offset, sizeof(uint32_t));
    *offset += sizeof(uint32_t);
    return src + *offset;
}

// Helper: read uint64_t from buffer
const uint8_t * llamaedge_read_u64(const uint8_t * src, size_t size, size_t * offset, uint64_t * out) {
    if (*offset + sizeof(uint64_t) > size) return nullptr;
    memcpy(out, src + *offset, sizeof(uint64_t));
    *offset += sizeof(uint64_t);
    return src + *offset;
}

// Helper: read int32_t from buffer
const uint8_t * llamaedge_read_i32(const uint8_t * src, size_t size, size_t * offset, int32_t * out) {
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
    if (!llamaedge_read_u32(serialized, size, &offset, &v_trans)) return nullptr;
    if (!llamaedge_read_u32(serialized, size, &offset, &n_layer)) return nullptr;
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
        if (!llamaedge_read_i32(serialized, size, &offset, &k_type_i)) break;
        k_info[il].row_size = 0;
        if (!llamaedge_read_u64(serialized, size, &offset, &k_info[il].row_size)) break;

        k_info[il].data = serialized + offset;
        k_info[il].data_size = 0; // will be computed after we know n_tokens
    }

    // Parse V layer headers (don't skip data yet)
    for (uint32_t il = 0; il < n_layer; il++) {
        int32_t v_type_i;
        if (!llamaedge_read_i32(serialized, size, &offset, &v_type_i)) break;
        v_info[il].data = serialized + offset;
        v_info[il].row_size = 0;

        if (!result->v_trans) {
            if (!llamaedge_read_u64(serialized, size, &offset, &v_info[il].row_size)) break;
            v_info[il].data_size = 0;
        } else {
            uint32_t v_size_el;
            uint32_t n_embd_v_gqa;
            if (!llamaedge_read_u32(serialized, size, &offset, &v_size_el)) break;
            if (!llamaedge_read_u32(serialized, size, &offset, &n_embd_v_gqa)) break;

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


// ============================================================
// B4: Wire Format for Chunk Transmission
// ============================================================

// Compute size needed for a chunk header in JSON format
size_t llamaedge_chunk_header_json_size(const llamaedge_chunk_header * hdr) {
    (void)hdr;
    // Approximate JSON size
    return 256;
}

// Serialize chunk header to JSON
// Returns bytes written, 0 on error
size_t llamaedge_chunk_header_to_json(
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
bool llamaedge_chunk_header_from_json(
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
