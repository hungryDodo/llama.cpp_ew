#include "llamaedge-internal.h"


// ============================================================
// B4: Quantization Implementation (Q3: K4/V3 + WHT)
// ============================================================

int32_t llamaedge_qrange(int32_t bits) {
    return (1 << (bits - 1)) - 1;
}

void llamaedge_walsh_hadamard_transform_inplace(float * data, uint32_t n) {
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
    uint32_t qmax = llamaedge_qrange(k_bits);
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
    uint32_t qmax = llamaedge_qrange(k_bits);
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
    uint32_t qmax = llamaedge_qrange(v_bits);

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
            llamaedge_walsh_hadamard_transform_inplace(row.data(), n_embd_head_v);
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
    uint32_t qmax = llamaedge_qrange(v_bits);

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
            llamaedge_walsh_hadamard_transform_inplace(row.data(), n_embd_head_v);
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
