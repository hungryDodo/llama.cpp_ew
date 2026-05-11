#include "llamaedge-internal.h"

// ============================================================
// Pending Span Management (internal)
// ============================================================

llamaedge_pending_span_table * llamaedge_frontier_pending(llamaedge_frontier * frontier) {
    return reinterpret_cast<llamaedge_pending_span_table *>(frontier->pending_spans);
}

void llamaedge_frontier_add_pending(
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

void llamaedge_frontier_merge_pending(llamaedge_frontier * frontier, int32_t layer_id) {
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

