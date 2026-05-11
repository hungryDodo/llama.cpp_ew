#include "llamaedge-internal.h"


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

uint32_t llamaedge_select_token_indices_by_l2(
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

        return llamaedge_select_token_indices_by_l2(
            scores.data(), n_tokens,
            config->top_k, config->top_ratio,
            config->protected_token_count,
            selected_indices
        );
    }

    return 0;
}

