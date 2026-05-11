#include "llamaedge-internal.h"



// ============================================================
// B4: Profile Change Callbacks
// ============================================================


// Profile state stored per context

#ifdef LLAMAEDGE_ENABLE_HOOKS
static thread_local llamaedge_profile_state * g_profile_state = nullptr;

llamaedge_profile_state * llamaedge_get_profile_state(const struct llama_context * ctx) {
    (void)ctx;
    if (!g_profile_state) {
        g_profile_state = new llamaedge_profile_state();
    }
    return g_profile_state;
}
#else
inline llamaedge_profile_state * llamaedge_get_profile_state(const struct llama_context * ctx) {
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

    auto * state = llamaedge_get_profile_state(ctx);
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

    auto * state = llamaedge_get_profile_state(ctx);
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

    auto * state = llamaedge_get_profile_state(ctx);
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
    auto * state = llamaedge_get_profile_state(ctx);
    if (!state) {
        return LLAMAEDGE_PROFILE_BALANCED;
    }
    return state->current_profile;
}
