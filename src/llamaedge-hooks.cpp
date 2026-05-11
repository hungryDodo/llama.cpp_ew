// llamaedge hooks implementation
// Public entry points and glue code
// Internal implementations split into feature files

#include "llamaedge-internal.h"

// ============================================================
// Hook Registry
// ============================================================

#ifdef LLAMAEDGE_ENABLE_HOOKS
static thread_local llamaedge_hook_registry * g_registry = nullptr;

llamaedge_hook_registry * llamaedge_get_registry(struct llama_context * ctx) {
    (void)ctx;
    if (!g_registry) {
        g_registry = new llamaedge_hook_registry();
    }
    return g_registry;
}
#else
static inline llamaedge_hook_registry * llamaedge_get_registry(struct llama_context * ctx) {
    (void)ctx;
    return nullptr;
}
#endif

// ============================================================
// Hook Management API
// ============================================================

LLAMA_API void llamaedge_hook_init(struct llama_context * ctx) {
    if (!ctx) return;
    auto * reg = llamaedge_get_registry(ctx);
    if (reg) {
        reg->general_hooks = nullptr;
        reg->kv_export_hooks = nullptr;
        reg->kv_install_hooks = nullptr;
        reg->active_hooks = 0;
    }
}

LLAMA_API int llamaedge_hook_register(struct llama_context * ctx, const llamaedge_hook * hook) {
    if (!ctx || !hook) return -1;
    auto * reg = llamaedge_get_registry(ctx);
    if (!reg) return -1;

    auto * entry = new llamaedge_hook();
    *entry = *hook;
    entry->next = reg->general_hooks;
    reg->general_hooks = entry;
    reg->active_hooks |= hook->hook_type;
    return 0;
}

LLAMA_API int llamaedge_hook_unregister(struct llama_context * ctx, const llamaedge_hook * hook) {
    if (!ctx || !hook) return -1;
    auto * reg = llamaedge_get_registry(ctx);
    if (!reg) return -1;

    auto ** prev = &reg->general_hooks;
    auto * curr = reg->general_hooks;
    while (curr) {
        if (curr->fn == hook->fn && curr->hook_type == hook->hook_type) {
            *prev = curr->next;
            delete curr;
            return 0;
        }
        prev = &curr->next;
        curr = curr->next;
    }
    return -1;
}

LLAMA_API bool llamaedge_hook_is_active(struct llama_context * ctx, uint32_t hook_type) {
    auto * reg = llamaedge_get_registry(ctx);
    if (!reg) return false;
    return (reg->active_hooks & hook_type) != 0;
}
