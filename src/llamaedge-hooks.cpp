// llamaedge hooks implementation
// Public entry points and glue code
// Internal implementations split into feature files

#include "llamaedge-internal.h"

#ifdef LLAMAEDGE_ENABLE_HOOKS
#include <mutex>
#include <unordered_map>
#endif

// ============================================================
// Hook Registry
// ============================================================

#ifdef LLAMAEDGE_ENABLE_HOOKS
namespace {

std::mutex g_registry_mu;
std::unordered_map<struct llama_context *, llamaedge_hook_registry *> g_registries;

template <typename Entry>
void llamaedge_free_entry_list(Entry *& head) {
    while (head) {
        Entry * next = head->next;
        delete head;
        head = next;
    }
}

} // namespace

llamaedge_hook_registry * llamaedge_get_registry(struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_registry_mu);
    auto & reg = g_registries[ctx];
    if (!reg) {
        reg = new llamaedge_hook_registry();
    }
    return reg;
}

void llamaedge_clear_registry(llamaedge_hook_registry * reg) {
    if (!reg) {
        return;
    }
    llamaedge_free_entry_list(reg->general_hooks);
    llamaedge_free_entry_list(reg->kv_export_hooks);
    llamaedge_free_entry_list(reg->kv_install_hooks);
    reg->active_hooks = 0;
}

void llamaedge_recompute_active_hooks(llamaedge_hook_registry * reg) {
    if (!reg) {
        return;
    }
    uint32_t active = 0;
    for (auto * entry = reg->general_hooks; entry; entry = entry->next) {
        active |= entry->hook_type;
    }
    if (reg->kv_export_hooks) {
        active |= LLAMAEDGE_HOOK_KV_EXPORT;
    }
    if (reg->kv_install_hooks) {
        active |= LLAMAEDGE_HOOK_KV_INSTALL;
    }
    reg->active_hooks = active;
}
#else
llamaedge_hook_registry * llamaedge_get_registry(struct llama_context * ctx) {
    (void)ctx;
    return nullptr;
}

void llamaedge_clear_registry(llamaedge_hook_registry * reg) {
    (void)reg;
}

void llamaedge_recompute_active_hooks(llamaedge_hook_registry * reg) {
    (void)reg;
}
#endif

// ============================================================
// Hook Management API
// ============================================================

LLAMA_API void llamaedge_hook_init(struct llama_context * ctx) {
    if (!ctx) return;
    auto * reg = llamaedge_get_registry(ctx);
    llamaedge_clear_registry(reg);
}

LLAMA_API void llamaedge_hook_cleanup(struct llama_context * ctx) {
    if (!ctx) return;
#ifdef LLAMAEDGE_ENABLE_HOOKS
    std::lock_guard<std::mutex> lock(g_registry_mu);
    auto it = g_registries.find(ctx);
    if (it == g_registries.end()) {
        return;
    }
    llamaedge_clear_registry(it->second);
    delete it->second;
    g_registries.erase(it);
#else
    (void)ctx;
#endif
}

LLAMA_API int llamaedge_hook_register(struct llama_context * ctx, const llamaedge_hook * hook) {
    if (!ctx || !hook || !hook->fn) return -1;
    auto * reg = llamaedge_get_registry(ctx);
    if (!reg) return -1;

    auto * entry = new llamaedge_hook();
    *entry = *hook;
    entry->next = reg->general_hooks;
    reg->general_hooks = entry;
    llamaedge_recompute_active_hooks(reg);
    return 0;
}

LLAMA_API int llamaedge_hook_unregister(struct llama_context * ctx, const llamaedge_hook * hook) {
    if (!ctx || !hook || !hook->fn) return -1;
    auto * reg = llamaedge_get_registry(ctx);
    if (!reg) return -1;

    auto ** prev = &reg->general_hooks;
    auto * curr = reg->general_hooks;
    while (curr) {
        if (curr->fn == hook->fn && curr->hook_type == hook->hook_type) {
            *prev = curr->next;
            delete curr;
            llamaedge_recompute_active_hooks(reg);
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
