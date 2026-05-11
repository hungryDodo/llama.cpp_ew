#include "llamaedge-internal.h"


// ============================================================
// B3: TX Ring Implementation
// ============================================================

LLAMA_API llamaedge_tx_ring * llamaedge_tx_ring_create(
    const llamaedge_tx_ring_config * config
) {
    if (!config || config->ring_size == 0 || config->n_destinations == 0) {
        return nullptr;
    }

    llamaedge_tx_ring * ring = new llamaedge_tx_ring();
    ring->ring_size = config->ring_size;
    ring->payload_pool_bytes = config->payload_pool_bytes;
    ring->n_destinations = config->n_destinations;
    ring->payload_pool.resize(config->payload_pool_bytes);

    ring->destinations.resize(config->n_destinations);
    for (uint32_t i = 0; i < config->n_destinations; ++i) {
        ring->destinations[i].descriptors.resize(config->ring_size);
        ring->destinations[i].in_use.resize(config->ring_size, 0);
        ring->destinations[i].sync = new llamaedge_tx_ring_sync();
    }

    return ring;
}

LLAMA_API void llamaedge_tx_ring_destroy(llamaedge_tx_ring * ring) {
    if (ring) {
        // Clean up sync objects
        for (uint32_t i = 0; i < ring->n_destinations; ++i) {
            delete ring->destinations[i].sync;
        }
        delete ring;
    }
}

LLAMA_API int llamaedge_tx_ring_push(
    llamaedge_tx_ring * ring,
    uint32_t destination,
    const llamaedge_tx_desc * desc
) {
    if (!ring || !desc || destination >= ring->n_destinations) {
        return -1;
    }

    auto & dest = ring->destinations[destination];
    std::lock_guard<std::mutex> lock(dest.sync->mutex);

    if (dest.count >= ring->ring_size) {
        return -1;  // ring full
    }

    uint32_t idx = dest.tail % ring->ring_size;
    dest.descriptors[idx] = *desc;
    dest.descriptors[idx].in_use = true;
    dest.tail++;
    dest.count++;

    return 0;
}

LLAMA_API int llamaedge_tx_ring_pop(
    llamaedge_tx_ring * ring,
    uint32_t destination,
    llamaedge_tx_desc * desc
) {
    if (!ring || !desc || destination >= ring->n_destinations) {
        return -1;
    }

    auto & dest = ring->destinations[destination];
    std::lock_guard<std::mutex> lock(dest.sync->mutex);

    if (dest.count == 0) {
        return -1;  // ring empty
    }

    uint32_t idx = dest.head % ring->ring_size;
    *desc = dest.descriptors[idx];
    dest.descriptors[idx].in_use = false;
    dest.head++;
    dest.count--;

    return 0;
}

LLAMA_API uint32_t llamaedge_tx_ring_pending(
    const llamaedge_tx_ring * ring,
    uint32_t destination
) {
    if (!ring || destination >= ring->n_destinations) {
        return 0;
    }
    return ring->destinations[destination].count;
}

LLAMA_API uint8_t * llamaedge_payload_pool_alloc(
    llamaedge_tx_ring * ring,
    size_t size
) {
    if (!ring) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(ring->pool_mutex);

    // Simple first-fit allocation
    // In production, should use a better allocator
    static size_t pool_offset = 0;

    if (pool_offset + size > ring->payload_pool_bytes) {
        pool_offset = 0;  // wrap around - should track allocations properly
    }

    uint8_t * result = ring->payload_pool.data() + pool_offset;
    pool_offset += size;

    return result;
}

LLAMA_API void llamaedge_payload_pool_free(
    llamaedge_tx_ring * ring,
    uint8_t * payload
) {
    (void)ring;
    (void)payload;
    // Shared pool - free is a no-op for now
    // In production, should track and reclaim
}

