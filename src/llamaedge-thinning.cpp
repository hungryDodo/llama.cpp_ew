// Compatibility shim for the bounded llama.cpp fork.
// The implementation is owned by EdgeWeaver engine code; keep this
// wrapper small so llama.cpp remains limited to native KV/GGML hooks
// plus ABI compatibility.
#include "../../engine/bridge/llamaedge_compat/llamaedge_token_pruning_impl.cpp"
