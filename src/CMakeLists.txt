llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            llamaedge-hooks.cpp
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_include_directories(llama PUBLIC ../include/llamaedge)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

# EdgeWeaver hook support
option(LLAMAEDGE_ENABLE_HOOKS "llama: enable EdgeWeaver PD hooks" OFF)
if(LLAMAEDGE_ENABLE_HOOKS)
    target_compile_definitions(llama PUBLIC LLAMAEDGE_ENABLE_HOOKS)
endif()

# EdgeWeaver G2 per-layer KV export helper
option(LLAMAEDGE_ENABLE_KV_LAYER_EXPORT_G2 "llama: enable G2 per-layer KV export helper" OFF)
if(LLAMAEDGE_ENABLE_KV_LAYER_EXPORT_G2)
    target_compile_definitions(llama PUBLIC LLAMAEDGE_ENABLE_KV_LAYER_EXPORT_G2)
endif()

# EdgeWeaver G3 per-layer KV import helper
option(LLAMAEDGE_ENABLE_KV_LAYER_IMPORT_G3 "llama: enable G3 per-layer KV import helper" OFF)
if(LLAMAEDGE_ENABLE_KV_LAYER_IMPORT_G3)
    target_compile_definitions(llama PUBLIC LLAMAEDGE_ENABLE_KV_LAYER_IMPORT_G3)
endif()

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
