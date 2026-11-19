file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)

set(AETHERMIND_ATTENTION_AVX2_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/attention/attention_avx2.cpp)
set(AETHERMIND_ATTENTION_KVF16_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/attention/attention_kvf16_avx2.cpp)
set(AETHERMIND_ADD_RMSNORM_AVX2_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/add_rmsnorm/add_rmsnorm_a32_w32_avx2.cpp)
set(AETHERMIND_SILU_MUL_AVX2_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/silu_mul/silu_mul_avx2.cpp)

set(AETHERMIND_AVX2_SOURCES
        ${AETHERMIND_ATTENTION_AVX2_SOURCE}
        ${AETHERMIND_ADD_RMSNORM_AVX2_SOURCE}
        ${AETHERMIND_SILU_MUL_AVX2_SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/common/dot_product_f32_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rmsnorm/rmsnorm_a32_w32_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_a32_w32_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_a32_wbf16_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_a32_w32_packed_b_avx2.cpp
)

if (NOT AETHERMIND_HAS_AVX2_FMA)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_AVX2_SOURCES})
endif ()

set(AETHERMIND_GEMM_A32_F16_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_a32_wf16_avx2.cpp)
if (NOT AETHERMIND_HAS_AVX2_FMA OR NOT AETHERMIND_HAS_F16C)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_GEMM_A32_F16_SOURCE} ${AETHERMIND_ATTENTION_KVF16_SOURCE})
endif ()

set(AETHERMIND_F16C_CONVERSION_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/fp16_conversion_f16c.cpp)
set(AETHERMIND_AVX2_CONVERSION_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/fp16_conversion_avx2.cpp)
set(AETHERMIND_AVX2_BF16_CONVERSION_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/bf16_conversion_avx2.cpp)

if (NOT AETHERMIND_HAS_F16C)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_F16C_CONVERSION_SOURCE})
endif ()

if (NOT AETHERMIND_HAS_AVX2)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_AVX2_CONVERSION_SOURCE} ${AETHERMIND_AVX2_BF16_CONVERSION_SOURCE})
endif ()

set(AETHERMIND_A16_AVX2_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_a16_w16_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rmsnorm/rmsnorm_a16_w32_avx2.cpp
)
set(AETHERMIND_A16_AVX512BF16_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_abf16_wbf16_avx512bf16.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rmsnorm/rmsnorm_a16_w32_avx512bf16.cpp
)
set(AETHERMIND_A16_AVX512FP16_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_af16_wf16_avx512fp16.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rmsnorm/rmsnorm_a16_w32_avx512fp16.cpp
)

if (NOT AETHERMIND_HAS_AVX2_FMA OR NOT AETHERMIND_HAS_F16C)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_A16_AVX2_SOURCES})
endif ()

if (NOT AETHERMIND_HAS_AVX512BF16)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_A16_AVX512BF16_SOURCES})
endif ()

if (NOT AETHERMIND_HAS_AVX512FP16)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_A16_AVX512FP16_SOURCES})
endif ()

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Reference kernels with an explicit rounded-F32 reduction contract (ascending
# K / head_dim / visible-token order). FMA contraction must stay off. The list
# spans the W16 (a32_w16), KV16 (a32_kv16) and A16 (a16_w16 / a16_w32 /
# a16_kv16) reference families.
set(AETHERMIND_STRICT_FP_REFERENCE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/linear/linear_a32_w16_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_a16_w16_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/embedding/embedding_a16_w16_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rmsnorm/rmsnorm_a16_w32_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rope/rope_a16_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/silu_mul/silu_mul_a16_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/kvcache_update/kvcache_update_a16_kv16_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/attention/attention_a32_kv16_reference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/attention/attention_a16_kv16_reference.cpp
)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_property(SOURCE ${AETHERMIND_STRICT_FP_REFERENCE_SOURCES} APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
elseif (MSVC)
    set_property(SOURCE ${AETHERMIND_STRICT_FP_REFERENCE_SOURCES} APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
endif ()

if (AETHERMIND_HAS_F16C)
    aethermind_configure_f16c_sources(${AETHERMIND_F16C_CONVERSION_SOURCE})
    target_compile_definitions(${PROJECT_NAME} PUBLIC FP16_HAS_F16C_CONVERSION=1)
endif ()

if (AETHERMIND_HAS_AVX2)
    aethermind_configure_avx2_integer_sources(${AETHERMIND_AVX2_CONVERSION_SOURCE}
                                              ${AETHERMIND_AVX2_BF16_CONVERSION_SOURCE})
    target_compile_definitions(${PROJECT_NAME} PUBLIC FP16_HAS_AVX2_CONVERSION=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC BF16_HAS_AVX2_CONVERSION=1)
endif ()
aethermind_apply_sanitizer(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        spdlog::spdlog
)

if (USE_LIBBACKTRACE)
    set(BACKTRACE_PATH "3rdparty/libbacktrace")
    include(${PROJECT_SOURCE_DIR}/cmake/AetherMindLibbacktrace.cmake)
endif ()

if (USE_ALLOC_ALIGNMENT)
    message(STATUS "KALLOC_ALIGNMENT set to ${KALLOC_ALIGNMENT}")
    target_compile_definitions(${PROJECT_NAME} PRIVATE KALLOC_ALIGNMENT=${KALLOC_ALIGNMENT}ULL)
endif ()

if (USE_LIBBACKTRACE)
    message(STATUS "Setting C++ macro USE_LIBBACKTRACE - 1")
    target_include_directories(${PROJECT_NAME} PRIVATE ${BACKTRACE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE libbacktrace)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_LIBBACKTRACE=1)
else ()
    message(STATUS "Setting C++ macro USE_LIBBACKTRACE - 0")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_LIBBACKTRACE=0)
endif (USE_LIBBACKTRACE)

if (BACKTRACE_ON_SEGFAULT)
    message(STATUS "Setting C++ macro BACKTRACE_ON_SEGFAULT - 1")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BACKTRACE_ON_SEGFAULT=1)
else ()
    message(STATUS "Setting C++ macro BACKTRACE_ON_SEGFAULT - 0")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BACKTRACE_ON_SEGFAULT=0)
endif (BACKTRACE_ON_SEGFAULT)

# Consumers use these definitions to know which optional symbols exist. Runtime
# eligibility still comes from CPU/OS detection and descriptor requirements.
# Keep the reference reduction order and IEEE special-value handling in the SIMD TUs.
set_property(SOURCE ${AETHERMIND_ATTENTION_AVX2_SOURCE} ${AETHERMIND_ATTENTION_KVF16_SOURCE}
                    ${AETHERMIND_ADD_RMSNORM_AVX2_SOURCE} ${AETHERMIND_SILU_MUL_AVX2_SOURCE}
             APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
if (AETHERMIND_HAS_AVX2_FMA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ATTENTION_HAS_AVX2_KERNEL=1)
    set_property(SOURCE ${AETHERMIND_ADD_RMSNORM_AVX2_SOURCE}
                        backend/cpu/kernels/add_rmsnorm/add_rmsnorm_entry.cpp
                 APPEND PROPERTY COMPILE_DEFINITIONS ADD_RMSNORM_HAS_AVX2_KERNEL=1)
    set_property(SOURCE ${AETHERMIND_SILU_MUL_AVX2_SOURCE}
                        backend/cpu/kernels/silu_mul/silu_mul_entry.cpp
                 APPEND PROPERTY COMPILE_DEFINITIONS SILU_MUL_HAS_AVX2_KERNEL=1)
    aethermind_configure_avx2_sources(${AETHERMIND_AVX2_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC RMSNORM_HAS_AVX2_FMA_KERNEL=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GEMM_HAS_AVX2_FMA_KERNEL=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOT_PRODUCT_HAS_AVX2_FMA_KERNEL=1)
endif ()

if (AETHERMIND_HAS_AVX2_FMA AND AETHERMIND_HAS_F16C)
    aethermind_configure_avx2_sources(${AETHERMIND_GEMM_A32_F16_SOURCE} ${AETHERMIND_ATTENTION_KVF16_SOURCE})
    aethermind_configure_f16c_sources(${AETHERMIND_GEMM_A32_F16_SOURCE} ${AETHERMIND_ATTENTION_KVF16_SOURCE})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ATTENTION_HAS_KVF16_AVX2_KERNEL=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GEMM_HAS_A32_F16_AVX2_FMA_KERNEL=1)
endif ()

# The A16 AVX2 fallback widens FP16 with F16C and BF16 with an exact shift, so
# it needs both toolchain capabilities.
if (AETHERMIND_HAS_AVX2_FMA AND AETHERMIND_HAS_F16C)
    aethermind_configure_avx2_sources(${AETHERMIND_A16_AVX2_SOURCES})
    aethermind_configure_f16c_sources(${AETHERMIND_A16_AVX2_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC A16_HAS_AVX2_FMA_KERNEL=1)
endif ()

if (AETHERMIND_HAS_AVX512BF16)
    aethermind_configure_avx512bf16_sources(${AETHERMIND_A16_AVX512BF16_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC A16_HAS_AVX512BF16_KERNEL=1)
endif ()

if (AETHERMIND_HAS_AVX512FP16)
    aethermind_configure_avx512fp16_sources(${AETHERMIND_A16_AVX512FP16_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC A16_HAS_AVX512FP16_KERNEL=1)
endif ()
