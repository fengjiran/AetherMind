file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)

set(AETHERMIND_AVX2_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/common/dot_product_f32_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rmsnorm/rmsnorm_f32_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_f32_avx2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_packed_b_avx2.cpp
)
if(NOT AETHERMIND_HAS_AVX2_FMA)
    list(REMOVE_ITEM SRC_FILES ${AETHERMIND_AVX2_SOURCES})
endif()

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
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
if(AETHERMIND_HAS_AVX2_FMA)
    aethermind_configure_avx2_sources(${AETHERMIND_AVX2_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC RMSNORM_HAS_AVX2_FMA_KERNEL=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GEMM_HAS_AVX2_FMA_KERNEL=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOT_PRODUCT_HAS_AVX2_FMA_KERNEL=1)
endif()
