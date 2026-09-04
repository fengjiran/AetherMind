file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        spdlog::spdlog
)

if (USE_LIBBACKTRACE)
    set(BACKTRACE_PATH "3rdparty/libbacktrace")
    include(${PROJECT_SOURCE_DIR}/cmake/AddLibbacktrace.cmake)
endif ()

if (USE_ALLOC_ALIGNMENT)
    message(STATUS "KALLOC_ALIGNMENT set to ${KALLOC_ALIGNMENT}")
    target_compile_definitions(${PROJECT_NAME} PRIVATE KALLOC_ALIGNMENT=${KALLOC_ALIGNMENT})
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

# AVX2+FMA flags for SIMD-optimized translation units. RMSNorm registers its
# AVX2+FMA descriptor only when this binary contains the corresponding
# translation unit implementation; reference RMSNorm remains available otherwise.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" CXX_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" CXX_SUPPORTS_FMA)
if (CXX_SUPPORTS_AVX2 AND CXX_SUPPORTS_FMA)
    set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/common/dot_product_f32_avx2.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/rmsnorm/rmsnorm_fp32_avx2.cpp
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE RMSNORM_HAS_AVX2_FMA_KERNEL=1)
    message(STATUS "AVX2/FMA enabled")
else ()
    message(STATUS "AVX2/FMA not supported by compiler")
endif ()
