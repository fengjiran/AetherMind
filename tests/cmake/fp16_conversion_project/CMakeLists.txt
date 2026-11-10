cmake_minimum_required(VERSION 3.28)
project(Fp16ConversionContract LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# Reuse actual production sources and ISA/sanitizer configuration. No network or
# full inference build is needed to exercise independent conversion build modes.
include("${AETHERMIND_SOURCE_DIR}/cmake/AetherMindCpuIsa.cmake")
include("${AETHERMIND_SOURCE_DIR}/cmake/AetherMindSanitizers.cmake")
find_package(Threads REQUIRED)

add_library(fp16_conversion_sources STATIC
    "${AETHERMIND_SOURCE_DIR}/src/dtypes/half.cpp"
    "${AETHERMIND_SOURCE_DIR}/src/dtypes/fp16_conversion.cpp"
    "${AETHERMIND_SOURCE_DIR}/src/backend/cpu/fp16_conversion.cpp"
    "${AETHERMIND_SOURCE_DIR}/src/backend/cpu/cpu_capabilities.cpp"
    "${AETHERMIND_SOURCE_DIR}/src/backend/cpu/cpu_info.cpp")
target_include_directories(fp16_conversion_sources PUBLIC "${AETHERMIND_SOURCE_DIR}/include")
aethermind_apply_sanitizer(fp16_conversion_sources)
if(AETHERMIND_HAS_F16C)
    set(f16c_source "${AETHERMIND_SOURCE_DIR}/src/backend/cpu/fp16_conversion_f16c.cpp")
    target_sources(fp16_conversion_sources PRIVATE "${f16c_source}")
    aethermind_configure_f16c_sources("${f16c_source}")
    target_compile_definitions(fp16_conversion_sources PUBLIC FP16_HAS_F16C_CONVERSION=1)
endif()
if(AETHERMIND_HAS_AVX2)
    set(avx2_source "${AETHERMIND_SOURCE_DIR}/src/backend/cpu/fp16_conversion_avx2.cpp")
    target_sources(fp16_conversion_sources PRIVATE "${avx2_source}")
    aethermind_configure_avx2_integer_sources("${avx2_source}")
    target_compile_definitions(fp16_conversion_sources PUBLIC FP16_HAS_AVX2_CONVERSION=1)
endif()

add_library(contract_gtest STATIC IMPORTED)
set_target_properties(contract_gtest PROPERTIES
    IMPORTED_LOCATION "${AETHERMIND_GTEST_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${AETHERMIND_GTEST_INCLUDE_DIR}")
add_library(contract_gtest_main STATIC IMPORTED)
set_target_properties(contract_gtest_main PROPERTIES IMPORTED_LOCATION "${AETHERMIND_GTEST_MAIN_LIBRARY}")
add_executable(fp16_conversion_contract
    "${AETHERMIND_SOURCE_DIR}/tests/unit/dtypes/test_fp16_batch_conversion.cpp")
target_link_libraries(fp16_conversion_contract PRIVATE fp16_conversion_sources contract_gtest_main contract_gtest Threads::Threads)
aethermind_apply_sanitizer(fp16_conversion_contract)
enable_testing()
add_test(NAME fp16_conversion_contract COMMAND fp16_conversion_contract)
