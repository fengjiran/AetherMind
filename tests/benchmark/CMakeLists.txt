set(BENCHMARK_TARGET aethermind_benchmark)
file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/tests/benchmark/*.cpp
)
# Model measurements have their own CLI and raw report contract.
# Keep that main out of the Google Benchmark executable.
list(FILTER TEST_SRC_FILES EXCLUDE REGEX "/models/")

add_executable(${BENCHMARK_TARGET} ${TEST_SRC_FILES})
aethermind_apply_sanitizer(${BENCHMARK_TARGET})

target_include_directories(${BENCHMARK_TARGET}
        PRIVATE
        #        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${BENCHMARK_TARGET} PRIVATE
        ${PROJECT_NAME}
        benchmark::benchmark_main
)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Model benchmark measurements currently require Linux")
    return()
endif ()

set(MODEL_BENCHMARK_TARGET aethermind_model_benchmark)
add_executable(${MODEL_BENCHMARK_TARGET} models/model_benchmark.cpp)
aethermind_apply_sanitizer(${MODEL_BENCHMARK_TARGET})
target_link_libraries(${MODEL_BENCHMARK_TARGET} PRIVATE ${PROJECT_NAME} spdlog::spdlog)

# Generate a private header rather than passing metadata through -D flags:
# quotes, backslashes and semicolons in compiler flags must remain literal.
set(_aethermind_model_bench_flags "${CMAKE_CXX_FLAGS}")
set(_aethermind_model_bench_configs ${CMAKE_CONFIGURATION_TYPES} "${CMAKE_BUILD_TYPE}")
list(REMOVE_DUPLICATES _aethermind_model_bench_configs)
foreach (_config IN LISTS _aethermind_model_bench_configs)
    if (NOT _config STREQUAL "")
        string(TOUPPER "${_config}" _config_upper)
        string(APPEND _aethermind_model_bench_flags
                " $<$<CONFIG:${_config}>:${CMAKE_CXX_FLAGS_${_config_upper}}>")
    endif ()
endforeach ()

set(_aethermind_model_bench_compiler "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
foreach (_field flags compiler)
    string(REPLACE "\\" "\\\\" _aethermind_model_bench_${_field}
            "${_aethermind_model_bench_${_field}}")
    string(REPLACE "\"" "\\\"" _aethermind_model_bench_${_field}
            "${_aethermind_model_bench_${_field}}")
    string(REPLACE "\n" "\\n" _aethermind_model_bench_${_field}
            "${_aethermind_model_bench_${_field}}")
    string(REPLACE "\r" "\\r" _aethermind_model_bench_${_field}
            "${_aethermind_model_bench_${_field}}")
endforeach ()

if (AETHERMIND_HAS_AVX2_FMA)
    set(_aethermind_model_bench_has_avx2_fma 1)
else ()
    set(_aethermind_model_bench_has_avx2_fma 0)
endif ()
set(_aethermind_model_bench_info
        "${CMAKE_CURRENT_BINARY_DIR}/model-build-info/$<CONFIG>/aethermind_model_benchmark_build_info.h")
file(GENERATE OUTPUT "${_aethermind_model_bench_info}" CONTENT
"#ifndef AETHERMIND_MODEL_BENCHMARK_BUILD_INFO_H
#define AETHERMIND_MODEL_BENCHMARK_BUILD_INFO_H
#define AETHERMIND_MODEL_BENCH_BUILD_TYPE \"$<CONFIG>\"
#define AETHERMIND_MODEL_BENCH_SANITIZER \"${AETHERMIND_SANITIZER}\"
#define AETHERMIND_MODEL_BENCH_COMPILER \"${_aethermind_model_bench_compiler}\"
#define AETHERMIND_MODEL_BENCH_CXX_FLAGS \"${_aethermind_model_bench_flags}\"
#define AETHERMIND_MODEL_BENCH_HAS_AVX2_FMA ${_aethermind_model_bench_has_avx2_fma}
#endif
")
if (MSVC)
    target_compile_options(${MODEL_BENCHMARK_TARGET} PRIVATE "/FI${_aethermind_model_bench_info}")
else ()
    target_compile_options(${MODEL_BENCHMARK_TARGET} PRIVATE -include "${_aethermind_model_bench_info}")
endif ()
