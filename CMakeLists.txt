cmake_minimum_required(VERSION 3.28)
project(AetherMind
        VERSION 1.0.0
        DESCRIPTION "Inference Engine"
        LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckCXXSourceCompiles)
check_cxx_source_compiles([=[
    #include <format>
    int main() {
        constexpr std::format_string<int> fmt{"{}"};
        return std::format(fmt, 42) == "42" ? 0 : 1;
    }
]=] AETHERMIND_HAS_STD_FORMAT)
if (NOT AETHERMIND_HAS_STD_FORMAT)
    message(FATAL_ERROR "AetherMind requires C++20 std::format and std::format_string. Select a standard library with <format> support (e.g. libstdc++ 13.1 or newer).")
endif ()

include(CMakeDependentOption)
cmake_dependent_option(
        USE_LIBBACKTRACE "Enable libbacktrace" ON
        "NOT MSVC" OFF
)
option(BACKTRACE_ON_SEGFAULT "Set signal handler to print traceback on segfault" ON)
option(BUILD_TESTS "Adding test targets." ON)
option(BUILD_BENCHMARKS "Adding benchmark targets." ON)
option(BUILD_INFERENCE_TOOL "Build the standalone token-ID inference tool." OFF)
option(USE_ALLOC_ALIGNMENT "Enable alloc alignment" ON)
option(BUILD_DOCS "Build Doxygen documentation (requires doxygen)" OFF)
option(AETHERMIND_VALIDATION_REQUIRED_ASSETS "Require model fixtures and a CLI for CTest validation" OFF)

if (USE_ALLOC_ALIGNMENT)
    set(KALLOC_ALIGNMENT_DEFAULT 64 CACHE INTERNAL "Default alignment value")
    set(KALLOC_ALIGNMENT "${KALLOC_ALIGNMENT_DEFAULT}" CACHE STRING "Memory allocation alignment in bytes")

    include(cmake/AetherMindBuildChecks.cmake)
    aethermind_validate_alloc_alignment("${KALLOC_ALIGNMENT}")
endif ()

find_package(Threads REQUIRED)

include(FetchContent)
include(cmake/AetherMindSanitizers.cmake)
include(cmake/AetherMindCpuIsa.cmake)
include(cmake/AetherMindDependencySources.cmake)

if (TARGET spdlog::spdlog)
    message(STATUS "AetherMind spdlog: reusing provided spdlog::spdlog target")
else ()
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "Build shared spdlog library")
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "Build position-independent spdlog code")
    if (NOT DEFINED SPDLOG_USE_STD_FORMAT)
        if (SPDLOG_FMT_EXTERNAL OR SPDLOG_FMT_EXTERNAL_HO)
            set(SPDLOG_USE_STD_FORMAT OFF CACHE BOOL "Use std::format instead of fmt")
        else ()
            set(SPDLOG_USE_STD_FORMAT ON CACHE BOOL "Use std::format instead of fmt")
        endif ()
    endif ()
    aethermind_declare_shared_dependency(spdlog
            "${CMAKE_CURRENT_SOURCE_DIR}/.deps/spdlog"
            https://github.com/gabime/spdlog.git
            v1.17.0)
    aethermind_make_shared_dependency_available(spdlog TARGETS spdlog::spdlog)

    # A provider may supply an imported target instead of building this source.
    # Only extend usage requirements on the dependency we actually built.
    if (TARGET spdlog)
        get_target_property(_aethermind_spdlog_imported spdlog IMPORTED)
        get_target_property(_aethermind_spdlog_source spdlog SOURCE_DIR)
    endif ()

    if (TARGET spdlog AND NOT _aethermind_spdlog_imported AND
            _aethermind_spdlog_source STREQUAL spdlog_SOURCE_DIR)
        message(STATUS "AetherMind spdlog: SPDLOG_USE_STD_FORMAT=${SPDLOG_USE_STD_FORMAT}")
        if (SPDLOG_USE_STD_FORMAT)
            target_compile_features(spdlog PUBLIC cxx_std_20)
            target_compile_features(spdlog_header_only INTERFACE cxx_std_20)
        else ()
            message(STATUS "AetherMind spdlog: fmt backend selected; set SPDLOG_USE_STD_FORMAT=ON to use std::format")
        endif ()
    else ()
        message(STATUS "AetherMind spdlog: using dependency-provider target configuration")
    endif ()
endif ()

add_subdirectory(src)
if (BUILD_INFERENCE_TOOL)
    add_subdirectory(apps/inference)
endif ()

if (BUILD_TESTS)
    message(STATUS "Enable Testing")
    enable_testing()

    if (TARGET GTest::gtest AND TARGET GTest::gtest_main)
        message(STATUS "AetherMind GoogleTest: reusing provided GTest targets")
    elseif (TARGET GTest::gtest OR TARGET GTest::gtest_main)
        message(FATAL_ERROR "AetherMind tests require both GTest::gtest and GTest::gtest_main; only one target was provided")
    else ()
        set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock")
        set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest")
        set(gtest_build_tests OFF CACHE BOOL "Build GoogleTest's own tests")
        set(gtest_build_samples OFF CACHE BOOL "Build GoogleTest samples")

        aethermind_declare_shared_dependency(googletest
                "${CMAKE_CURRENT_SOURCE_DIR}/.deps/googletest"
                https://github.com/google/googletest.git
                v1.18.0)
        aethermind_make_shared_dependency_available(googletest TARGETS GTest::gtest GTest::gtest_main)

        if (googletest_SOURCE_DIR)
            message(STATUS "AetherMind GoogleTest: using source ${googletest_SOURCE_DIR}")
        else ()
            message(STATUS "AetherMind GoogleTest: using dependency-provider targets")
        endif ()
    endif ()

    if (NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
        message(FATAL_ERROR "GoogleTest did not provide GTest::gtest and GTest::gtest_main")
    endif ()

    add_subdirectory(tests/unit)
endif ()

if (BUILD_BENCHMARKS)
    message(STATUS "Enable Benchmarks")
    if (TARGET benchmark::benchmark AND TARGET benchmark::benchmark_main)
        message(STATUS "AetherMind Google Benchmark: reusing provided benchmark targets")
    elseif (TARGET benchmark::benchmark OR TARGET benchmark::benchmark_main)
        message(FATAL_ERROR "AetherMind benchmarks require both benchmark::benchmark and benchmark::benchmark_main; only one target was provided")
    else ()
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Build Google Benchmark tests that require GoogleTest")
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Build Google Benchmark's own tests")
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Install Google Benchmark")

        aethermind_declare_shared_dependency(google_benchmark
                "${CMAKE_CURRENT_SOURCE_DIR}/.deps/benchmark"
                https://github.com/google/benchmark.git
                v1.9.5)
        aethermind_make_shared_dependency_available(google_benchmark TARGETS benchmark::benchmark benchmark::benchmark_main)

        if (google_benchmark_SOURCE_DIR)
            message(STATUS "AetherMind Google Benchmark: using source ${google_benchmark_SOURCE_DIR}")
        else ()
            message(STATUS "AetherMind Google Benchmark: using dependency-provider targets")
        endif ()
    endif ()

    if (NOT TARGET benchmark::benchmark OR NOT TARGET benchmark::benchmark_main)
        message(FATAL_ERROR "Google Benchmark did not provide benchmark::benchmark and benchmark::benchmark_main")
    endif ()

    add_subdirectory(tests/benchmark)
endif ()

if (BUILD_DOCS)
    find_package(Doxygen REQUIRED dot)
    set(DOXYGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs")
    file(MAKE_DIRECTORY "${DOXYGEN_OUTPUT_DIR}")
    add_custom_target(aethermind_docs
            COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_SOURCE_DIR}/Doxyfile"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Generating Doxygen docs; warnings -> ${DOXYGEN_OUTPUT_DIR}/doxygen_warnings.log"
            VERBATIM
    )
    message(STATUS "Docs target enabled: cmake --build build --target aethermind_docs")
endif ()
