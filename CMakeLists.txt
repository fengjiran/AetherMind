cmake_minimum_required(VERSION 3.28)
project(AetherMind
        VERSION 1.0.0
        DESCRIPTION "Inference Engine"
        LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Memory allocation alignment configuration
set(KALLOC_ALIGNMENT_DEFAULT 64 CACHE INTERNAL "Default alignment value")
set(KALLOC_ALIGNMENT "${KALLOC_ALIGNMENT_DEFAULT}" CACHE STRING "Memory allocation alignment in bytes")

set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)

option(BUILD_TESTS "Adding test targets." ON)
option(BUILD_BENCHMARKS "Adding benchmark targets." ON)
option(BUILD_INFERENCE_TOOL "Build the standalone token-ID inference tool." OFF)
option(USE_LIBBACKTRACE "Enable libbacktrace" ON)
option(BACKTRACE_ON_SEGFAULT "Set signal handler to print traceback on segfault" ON)
option(USE_ALLOC_ALIGNMENT "Enable alloc alignment" ON)
option(BUILD_DOCS "Build Doxygen documentation (requires doxygen)" OFF)
option(AETHERMIND_VALIDATION_REQUIRED_ASSETS "Require model fixtures and a CLI for CTest validation" OFF)

find_package(Threads REQUIRED)

include(cmake/AetherMindSanitizers.cmake)

add_compile_definitions(SPDLOG_USE_STD_FORMAT)
# add spdlog
include(FetchContent)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(spdlog)
if (BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            google_benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.5
    )
    FetchContent_MakeAvailable(google_benchmark)
endif ()

add_subdirectory(src)
if (BUILD_INFERENCE_TOOL)
    add_subdirectory(apps/inference)
endif ()

if (BUILD_TESTS)
    message(STATUS "Enable Testing")
    enable_testing()
    add_subdirectory(tests/unit)
endif ()

if (BUILD_BENCHMARKS)
    message(STATUS "Enable Benchmarks")
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
