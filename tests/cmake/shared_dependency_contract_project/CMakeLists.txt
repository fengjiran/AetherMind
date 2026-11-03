cmake_minimum_required(VERSION 3.28)
project(SharedDependencyContract NONE)

include(FetchContent)
include("${AETHERMIND_SOURCE_DIR}/cmake/AetherMindDependencySources.cmake")
aethermind_shared_dependency_source_options("${SHARED_SOURCE}" source_options)
if(CONTRACT_CASE STREQUAL "missing")
    if(source_options)
        message(FATAL_ERROR "Missing shared sources must preserve normal FetchContent acquisition")
    endif()
    return()
endif()

# Any accidental default acquisition must fail instead of contacting a server.
FetchContent_Declare(shared_probe
        GIT_REPOSITORY "${CMAKE_CURRENT_BINARY_DIR}/must-not-fetch"
        GIT_TAG must-not-checkout
        ${source_options})
FetchContent_MakeAvailable(shared_probe)
FetchContent_GetProperties(shared_probe SOURCE_DIR actual_source BINARY_DIR actual_binary)
file(REAL_PATH "${actual_source}" actual_source)
file(REAL_PATH "${EXPECTED_SOURCE}" expected_source)
if(NOT actual_source STREQUAL expected_source)
    message(FATAL_ERROR "Wrong shared source: ${actual_source}; expected ${expected_source}")
endif()
cmake_path(IS_PREFIX CMAKE_BINARY_DIR "${actual_binary}" NORMALIZE binary_is_local)
if(NOT binary_is_local)
    message(FATAL_ERROR "Dependency binary directory escaped its build tree: ${actual_binary}")
endif()
if(EXISTS "${CMAKE_BINARY_DIR}/_deps/shared_probe-src")
    message(FATAL_ERROR "Shared source was copied into the build tree")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dependency-binary.txt" "${actual_binary}")
