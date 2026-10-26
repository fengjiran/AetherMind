cmake_minimum_required(VERSION 3.28)
project(AetherMindAlignmentContract LANGUAGES CXX)

include("${AETHERMIND_SOURCE_DIR}/cmake/AetherMindBuildChecks.cmake")
if(USE_ALLOC_ALIGNMENT)
    aethermind_validate_alloc_alignment("${KALLOC_ALIGNMENT}")
endif()

include(CheckTypeSize)
set(CMAKE_EXTRA_INCLUDE_FILES cstddef)
check_type_size("std::size_t" contract_size_t_bytes LANGUAGE CXX)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/size_t_bytes.txt" "${contract_size_t_bytes}")
