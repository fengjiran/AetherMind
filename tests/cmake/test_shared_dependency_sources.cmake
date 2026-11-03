cmake_minimum_required(VERSION 3.28)

foreach(required AETHERMIND_SOURCE_DIR AETHERMIND_TEST_BINARY_DIR)
    if(NOT DEFINED ${required} OR NOT IS_ABSOLUTE "${${required}}")
        message(FATAL_ERROR "Pass an absolute ${required}")
    endif()
endforeach()
set(root "${AETHERMIND_TEST_BINARY_DIR}/shared dependency contract")
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/shared source" "${root}/override source")
set(dependency_project [=[
cmake_minimum_required(VERSION 3.28)
add_library(shared_probe_target INTERFACE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/variant.txt" "${CMAKE_BUILD_TYPE}")
]=])
file(WRITE "${root}/shared source/CMakeLists.txt" "${dependency_project}")
file(WRITE "${root}/override source/CMakeLists.txt" "${dependency_project}")
file(SHA256 "${root}/shared source/CMakeLists.txt" source_before)

function(configure_contract name build_type expected_source)
    execute_process(COMMAND "${CMAKE_COMMAND}"
            -S "${AETHERMIND_SOURCE_DIR}/tests/cmake/shared_dependency_contract_project"
            -B "${root}/${name}"
            "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
            "-DSHARED_SOURCE=${root}/shared source"
            "-DEXPECTED_SOURCE=${expected_source}"
            "-DCMAKE_BUILD_TYPE=${build_type}"
            ${ARGN}
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 45)
    file(WRITE "${root}/${name}.log" "${output}\n${error}")
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "Shared dependency ${name} failed (${result}):\n${output}\n${error}")
    endif()
    file(READ "${root}/${name}/dependency-binary.txt" binary_dir)
    file(READ "${binary_dir}/variant.txt" variant)
    if(NOT variant STREQUAL build_type)
        message(FATAL_ERROR "Shared dependency ${name} reused another build's configuration")
    endif()
endfunction()

configure_contract(debug Debug "${root}/shared source")
configure_contract(release Release "${root}/shared source")
configure_contract(override Debug "${root}/override source"
        "-DFETCHCONTENT_SOURCE_DIR_SHARED_PROBE=${root}/override source")
configure_contract(provider Debug "${root}/override source"
        "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${AETHERMIND_SOURCE_DIR}/tests/cmake/shared_dependency_contract_project/provider.cmake"
        "-DPROVIDER_SOURCE=${root}/override source")

# Reconfiguring one tree must leave the other tree's generated files intact.
configure_contract(debug RelWithDebInfo "${root}/shared source")
file(READ "${root}/release/dependency-binary.txt" release_binary)
file(READ "${release_binary}/variant.txt" release_variant)
if(NOT release_variant STREQUAL "Release")
    message(FATAL_ERROR "Independent build trees shared generated dependency files")
endif()
file(SHA256 "${root}/shared source/CMakeLists.txt" source_after)
if(NOT source_before STREQUAL source_after OR EXISTS "${root}/shared source/variant.txt")
    message(FATAL_ERROR "Dependency configuration modified shared sources")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}"
        -S "${AETHERMIND_SOURCE_DIR}/tests/cmake/shared_dependency_contract_project"
        -B "${root}/missing"
        "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
        "-DSHARED_SOURCE=${root}/absent source"
        -DCONTRACT_CASE=missing
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 45)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Missing shared-source fallback failed:\n${output}\n${error}")
endif()
message(STATUS "Shared sources, isolated build directories, overrides, providers and fallback passed")
