cmake_minimum_required(VERSION 3.28)

if(DEFINED AETHERMIND_CONTRACT_CONFIG)
    include("${AETHERMIND_CONTRACT_CONFIG}")
endif()
foreach(required AETHERMIND_SOURCE_DIR AETHERMIND_CONTRACT_BINARY_DIR AETHERMIND_CONTRACT_CASE
        AETHERMIND_SPDLOG_SOURCE_DIR AETHERMIND_GOOGLETEST_SOURCE_DIR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Pass ${required} to the GoogleTest dependency contract")
    endif()
endforeach()
if(NOT IS_ABSOLUTE "${AETHERMIND_CONTRACT_BINARY_DIR}" OR
   NOT AETHERMIND_CONTRACT_CASE MATCHES "^(offline|existing|options|partial-gtest|partial-gtest_main|disabled)$")
    message(FATAL_ERROR "Pass an absolute contract binary directory and a known GoogleTest contract case")
endif()
foreach(dependency SPDLOG GOOGLETEST)
    if(NOT EXISTS "${AETHERMIND_${dependency}_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "GoogleTest contracts require an existing ${dependency} source checkout; no downloads are performed")
    endif()
endforeach()

set(configure_options)
if(AETHERMIND_GENERATOR)
    list(APPEND configure_options -G "${AETHERMIND_GENERATOR}")
endif()
if(AETHERMIND_GENERATOR_PLATFORM)
    list(APPEND configure_options -A "${AETHERMIND_GENERATOR_PLATFORM}")
endif()
if(AETHERMIND_GENERATOR_TOOLSET)
    list(APPEND configure_options -T "${AETHERMIND_GENERATOR_TOOLSET}")
endif()
foreach(variable CMAKE_MAKE_PROGRAM CMAKE_TOOLCHAIN_FILE CMAKE_C_COMPILER CMAKE_CXX_COMPILER
        CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_SYSROOT CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET)
    if(DEFINED AETHERMIND_${variable} AND NOT "${AETHERMIND_${variable}}" STREQUAL "")
        list(APPEND configure_options "-D${variable}=${AETHERMIND_${variable}}")
    endif()
endforeach()
list(APPEND configure_options "-DCMAKE_BUILD_TYPE=${AETHERMIND_BUILD_CONFIG}")

set(case_dir "${AETHERMIND_CONTRACT_BINARY_DIR}/googletest-${AETHERMIND_CONTRACT_CASE}")
# Fresh caches ensure the defaults and disabled-fetch cases are exercised each run.
file(REMOVE_RECURSE "${case_dir}")
file(MAKE_DIRECTORY "${case_dir}")
function(run_googletest_contract log_name expected_error)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 180)
    file(WRITE "${case_dir}/${log_name}.log" "${output}\n${error}")
    # CMake wraps diagnostics according to the length of the source/build paths.
    string(REGEX REPLACE "[ \t\r\n]+" " " diagnostic "${output}\n${error}")
    if("${expected_error}" STREQUAL "")
        if(NOT "${result}" STREQUAL "0")
            message(FATAL_ERROR "${log_name} failed (${result}); see ${case_dir}/${log_name}.log\n${output}\n${error}")
        endif()
    elseif("${result}" STREQUAL "0" OR NOT "${diagnostic}" MATCHES "${expected_error}")
        message(FATAL_ERROR "${log_name} did not fail as expected (${result}); see ${case_dir}/${log_name}.log\n${output}\n${error}")
    endif()
endfunction()

set(expected_error "")
if(AETHERMIND_CONTRACT_CASE MATCHES "^partial-")
    set(expected_error "only one target was provided")
endif()
run_googletest_contract(configure "${expected_error}" "${CMAKE_COMMAND}"
    -S "${AETHERMIND_SOURCE_DIR}/tests/cmake/googletest_contract_project"
    -B "${case_dir}/project" ${configure_options}
    "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
    "-DAETHERMIND_SPDLOG_SOURCE_DIR=${AETHERMIND_SPDLOG_SOURCE_DIR}"
    "-DAETHERMIND_GOOGLETEST_SOURCE_DIR=${AETHERMIND_GOOGLETEST_SOURCE_DIR}"
    "-DAETHERMIND_CONTRACT_CASE=${AETHERMIND_CONTRACT_CASE}")
if(AETHERMIND_CONTRACT_CASE STREQUAL "disabled" OR AETHERMIND_CONTRACT_CASE MATCHES "^partial-")
    return()
endif()

set(build_options --parallel 2)
if(AETHERMIND_BUILD_CONFIG)
    list(APPEND build_options --config "${AETHERMIND_BUILD_CONFIG}")
endif()
run_googletest_contract(build "" "${CMAKE_COMMAND}" --build "${case_dir}/project"
    --target googletest_contract_probe ${build_options})
file(READ "${case_dir}/project/probe-location-${AETHERMIND_BUILD_CONFIG}.txt" probe)
run_googletest_contract(optional-skip "" "${CMAKE_COMMAND}" -E env AETHERMIND_REQUIRED_VALIDATION=0 "${probe}")
run_googletest_contract(required-skip "Required validation case skipped"
    "${CMAKE_COMMAND}" -E env AETHERMIND_REQUIRED_VALIDATION=1 "${probe}")
message(STATUS "GoogleTest ${AETHERMIND_CONTRACT_CASE}: public target linkage and required-execution guard passed")
