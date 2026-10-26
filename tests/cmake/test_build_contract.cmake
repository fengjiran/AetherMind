cmake_minimum_required(VERSION 3.28)

if(DEFINED AETHERMIND_CONTRACT_CONFIG)
    include("${AETHERMIND_CONTRACT_CONFIG}")
endif()
foreach(required AETHERMIND_SOURCE_DIR AETHERMIND_CONTRACT_BINARY_DIR AETHERMIND_CONTRACT_CASE)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Pass ${required} to the build contract")
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

set(case_dir "${AETHERMIND_CONTRACT_BINARY_DIR}/${AETHERMIND_CONTRACT_CASE}")
file(MAKE_DIRECTORY "${case_dir}")

function(run_contract_command log_name expected_success)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 180)
    file(WRITE "${case_dir}/${log_name}.log" "${output}\n${error}")
    if(expected_success)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "${log_name} failed (${result}); see ${case_dir}/${log_name}.log\n${output}\n${error}")
        endif()
    elseif(result EQUAL 0 OR NOT "${output}\n${error}" MATCHES "KALLOC_ALIGNMENT")
        message(FATAL_ERROR "${log_name} did not reject KALLOC_ALIGNMENT (${result})\n${output}\n${error}")
    endif()
endfunction()

if(AETHERMIND_CONTRACT_CASE STREQUAL "alignment")
    set(alignment_dir "${case_dir}/probe")
    set(alignment_command "${CMAKE_COMMAND}"
        -S "${AETHERMIND_SOURCE_DIR}/tests/cmake/alignment_contract_project"
        -B "${alignment_dir}" ${configure_options}
        "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}" -DUSE_ALLOC_ALIGNMENT=ON)
    foreach(value 1 8 64 4096)
        run_contract_command("valid-${value}" TRUE ${alignment_command} "-DKALLOC_ALIGNMENT=${value}")
    endforeach()
    file(READ "${alignment_dir}/size_t_bytes.txt" size_t_bytes)
    if(size_t_bytes EQUAL 8)
        set(largest_alignment 9223372036854775808)
    elseif(size_t_bytes EQUAL 4)
        set(largest_alignment 2147483648)
    else()
        message(FATAL_ERROR "The alignment contract needs a boundary for ${size_t_bytes}-byte size_t")
    endif()
    run_contract_command(largest-valid TRUE ${alignment_command} "-DKALLOC_ALIGNMENT=${largest_alignment}")
    set(index 0)
    foreach(value "" 0 -1 24 abc 08 8+8 1.5 18446744073709551616 999999999999999999999999999999999)
        math(EXPR index "${index} + 1")
        run_contract_command("invalid-${index}" FALSE ${alignment_command} "-DKALLOC_ALIGNMENT=${value}")
    endforeach()
    # Reusing one build directory must not retain a successful cached check.
    run_contract_command(recheck-valid TRUE ${alignment_command} -DKALLOC_ALIGNMENT=64)
    run_contract_command(recheck-invalid FALSE ${alignment_command} -DKALLOC_ALIGNMENT=24)
    run_contract_command(disabled TRUE ${alignment_command} -DUSE_ALLOC_ALIGNMENT=OFF -DKALLOC_ALIGNMENT=abc)
    message(STATUS "Alignment configuration boundaries and cache changes passed")
    return()
endif()

if(NOT EXISTS "${AETHERMIND_SPDLOG_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Build contracts require an existing spdlog source checkout; no downloads are performed")
endif()
if(NOT AETHERMIND_CONTRACT_CASE MATCHES "^(std|fmt|existing-std|existing-fmt)$")
    message(FATAL_ERROR "Unknown build contract case: ${AETHERMIND_CONTRACT_CASE}")
endif()

set(dependency_options)
if(AETHERMIND_CONTRACT_CASE MATCHES "fmt$")
    list(APPEND dependency_options -DSPDLOG_USE_STD_FORMAT=OFF -DAETHERMIND_EXPECT_STD_FORMAT=OFF)
else()
    # The ordinary std case deliberately tests a fresh configuration's default.
    list(APPEND dependency_options -DAETHERMIND_EXPECT_STD_FORMAT=ON)
    if(AETHERMIND_CONTRACT_CASE MATCHES "^existing-")
        list(APPEND dependency_options -DSPDLOG_USE_STD_FORMAT=ON)
    else()
        list(APPEND dependency_options -USPDLOG_USE_STD_FORMAT)
    endif()
endif()
if(AETHERMIND_CONTRACT_CASE MATCHES "^existing-")
    list(APPEND dependency_options -DAETHERMIND_EXISTING_SPDLOG=ON)
endif()

set(project_dir "${case_dir}/project")
set(project_command "${CMAKE_COMMAND}" -S "${AETHERMIND_SOURCE_DIR}/tests/cmake/build_contract_project"
    -B "${project_dir}" ${configure_options}
    "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
    "-DAETHERMIND_SPDLOG_SOURCE_DIR=${AETHERMIND_SPDLOG_SOURCE_DIR}"
    -DUSE_ALLOC_ALIGNMENT=ON ${dependency_options})
run_contract_command(configure TRUE ${project_command} -DKALLOC_ALIGNMENT=64)

set(build_options --parallel 2)
if(AETHERMIND_BUILD_CONFIG)
    list(APPEND build_options --config "${AETHERMIND_BUILD_CONFIG}")
endif()
run_contract_command(logger-build TRUE "${CMAKE_COMMAND}" --build "${project_dir}"
    --target contract_logger ${build_options})
file(READ "${project_dir}/logger-location-${AETHERMIND_BUILD_CONFIG}.txt" logger_binary)
run_contract_command(logger-run TRUE "${logger_binary}")
file(READ "${case_dir}/logger-run.log" logger_output)
if(NOT logger_output MATCHES "AetherMind dependency contract 42")
    message(FATAL_ERROR "The dependency consumer did not produce the expected formatted message")
endif()

set(consumer_dir "${case_dir}/consumer")
run_contract_command(consumer-configure TRUE "${CMAKE_COMMAND}"
    -S "${AETHERMIND_SOURCE_DIR}/tests/cmake/build_contract_consumer" -B "${consumer_dir}"
    ${configure_options} "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
    "-DAETHERMIND_CONTRACT_TARGETS=${project_dir}/AetherMindContractTargets.cmake")
run_contract_command(consumer-build TRUE "${CMAKE_COMMAND}" --build "${consumer_dir}"
    --target contract_cpp_consumer contract_c_consumer ${build_options})

if(AETHERMIND_CONTRACT_CASE STREQUAL "std")
    # Exercise the real project's option gate as well as the isolated helper.
    run_contract_command(project-invalid FALSE ${project_command} -DKALLOC_ALIGNMENT=24)
    run_contract_command(project-disabled TRUE ${project_command} -DKALLOC_ALIGNMENT=24 -DUSE_ALLOC_ALIGNMENT=OFF)
endif()
message(STATUS "${AETHERMIND_CONTRACT_CASE}: C++/C consumers and logging dependency passed")
