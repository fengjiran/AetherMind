cmake_minimum_required(VERSION 3.28)

foreach(required AETHERMIND_SOURCE_DIR AETHERMIND_TEST_BINARY_DIR AETHERMIND_TEST_PYTHON_EXECUTABLE)
    if(NOT DEFINED ${required} OR NOT IS_ABSOLUTE "${${required}}")
        message(FATAL_ERROR "Pass an absolute ${required} to the Python discovery contract")
    endif()
endforeach()
if(NOT EXISTS "${AETHERMIND_TEST_PYTHON_EXECUTABLE}")
    message(FATAL_ERROR "Python discovery contracts require an existing Python interpreter")
endif()

set(contract_dir "${AETHERMIND_TEST_BINARY_DIR}/python-discovery-contract")
file(MAKE_DIRECTORY "${contract_dir}")

function(configure_contract case stage expected_error)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        --unset=VIRTUAL_ENV --unset=UV_PROJECT_ENVIRONMENT --unset=CONDA_PREFIX
        "${CMAKE_COMMAND}"
        -S "${AETHERMIND_SOURCE_DIR}/tests/cmake/python_discovery_project"
        -B "${contract_dir}/${case}"
        "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
        "-DAETHERMIND_TEST_PYTHON_EXECUTABLE=${AETHERMIND_TEST_PYTHON_EXECUTABLE}"
        "-DAETHERMIND_PYTHON_CONTRACT_CASE=${case}"
        "-DAETHERMIND_PYTHON_CONTRACT_STAGE=${stage}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)
    set(log "${contract_dir}/${case}-${stage}.log")
    file(WRITE "${log}" "${output}\n${error}")
    if("${expected_error}" STREQUAL "")
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "${case}/${stage} failed (${result}); see ${log}\n${output}\n${error}")
        endif()
    elseif(result EQUAL 0 OR NOT "${output}\n${error}" MATCHES "${expected_error}")
        message(FATAL_ERROR "${case}/${stage} did not reject the expected condition; see ${log}\n${output}\n${error}")
    endif()
endfunction()

foreach(case notfound disabled disabled-required valid-pin invalid-pin)
    # The first configure must exercise fresh discovery on every test run.
    file(REMOVE_RECURSE "${contract_dir}/${case}")
endforeach()

configure_contract(notfound missing "")
configure_contract(notfound recovered "")
configure_contract(disabled optional "")
configure_contract(disabled-required required "CMAKE_DISABLE_FIND_PACKAGE_Python3")
configure_contract(valid-pin pinned "")
configure_contract(invalid-pin relative "Pinned Python3_EXECUTABLE must be an absolute path")
configure_contract(invalid-pin missing "Pinned Python3_EXECUTABLE.*not a runnable Python interpreter")

message(STATUS "Python discovery: NOTFOUND recovery, disabled results, REQUIRED rejection, and explicit pins passed")
