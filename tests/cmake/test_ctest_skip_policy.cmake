cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED AETHERMIND_TEST_BINARY_DIR OR NOT IS_ABSOLUTE "${AETHERMIND_TEST_BINARY_DIR}")
    message(FATAL_ERROR "Pass an absolute AETHERMIND_TEST_BINARY_DIR for contract artifacts")
endif()
if(NOT EXISTS "${AETHERMIND_REQUIRED_EXECUTION_PROBE}")
    message(FATAL_ERROR "Build aethermind_required_execution_probe before running the skip contract")
endif()
set(probe_dir "${AETHERMIND_TEST_BINARY_DIR}/ctest-skip-policy-contract")
file(MAKE_DIRECTORY "${probe_dir}")
set(required "PreparationPaths/TinyLlamaExecution.TinyGenerateMatchesScalarOracleAndStartsFreshEachCall/O1Plain")
set(brief "PreparationPaths/TinyLlamaExecution.TinyGenerateMatchesScalarOracleAndStartsFreshEachCall/O2Packed")
file(WRITE "${probe_dir}/CTestTestfile.cmake"
    "add_test(${required} \"${AETHERMIND_REQUIRED_EXECUTION_PROBE}\")\n"
    "add_test(${brief} \"${AETHERMIND_REQUIRED_EXECUTION_PROBE}\" --gtest_brief=1)\n"
    "add_test(OptionalProbe.Skipped \"${AETHERMIND_REQUIRED_EXECUTION_PROBE}\")\n"
    "set_tests_properties(${required} ${brief} OptionalProbe.Skipped PROPERTIES SKIP_REGULAR_EXPRESSION \"\\\\[  SKIPPED \\\\]\")\n"
    "set_tests_properties(OptionalProbe.Skipped PROPERTIES ENVIRONMENT AETHERMIND_REQUIRED_VALIDATION=0)\n"
    "set(probe_tests ${required} ${brief} OptionalProbe.Skipped)\n"
    "include(\"${AETHERMIND_SOURCE_DIR}/cmake/AetherMindTests.cmake\")\n"
    "aethermind_set_discovered_test_properties(probe_tests none TRUE TRUE TRUE TRUE)\n")

execute_process(COMMAND "${AETHERMIND_CTEST_COMMAND}" --test-dir "${probe_dir}"
    -R "^PreparationPaths/" --output-on-failure
    RESULT_VARIABLE required_result OUTPUT_VARIABLE required_output ERROR_VARIABLE required_error)
file(WRITE "${probe_dir}/required.txt" "${required_output}\n${required_error}")
if(required_result EQUAL 0 OR NOT required_output MATCHES "2 tests failed" OR
   NOT required_output MATCHES "Required validation case skipped")
    message(FATAL_ERROR "Required GTEST_SKIP was not a CTest failure: ${required_result}\n${required_output}\n${required_error}")
endif()
execute_process(COMMAND "${AETHERMIND_CTEST_COMMAND}" --test-dir "${probe_dir}"
    -R "^OptionalProbe" --output-on-failure
    RESULT_VARIABLE optional_result OUTPUT_VARIABLE optional_output ERROR_VARIABLE optional_error)
file(WRITE "${probe_dir}/optional.txt" "${optional_output}\n${optional_error}")
if(NOT optional_result EQUAL 0 OR NOT optional_output MATCHES "Skipped")
    message(FATAL_ERROR "Optional GTEST_SKIP lost its skipped status: ${optional_result}\n${optional_output}\n${optional_error}")
endif()
message(STATUS "Required skip fails with both printers; optional skip remains explicitly skipped")
