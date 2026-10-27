set(TEST_TARGET aethermind_unit_tests)
option(BUILD_WITH_TORCH "build target with torch" OFF)
include(${PROJECT_SOURCE_DIR}/cmake/AetherMindTests.cmake)
aethermind_configure_test_options()

file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/tests/unit/*.cpp
)
set(AETHERMIND_AVX2_TEST_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/avx_learning/test_avx_learning_avx2.cpp)
if(NOT AETHERMIND_HAS_AVX2_FMA)
    list(REMOVE_ITEM TEST_SRC_FILES "${AETHERMIND_AVX2_TEST_SOURCE}")
endif()

find_package(GTest REQUIRED)
add_executable(${TEST_TARGET} ${TEST_SRC_FILES})
aethermind_apply_sanitizer(${TEST_TARGET})

# Compile the frozen ABI contract as C11, not as a C++ translation unit. New
# runtime declarations are checked without providing placeholder definitions.
target_sources(${TEST_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/c_api/test_c_api_contract.c
        ${CMAKE_CURRENT_SOURCE_DIR}/c_api/test_c_api_runtime_consumer.c
)
set_target_properties(${TEST_TARGET} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
)

# Compile the same private application helpers that the future CLI will use.
# Keep application code out of the AetherMind shared library.
target_sources(${TEST_TARGET} PRIVATE
        ${PROJECT_SOURCE_DIR}/apps/inference/inference_runner.cpp
)

# The builder injection entry exists only in the runner's test compilation.
set_property(SOURCE
        ${PROJECT_SOURCE_DIR}/apps/inference/inference_runner.cpp
        APPEND PROPERTY COMPILE_DEFINITIONS AETHERMIND_INFERENCE_RUNNER_TESTING=1
)

# Export the test executable's glibc malloc wrappers so calls from AetherMind's
# shared library are interposed during the scoped steady-state allocation check.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_target_properties(${TEST_TARGET} PROPERTIES ENABLE_EXPORTS ON)
endif ()

target_include_directories(${TEST_TARGET}
        PRIVATE
        #        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/apps/inference
        ${PROJECT_SOURCE_DIR}/tests/unit
)

target_link_libraries(${TEST_TARGET} PRIVATE
        ${PROJECT_NAME}
        GTest::gtest_main
)

# This intentional skipped GTest is used only by the CMake negative contract.
add_executable(aethermind_required_execution_probe EXCLUDE_FROM_ALL
        ${PROJECT_SOURCE_DIR}/tests/cmake/test_required_execution_probe.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/test_utils/test_required_execution_guard.cpp
)
target_include_directories(aethermind_required_execution_probe PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit)
target_link_libraries(aethermind_required_execution_probe PRIVATE GTest::gtest_main)
aethermind_apply_sanitizer(aethermind_required_execution_probe)
add_dependencies(${TEST_TARGET} aethermind_required_execution_probe)

target_compile_definitions(${TEST_TARGET} PRIVATE
        AETHERMIND_TEST_MODELS_DIR="${AETHERMIND_TEST_MODELS_DIR}"
)

set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/inference/test_hf_fixed_trajectory.cpp
        APPEND PROPERTY COMPILE_DEFINITIONS
        "AETHERMIND_TEST_NUMERICAL_DIR=\"${AETHERMIND_TEST_NUMERICAL_DIR}\"")

# GTest registration and runtime guards remain baseline code. Only the helper
# that executes intrinsics receives ISA flags.
if(AETHERMIND_HAS_AVX2_FMA)
    aethermind_configure_avx2_sources("${AETHERMIND_AVX2_TEST_SOURCE}")
    target_compile_definitions(${TEST_TARGET} PRIVATE AETHERMIND_TEST_HAS_AVX2_FMA=1)
endif()

if (BUILD_WITH_TORCH)
    message(STATUS "Setting C++ macro BUILD_WITH_TORCH - 1")
    find_package(Torch REQUIRED)
    #    message(STATUS "Torch LIB path: " ${TORCH_LIBRARIES})
    #    message(STATUS "Torch include path: " ${TORCH_INCLUDE_DIRS})
    target_include_directories(${TEST_TARGET} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${TEST_TARGET} ${TORCH_LIBRARIES})
    target_compile_definitions(${TEST_TARGET} PRIVATE BUILD_WITH_TORCH=1)
else ()
    message(STATUS "Setting C++ macro BUILD_WITH_TORCH - 0")
    target_compile_definitions(${TEST_TARGET} PRIVATE BUILD_WITH_TORCH=0)
endif ()

aethermind_register_tests(${TEST_TARGET})
