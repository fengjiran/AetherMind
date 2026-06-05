set(TEST_TARGET aethermind_unit_tests)
option(BUILD_WITH_TORCH "build target with torch" OFF)

file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/tests/unit/*.cpp
)

find_package(GTest REQUIRED)
add_executable(${TEST_TARGET} ${TEST_SRC_FILES})

target_include_directories(${TEST_TARGET}
        PRIVATE
        #        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${TEST_TARGET} PRIVATE
        ${PROJECT_NAME}
        GTest::gtest_main
)

target_compile_definitions(${TEST_TARGET} PRIVATE
        AETHERMIND_TEST_MODELS_DIR="${PROJECT_SOURCE_DIR}/.models"
)

# SIMD learning tests (test_avx_learning.cpp): add -mavx2 -mfma so the
# intrinsics compile.  The test file uses #ifdef guards + GTEST_SKIP for
# portability, so this is safe even on non-x86 (the option is ignored by
# the #ifdef).  We check compiler support anyway for robustness.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" CXX_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" CXX_SUPPORTS_FMA)
if (CXX_SUPPORTS_AVX2 AND CXX_SUPPORTS_FMA)
    target_compile_options(${TEST_TARGET} PRIVATE -mavx2 -mfma)
    message(STATUS "AVX2/FMA enabled for unit tests")
else ()
    message(STATUS "AVX2/FMA not supported by compiler — SIMD learning tests will GTEST_SKIP")
endif ()

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
