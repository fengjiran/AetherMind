set(TEST_TARGET aethermind_benchmark_tests)
file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/tests/benchmark/*.cpp
)

add_executable(${TEST_TARGET} ${TEST_SRC_FILES})

target_include_directories(${TEST_TARGET}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${TEST_TARGET}
        ${PROJECT_NAME}
        benchmark::benchmark_main
        GTest::gtest_main
)

find_package(GTest REQUIRED)
