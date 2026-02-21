set(BENCHMARK_TARGET aethermind_benchmark)
file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/tests/benchmark/*.cpp
)

add_executable(${BENCHMARK_TARGET} ${TEST_SRC_FILES})

target_include_directories(${BENCHMARK_TARGET}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${BENCHMARK_TARGET}
        ${PROJECT_NAME}
        benchmark::benchmark_main
)
