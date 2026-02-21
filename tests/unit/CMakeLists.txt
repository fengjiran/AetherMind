set(TEST_TARGET aethermind_unit_tests)
option(BUILD_WITH_TORCH "build target with torch" OFF)

file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/tests/unit/*.cpp
)

add_executable(${TEST_TARGET} ${TEST_SRC_FILES})

target_include_directories(${TEST_TARGET}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${TEST_TARGET}
        ${PROJECT_NAME}
        GTest::gtest_main
)

find_package(GTest REQUIRED)

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
