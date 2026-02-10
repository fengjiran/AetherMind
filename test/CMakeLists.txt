cmake_minimum_required(VERSION 3.28)
project(AetherMindTest)

set(CMAKE_CXX_STANDARD 20)


option(BUILD_WITH_TORCH "build target with torch" OFF)

file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/*.cpp
)

#add_executable(unit_tests ${TEST_SRC_FILES})
add_executable(${PROJECT_NAME} ${TEST_SRC_FILES})

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
        AetherMind
        GTest::gtest_main
)

find_package(GTest REQUIRED)

if (BUILD_WITH_TORCH)
    message(STATUS "Setting C++ macro BUILD_WITH_TORCH - 1")
    find_package(Torch REQUIRED)
    #    message(STATUS "Torch LIB path: " ${TORCH_LIBRARIES})
    #    message(STATUS "Torch include path: " ${TORCH_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_WITH_TORCH=1)
else ()
    message(STATUS "Setting C++ macro BUILD_WITH_TORCH - 0")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_WITH_TORCH=0)
endif ()

