cmake_minimum_required(VERSION 3.22)
project(AetherMindTest)
set(CMAKE_CXX_STANDARD 20)

option(BUILD_WITH_TORCH "build target with torch" OFF)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${TEST_SRC_FILES})

target_link_libraries(${PROJECT_NAME}
        AetherMind
        GTest::gtest_main
        glog::glog
)

if (BUILD_WITH_TORCH)
    message(STATUS "Setting C++ macro BUILD_WITH_TORCH - 1")
    find_package(Torch REQUIRED)
    #    message(STATUS "Torch LIB path: " ${TORCH_LIBRARIES})
    #    message(STATUS "Torch include path: " ${TORCH_INCLUDE_DIRS})
    include_directories(${TORCH_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_WITH_TORCH=1)
else ()
    message(STATUS "Setting C++ macro BUILD_WITH_TORCH - 0")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_WITH_TORCH=0)
endif ()

