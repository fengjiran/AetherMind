cmake_minimum_required(VERSION 3.22)
project(AetherMind)

set(CMAKE_CXX_STANDARD 20)

# Set custom Alloc Alignment for device allocated memory ndarray points to
set(USE_KALLOC_ALIGNMENT 64)

if(USE_KALLOC_ALIGNMENT)
    message(STATUS "Build Alloc alignment set to ${USE_KALLOC_ALIGNMENT}")
    add_definitions(-DKALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif(USE_KALLOC_ALIGNMENT)

find_package(glog REQUIRED)

set(FMT_PATH "3rdparty/fmt/include")

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

#message(STATUS ${PROJECT_SOURCE_DIR})
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${FMT_PATH}
)

target_link_libraries(${PROJECT_NAME}
        glog::glog
        pthread
        fmt::fmt
)

add_subdirectory(3rdparty/fmt)
add_subdirectory(test)
