cmake_minimum_required(VERSION 3.22)
project(AetherMind)

set(CMAKE_CXX_STANDARD 20)

# Set custom Alloc Alignment for device allocated memory ndarray points to
set(USE_KALLOC_ALIGNMENT 64)

if (USE_KALLOC_ALIGNMENT)
    message(STATUS "Build Alloc alignment set to ${USE_KALLOC_ALIGNMENT}")
    add_definitions(-DKALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif (USE_KALLOC_ALIGNMENT)

option(BUILD_TESTS "Adding test targets." ON)
option(USE_LIBBACKTRACE "Enable libbacktrace" ON)
option(BACKTRACE_ON_SEGFAULT "Set signal handler to print traceback on segfault" ON)

if (USE_LIBBACKTRACE)
    include(cmake/AddLibbacktrace.cmake)
endif ()

find_package(glog REQUIRED)

set(FMT_PATH "3rdparty/fmt/include")
set(BACKTRACE_PATH "3rdparty/libbacktrace")

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

#message(STATUS ${PROJECT_SOURCE_DIR})
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${FMT_PATH}
        ${BACKTRACE_PATH}
)

target_link_libraries(${PROJECT_NAME}
        glog::glog
        pthread
        fmt::fmt
        libbacktrace
)

if (USE_LIBBACKTRACE)
    message(STATUS "Setting C++ macro USE_LIBBACKTRACE - 1")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_LIBBACKTRACE=1)
else ()
    message(STATUS "Setting C++ macro USE_LIBBACKTRACE - 0")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_LIBBACKTRACE=0)
endif (USE_LIBBACKTRACE)

if (BACKTRACE_ON_SEGFAULT)
    message(STATUS "Setting C++ macro BACKTRACE_ON_SEGFAULT - 1")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BACKTRACE_ON_SEGFAULT=1)
else ()
    message(STATUS "Setting C++ macro BACKTRACE_ON_SEGFAULT - 0")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BACKTRACE_ON_SEGFAULT=0)
endif (BACKTRACE_ON_SEGFAULT)

add_subdirectory(3rdparty/fmt)

########## Adding tests ##########
if (BUILD_TESTS)
    message(STATUS "Enable Testing")
    add_subdirectory(test)
endif ()
