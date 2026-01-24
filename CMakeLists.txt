cmake_minimum_required(VERSION 3.28)
project(AetherMind)

add_compile_definitions(SPDLOG_USE_STD_FORMAT)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_MODULE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

# Set custom Alloc Alignment for device allocated memory ndarray points to
set(USE_KALLOC_ALIGNMENT 64)

if (USE_KALLOC_ALIGNMENT)
    message(STATUS "Build Alloc alignment set to ${USE_KALLOC_ALIGNMENT}")
    add_definitions(-DKALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif (USE_KALLOC_ALIGNMENT)

option(BUILD_TESTS "Adding test targets." ON)
option(USE_LIBBACKTRACE "Enable libbacktrace" ON)
option(BACKTRACE_ON_SEGFAULT "Set signal handler to print traceback on segfault" ON)

# add -fmodule-ts compiler option for module
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    add_compile_options(-fmodules-ts)
endif ()

if (USE_LIBBACKTRACE)
    include(cmake/AddLibbacktrace.cmake)
endif ()

find_package(glog REQUIRED)

set(BACKTRACE_PATH "3rdparty/libbacktrace")

# add spdlog
include(FetchContent)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.17.0
)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)
file(GLOB_RECURSE MODULE_INTERFACE_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/modules/*.ixx
        ${PROJECT_SOURCE_DIR}/src/modules/*.cppm
)

file(GLOB_RECURSE MODULE_IMPL_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/modules/*.cpp
)

list(FILTER SRC_FILES EXCLUDE REGEX ${PROJECT_SOURCE_DIR}/src/modules/*.*)

set(MODULE_NAME all_modules)
add_library(${MODULE_NAME} OBJECT)
target_sources(${MODULE_NAME} PUBLIC
        FILE_SET CXX_MODULES FILES
        ${MODULE_INTERFACE_FILES}
)
target_sources(${MODULE_NAME} PRIVATE ${MODULE_IMPL_FILES})

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
        glog::glog
        pthread
        spdlog::spdlog
        ${MODULE_NAME}
)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_SCAN_FOR_MODULES ON)

if (USE_LIBBACKTRACE)
    message(STATUS "Setting C++ macro USE_LIBBACKTRACE - 1")
    include_directories(${BACKTRACE_PATH})
    target_link_libraries(${PROJECT_NAME} libbacktrace)
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

#add_subdirectory(3rdparty/fmt)
########## Adding tests ##########
if (BUILD_TESTS)
    message(STATUS "Enable Testing")
    add_subdirectory(test)
endif ()
