cmake_minimum_required(VERSION 3.28)
project(AetherMind
        VERSION 1.0.0
        DESCRIPTION "Inference Engine"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(USE_KALLOC_ALIGNMENT 64)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)

option(BUILD_TESTS "Adding test targets." ON)
option(USE_LIBBACKTRACE "Enable libbacktrace" ON)
option(BACKTRACE_ON_SEGFAULT "Set signal handler to print traceback on segfault" ON)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if (USE_KALLOC_ALIGNMENT)
    message(STATUS "Build Alloc alignment set to ${USE_KALLOC_ALIGNMENT}")
    add_compile_definitions(KALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif (USE_KALLOC_ALIGNMENT)

if(ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    # 1. 添加编译选项
    add_compile_options(-fsanitize=thread -g -O1)
    # -g:  保留调试信息，让 TSAN 能打印出行号
    # -O1: 适度优化，TSAN 建议在 O1 或 O2 下运行，O0 可能会太慢或误报

    # 2. 添加链接选项 (这一步非常重要，否则链接会失败)
    add_link_options(-fsanitize=thread)

    # 可选：为了获得更好的堆栈回溯，建议加上这个
    add_compile_options(-fno-omit-frame-pointer)
endif()

if (USE_LIBBACKTRACE)
    set(BACKTRACE_PATH "3rdparty/libbacktrace")
    include(cmake/AddLibbacktrace.cmake)
endif ()

add_compile_definitions(SPDLOG_USE_STD_FORMAT)

# add spdlog
include(FetchContent)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(spdlog)
find_package(Threads REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)
#file(GLOB_RECURSE MODULE_INTERFACE_FILES CONFIGURE_DEPENDS
#        ${PROJECT_SOURCE_DIR}/src/modules/*.ixx
#        ${PROJECT_SOURCE_DIR}/src/modules/*.cppm
#)
#
#file(GLOB_RECURSE MODULE_IMPL_FILES CONFIGURE_DEPENDS
#        ${PROJECT_SOURCE_DIR}/src/modules/*.cpp
#)

#list(FILTER SRC_FILES EXCLUDE REGEX ${PROJECT_SOURCE_DIR}/src/modules/*.*)

#set(MODULE_NAME all_modules)
#add_library(${MODULE_NAME} OBJECT)
#target_sources(${MODULE_NAME} PUBLIC
#        FILE_SET CXX_MODULES FILES
#        ${MODULE_INTERFACE_FILES}
#)
#target_sources(${MODULE_NAME} PRIVATE ${MODULE_IMPL_FILES})

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
        Threads::Threads
        spdlog::spdlog
#        ${MODULE_NAME}
)

#set_target_properties(${PROJECT_NAME} PROPERTIES CXX_SCAN_FOR_MODULES ON)

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

add_subdirectory(src)

########## Adding tests ##########
if (BUILD_TESTS)
    message(STATUS "Enable Testing")
    add_compile_definitions(PAGE_ALLOCATOR_TEST)
    add_subdirectory(test)
endif ()
