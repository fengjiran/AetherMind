cmake_minimum_required(VERSION 3.28)
project(AetherMind
        VERSION 1.0.0
        DESCRIPTION "Inference Engine"
        LANGUAGES CXX C
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

find_package(Threads REQUIRED)

if (ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    # 1. 添加编译选项
    add_compile_options(-fsanitize=thread -g -O1)
    # -g:  保留调试信息，让 TSAN 能打印出行号
    # -O1: 适度优化，TSAN 建议在 O1 或 O2 下运行，O0 可能会太慢或误报

    # 2. 添加链接选项 (这一步非常重要，否则链接会失败)
    add_link_options(-fsanitize=thread)

    # 可选：为了获得更好的堆栈回溯，建议加上这个
    add_compile_options(-fno-omit-frame-pointer)
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

add_subdirectory(src)
if (BUILD_TESTS)
    message(STATUS "Enable Testing")
    add_subdirectory(test)
endif ()
