file(GLOB_RECURSE AMMALLOC_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/ammalloc/src/*.cpp
)

add_library(ammalloc STATIC ${AMMALLOC_SRC_FILES})

option(USE_57BIT_VA "System use 57bits virtual address space." OFF)

target_include_directories(ammalloc
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ammalloc
        Threads::Threads
        spdlog::spdlog
)

if (BUILD_TESTS)
    target_compile_definitions(ammalloc PUBLIC AMMALLOC_TEST)
endif()

if (USE_57BIT_VA)
    target_compile_definitions(ammalloc PUBLIC AM_USE_57BIT_VA)
endif ()