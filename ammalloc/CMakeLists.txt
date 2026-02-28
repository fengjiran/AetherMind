file(GLOB_RECURSE AMMALLOC_SRC_FILES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/ammalloc/src/*.cpp
)

add_library(ammalloc ${AMMALLOC_SRC_FILES})

target_include_directories(ammalloc
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include
        PRIVATE
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