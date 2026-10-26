cmake_minimum_required(VERSION 3.28)
project(AetherMindBuildContract LANGUAGES C CXX)

# A consumer must obtain C++20 from AetherMind's usage requirements.
set(CMAKE_CXX_STANDARD 17)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_INFERENCE_TOOL OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(USE_LIBBACKTRACE OFF CACHE BOOL "" FORCE)
set(AETHERMIND_SANITIZER none CACHE STRING "" FORCE)
set(FETCHCONTENT_SOURCE_DIR_SPDLOG "${AETHERMIND_SPDLOG_SOURCE_DIR}" CACHE PATH "" FORCE)

if(AETHERMIND_EXISTING_SPDLOG)
    # Use a real shared dependency so overwriting the host's build policy is
    # observable, in addition to the logging ABI checked by the executable.
    set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
    add_subdirectory("${AETHERMIND_SPDLOG_SOURCE_DIR}" host-spdlog)
    if(SPDLOG_USE_STD_FORMAT)
        target_compile_features(spdlog PUBLIC cxx_std_20)
        target_compile_features(spdlog_header_only INTERFACE cxx_std_20)
    endif()
    foreach(property TYPE INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_FEATURES)
        get_target_property(before_${property} spdlog::spdlog ${property})
        if("${before_${property}}" MATCHES "-NOTFOUND$")
            set(before_${property} "")
        endif()
    endforeach()
    set(before_format "${SPDLOG_USE_STD_FORMAT}")
    set(before_shared "${SPDLOG_BUILD_SHARED}")
    set(before_pic "${SPDLOG_BUILD_PIC}")
endif()

add_subdirectory("${AETHERMIND_SOURCE_DIR}" aethermind)

if(AETHERMIND_EXISTING_SPDLOG)
    foreach(property TYPE INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_FEATURES)
        get_target_property(after_${property} spdlog::spdlog ${property})
        if("${after_${property}}" MATCHES "-NOTFOUND$")
            set(after_${property} "")
        endif()
        if(NOT "${before_${property}}" STREQUAL "${after_${property}}")
            message(FATAL_ERROR "AetherMind changed host spdlog ${property}")
        endif()
    endforeach()
    if(NOT "${before_format}" STREQUAL "${SPDLOG_USE_STD_FORMAT}" OR
       NOT "${before_shared}" STREQUAL "${SPDLOG_BUILD_SHARED}" OR
       NOT "${before_pic}" STREQUAL "${SPDLOG_BUILD_PIC}")
        message(FATAL_ERROR "AetherMind changed the host's spdlog options")
    endif()
endif()

if(NOT "${SPDLOG_USE_STD_FORMAT}" STREQUAL "${AETHERMIND_EXPECT_STD_FORMAT}")
    message(FATAL_ERROR "Unexpected spdlog format backend: ${SPDLOG_USE_STD_FORMAT}")
endif()

# Export the actual target rather than copying its properties into a test
# facade. The compile-only consumer can import this contract without building
# the entire shared library in every dependency-mode regression.
export(TARGETS AetherMind spdlog FILE "${CMAKE_CURRENT_BINARY_DIR}/AetherMindContractTargets.cmake")

add_executable(contract_logger logger.cpp)
target_link_libraries(contract_logger PRIVATE spdlog::spdlog)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/logger-location-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:contract_logger>")
