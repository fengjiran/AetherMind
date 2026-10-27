include_guard(GLOBAL)

set(AETHERMIND_SANITIZER "none" CACHE STRING "Project sanitizer: none, asan-ubsan, or tsan")
set_property(CACHE AETHERMIND_SANITIZER PROPERTY STRINGS none asan-ubsan tsan)

if (NOT AETHERMIND_SANITIZER MATCHES "^(none|asan-ubsan|tsan)$")
    message(FATAL_ERROR "Unknown AETHERMIND_SANITIZER='${AETHERMIND_SANITIZER}'; choose none, asan-ubsan, or tsan")
endif ()
message(STATUS "AetherMind project sanitizer: ${AETHERMIND_SANITIZER}")

# Probe compile and link together: accepting a flag alone does not establish
# that the compiler's sanitizer runtime is installed and linkable.
if (NOT AETHERMIND_SANITIZER STREQUAL "none")
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" OR
            NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" OR WIN32)
        message(FATAL_ERROR "${AETHERMIND_SANITIZER} requires a supported GNU/Clang C and C++ toolchain on Unix")
    endif ()

    include(CheckCSourceCompiles)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    if (AETHERMIND_SANITIZER STREQUAL "asan-ubsan")
        set(_aethermind_sanitize_flag "-fsanitize=address,undefined")
        set(_aethermind_check_suffix "ADDRESS_UNDEFINED")
    else ()
        set(_aethermind_sanitize_flag "-fsanitize=thread")
        set(_aethermind_check_suffix "THREAD")
    endif ()

    set(CMAKE_REQUIRED_FLAGS "${_aethermind_sanitize_flag} -fno-omit-frame-pointer")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_aethermind_sanitize_flag}")
    check_c_source_compiles("int main(void) { return 0; }" AETHERMIND_C_HAS_${_aethermind_check_suffix})
    check_cxx_source_compiles("int main() { return 0; }" AETHERMIND_CXX_HAS_${_aethermind_check_suffix})
    cmake_pop_check_state()
    if (NOT AETHERMIND_C_HAS_${_aethermind_check_suffix} OR NOT AETHERMIND_CXX_HAS_${_aethermind_check_suffix})
        message(FATAL_ERROR "${AETHERMIND_SANITIZER} cannot compile and link with the selected C/C++ toolchain")
    endif ()
    unset(_aethermind_sanitize_flag)
    unset(_aethermind_check_suffix)
endif ()

# Only project-owned targets opt in. Third-party targets retain their own
# compile options, and executables link the same runtime as the shared library.
function(aethermind_apply_sanitizer target)
    if (AETHERMIND_SANITIZER STREQUAL "none")
        return()
    endif ()

    if (AETHERMIND_SANITIZER STREQUAL "asan-ubsan")
        set(sanitize_flag "-fsanitize=address,undefined")
    else ()
        set(sanitize_flag "-fsanitize=thread")
        target_compile_options(${target} PRIVATE -O1)
    endif ()

    target_compile_options(${target} PRIVATE "${sanitize_flag}" -fno-omit-frame-pointer -g)
    target_link_options(${target} PRIVATE "${sanitize_flag}")
endfunction()
