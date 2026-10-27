include_guard(GLOBAL)

option(AETHERMIND_ENABLE_AVX2 "Build optional x86 AVX2/FMA implementations" ON)
set(AETHERMIND_HAS_AVX2_FMA FALSE)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _aethermind_processor)
if(AETHERMIND_ENABLE_AVX2 AND _aethermind_processor MATCHES "^(x86_64|amd64|x64|i[3-6]86|x86)$" AND
   CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -fno-lto")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            float output[8];
            const __m256 value = _mm256_set1_ps(1.0f);
            _mm256_storeu_ps(output, _mm256_fmadd_ps(value, value, value));
            return output[0] == 2.0f ? 0 : 1;
        }" AETHERMIND_COMPILER_HAS_AVX2_FMA)
    cmake_pop_check_state()
    if(AETHERMIND_COMPILER_HAS_AVX2_FMA)
        set(AETHERMIND_HAS_AVX2_FMA TRUE)
    endif()
endif()
unset(_aethermind_processor)
message(STATUS "AetherMind optional AVX2/FMA implementations: ${AETHERMIND_HAS_AVX2_FMA}")

function(aethermind_configure_avx2_sources)
    if(NOT AETHERMIND_HAS_AVX2_FMA)
        message(FATAL_ERROR "AVX2/FMA sources require a supported compiler and AETHERMIND_ENABLE_AVX2=ON")
    endif()
    # Keep ISA-specific instructions inside these translation units even when a
    # parent project enables unity builds or interprocedural optimization.
    set_property(SOURCE ${ARGN} APPEND PROPERTY COMPILE_OPTIONS -mavx2 -mfma -fno-lto)
    set_source_files_properties(${ARGN} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endfunction()
