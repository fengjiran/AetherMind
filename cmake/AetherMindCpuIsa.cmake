include_guard(GLOBAL)

option(AETHERMIND_ENABLE_AVX2 "Build optional x86 AVX2 implementations" ON)
option(AETHERMIND_ENABLE_F16C "Build optional x86 F16C conversions" ON)
set(AETHERMIND_HAS_AVX2_FMA FALSE)
set(AETHERMIND_HAS_AVX2 FALSE)
set(AETHERMIND_HAS_F16C FALSE)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _aethermind_processor)
if (_aethermind_processor MATCHES "^(x86_64|amd64|x64|i[3-6]86|x86)$" AND
        CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    if (AETHERMIND_ENABLE_AVX2)
        set(CMAKE_REQUIRED_FLAGS "-mavx2 -fno-lto")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                const __m256i value = _mm256_set1_epi32(8);
                const __m256i shifted = _mm256_srlv_epi32(value, value);
                return _mm256_extract_epi32(shifted, 0);
            }" AETHERMIND_COMPILER_HAS_AVX2)
        if (AETHERMIND_COMPILER_HAS_AVX2)
            set(AETHERMIND_HAS_AVX2 TRUE)
        endif ()
    endif ()

    if (AETHERMIND_ENABLE_F16C)
        set(CMAKE_REQUIRED_FLAGS "-mavx -mf16c -fno-lto")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                float output[8];
                _mm256_storeu_ps(output, _mm256_cvtph_ps(_mm_setzero_si128()));
                return output[0] != 0.0f;
            }" AETHERMIND_COMPILER_HAS_F16C)
        if (AETHERMIND_COMPILER_HAS_F16C)
            set(AETHERMIND_HAS_F16C TRUE)
        endif ()
    endif ()
    cmake_pop_check_state()
endif ()

if (AETHERMIND_ENABLE_AVX2 AND _aethermind_processor MATCHES "^(x86_64|amd64|x64|i[3-6]86|x86)$" AND
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
    if (AETHERMIND_COMPILER_HAS_AVX2_FMA)
        set(AETHERMIND_HAS_AVX2_FMA TRUE)
    endif ()
endif ()

unset(_aethermind_processor)
message(STATUS "AetherMind optional AVX2/FMA implementations: ${AETHERMIND_HAS_AVX2_FMA}")
message(STATUS "AetherMind optional FP16 conversions: F16C=${AETHERMIND_HAS_F16C}, AVX2=${AETHERMIND_HAS_AVX2}")

function(aethermind_configure_f16c_sources)
    if (NOT AETHERMIND_HAS_F16C)
        message(FATAL_ERROR "F16C sources require a supported compiler and AETHERMIND_ENABLE_F16C=ON")
    endif ()
    set_property(SOURCE ${ARGN} APPEND PROPERTY COMPILE_OPTIONS -mavx -mf16c -fno-lto)
    set_source_files_properties(${ARGN} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endfunction()

function(aethermind_configure_avx2_integer_sources)
    if (NOT AETHERMIND_HAS_AVX2)
        message(FATAL_ERROR "AVX2 sources require a supported compiler and AETHERMIND_ENABLE_AVX2=ON")
    endif ()
    set_property(SOURCE ${ARGN} APPEND PROPERTY COMPILE_OPTIONS -mavx2 -fno-lto)
    set_source_files_properties(${ARGN} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endfunction()

function(aethermind_configure_avx2_sources)
    if (NOT AETHERMIND_HAS_AVX2_FMA)
        message(FATAL_ERROR "AVX2/FMA sources require a supported compiler and AETHERMIND_ENABLE_AVX2=ON")
    endif ()
    # Keep ISA-specific instructions inside these translation units even when a
    # parent project enables unity builds or interprocedural optimization.
    set_property(SOURCE ${ARGN} APPEND PROPERTY COMPILE_OPTIONS -mavx2 -mfma -fno-lto)
    set_source_files_properties(${ARGN} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endfunction()
