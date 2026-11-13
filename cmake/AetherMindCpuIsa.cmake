include_guard(GLOBAL)

option(AETHERMIND_ENABLE_AVX2 "Build optional x86 AVX2 implementations" ON)
option(AETHERMIND_ENABLE_F16C "Build optional x86 F16C conversions" ON)
option(AETHERMIND_ENABLE_AVX512BF16 "Build optional x86 AVX512-BF16 implementations" ON)
option(AETHERMIND_ENABLE_AVX512FP16 "Build optional x86 AVX512-FP16 implementations" ON)
set(AETHERMIND_HAS_AVX2_FMA FALSE)
set(AETHERMIND_HAS_AVX2 FALSE)
set(AETHERMIND_HAS_F16C FALSE)
set(AETHERMIND_HAS_AVX512BF16 FALSE)
set(AETHERMIND_HAS_AVX512FP16 FALSE)

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

# These probes test toolchain support, not the host CPU: ISA translation units
# are compiled whenever the compiler can emit them and are selected at runtime
# from effective CPU features. Building them on a machine without the ISA is
# therefore expected and must not influence kernel eligibility.
if (_aethermind_processor MATCHES "^(x86_64|amd64|x64|i[3-6]86|x86)$" AND
        CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    if (AETHERMIND_ENABLE_AVX512BF16)
        set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bf16 -fno-lto")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                const __m512bh narrowed = _mm512_cvtne2ps_pbh(_mm512_set1_ps(1.0f),
                                                              _mm512_set1_ps(2.0f));
                const __m512 sum = _mm512_dpbf16_ps(_mm512_setzero_ps(), narrowed, narrowed);
                return static_cast<int>(_mm512_cvtss_f32(sum));
            }" AETHERMIND_COMPILER_HAS_AVX512BF16)
        if (AETHERMIND_COMPILER_HAS_AVX512BF16)
            set(AETHERMIND_HAS_AVX512BF16 TRUE)
        endif ()
    endif ()

    if (AETHERMIND_ENABLE_AVX512FP16)
        set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512fp16 -fno-lto")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                const __m512h value = _mm512_set1_ph(1.0f);
                const __m512 sum = _mm512_castph_ps(_mm512_mul_ph(value, value));
                return static_cast<int>(_mm512_cvtss_f32(sum));
            }" AETHERMIND_COMPILER_HAS_AVX512FP16)
        if (AETHERMIND_COMPILER_HAS_AVX512FP16)
            set(AETHERMIND_HAS_AVX512FP16 TRUE)
        endif ()
    endif ()
    cmake_pop_check_state()
endif ()

unset(_aethermind_processor)
message(STATUS "AetherMind AVX2/FMA toolchain support (compile-time; runtime eligibility uses detected CPU features): ${AETHERMIND_HAS_AVX2_FMA}")
message(STATUS "AetherMind FP16 conversion toolchain support (compile-time; runtime eligibility uses detected CPU features): F16C=${AETHERMIND_HAS_F16C}, AVX2=${AETHERMIND_HAS_AVX2}")
message(STATUS "AetherMind AVX512 toolchain support (compile-time; runtime eligibility uses detected CPU features): BF16=${AETHERMIND_HAS_AVX512BF16}, FP16=${AETHERMIND_HAS_AVX512FP16}")

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

# AVX512-BF16 dot products also use the AVX-512F ZMM/opmask state, so the base
# ISA flag is part of the contract rather than relying on compiler implication.
function(aethermind_configure_avx512bf16_sources)
    if (NOT AETHERMIND_HAS_AVX512BF16)
        message(FATAL_ERROR "AVX512-BF16 sources require a supported compiler and AETHERMIND_ENABLE_AVX512BF16=ON")
    endif ()
    set_property(SOURCE ${ARGN} APPEND PROPERTY COMPILE_OPTIONS -mavx512f -mavx512bf16 -fno-lto)
    set_source_files_properties(${ARGN} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endfunction()

function(aethermind_configure_avx512fp16_sources)
    if (NOT AETHERMIND_HAS_AVX512FP16)
        message(FATAL_ERROR "AVX512-FP16 sources require a supported compiler and AETHERMIND_ENABLE_AVX512FP16=ON")
    endif ()
    set_property(SOURCE ${ARGN} APPEND PROPERTY COMPILE_OPTIONS -mavx512f -mavx512fp16 -fno-lto)
    set_source_files_properties(${ARGN} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endfunction()
