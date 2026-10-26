include_guard(GLOBAL)

# Parse in the target compiler rather than CMake's signed 64-bit arithmetic.
# The check only compiles, so it also works when the target cannot run locally.
function(aethermind_validate_alloc_alignment alignment)
    if (NOT "${alignment}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "KALLOC_ALIGNMENT='${alignment}' must be a canonical positive decimal integer")
    endif ()

    set(check_source [=[
#include <cstddef>
#include <limits>

constexpr bool IsValidAlignment() {
    constexpr char digits[] = "@alignment@";
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    std::size_t value = 0;
    for (std::size_t i = 0; i + 1 < sizeof(digits); ++i) {
        const auto digit = static_cast<std::size_t>(digits[i] - '0');
        if (value > (maximum - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    return value != 0 && (value & (value - 1)) == 0;
}

static_assert(IsValidAlignment(),
              "KALLOC_ALIGNMENT must be a power of two representable as size_t");
]=])
    string(CONFIGURE "${check_source}" check_source @ONLY)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    try_compile(alignment_valid
        SOURCE_FROM_CONTENT alignment_check.cpp "${check_source}"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        NO_CACHE
        OUTPUT_VARIABLE check_output
    )
    if (NOT alignment_valid)
        message(FATAL_ERROR
            "KALLOC_ALIGNMENT='${alignment}' must be a power of two representable as target size_t.\n${check_output}")
    endif ()
endfunction()
