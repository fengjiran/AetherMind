#ifndef AETHERMIND_TEST_MALLOC_INTERPOSER_H
#define AETHERMIND_TEST_MALLOC_INTERPOSER_H

#include <cstddef>

namespace aethermind::test {

struct MallocCallCounts {
    size_t malloc_calls = 0;
    size_t calloc_calls = 0;
    size_t realloc_calls = 0;
    size_t free_calls = 0;
    size_t aligned_alloc_calls = 0;
    size_t posix_memalign_calls = 0;
    size_t memalign_calls = 0;
};

/// @brief Returns whether this process has the glibc interposer enabled.
bool MallocInterposerAvailable() noexcept;

/// @brief Starts counting process-wide malloc-family calls.
void BeginMallocCallCounting() noexcept;

/// @brief Stops counting and returns the calls observed since Begin.
MallocCallCounts EndMallocCallCounting() noexcept;

} // namespace aethermind::test

#endif
