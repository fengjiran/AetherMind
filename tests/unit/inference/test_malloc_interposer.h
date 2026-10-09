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

struct MallocFailureResult {
    bool allocation_failed = false;
    size_t aligned_allocation_calls = 0;
    bool last_aligned_allocation_released = false;
};

/// @brief Fails the next matching malloc call on this thread, at most once.
///
/// Requires MallocInterposerAvailable(). Scopes must not nest. Tracking the last
/// successful posix_memalign pointer verifies cleanup after metadata failure.
class ScopedMallocFailure {
public:
    explicit ScopedMallocFailure(size_t allocation_size) noexcept;
    ~ScopedMallocFailure() noexcept;

    ScopedMallocFailure(const ScopedMallocFailure&) = delete;
    ScopedMallocFailure& operator=(const ScopedMallocFailure&) = delete;

    /// @brief Disables injection and clears its thread-local state.
    /// @return Observed failure and aligned payload cleanup, or empty if stopped.
    MallocFailureResult Stop() noexcept;

private:
    bool active_ = true;
};

} // namespace aethermind::test

#endif
