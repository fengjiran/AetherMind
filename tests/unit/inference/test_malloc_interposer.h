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

enum class MallocFailureTarget {
    kMalloc,
    kPosixMemalign,
};

/// @brief Fails one matching allocation call on this thread, at most once.
///
/// Requires MallocInterposerAvailable(). Scopes must not nest. Tracking the last
/// successful posix_memalign pointer verifies cleanup after metadata failure.
class ScopedMallocFailure {
public:
    /// @param allocation_size Matching byte count, or zero to match any size.
    /// @param skip_matching_allocations Number of matching calls to allow first.
    /// @param after_aligned_allocation Start matching after a successful
    ///        posix_memalign, so public-API setup allocations remain unaffected.
    /// @param target Allocation function to fail. posix_memalign returns ENOMEM
    ///        without modifying its result pointer or errno.
    explicit ScopedMallocFailure(size_t allocation_size,
                                 size_t skip_matching_allocations = 0,
                                 bool after_aligned_allocation = false,
                                 MallocFailureTarget target = MallocFailureTarget::kMalloc) noexcept;
    ~ScopedMallocFailure() noexcept;

    ScopedMallocFailure(const ScopedMallocFailure&) = delete;
    ScopedMallocFailure& operator=(const ScopedMallocFailure&) = delete;

    /// @brief Disables injection and clears its thread-local state.
    /// @return Observed failure and aligned payload cleanup, or empty if stopped.
    MallocFailureResult Stop() noexcept;

private:
    bool active_ = true;
};

/// @brief Fails one matching MADV_HUGEPAGE call on this thread.
///
/// Requires MallocInterposerAvailable(). Other advice and lengths are forwarded
/// normally. Scopes must not nest.
class ScopedHugePageAdviceFailure {
public:
    explicit ScopedHugePageAdviceFailure(size_t expected_bytes) noexcept;
    ~ScopedHugePageAdviceFailure() noexcept;

    ScopedHugePageAdviceFailure(const ScopedHugePageAdviceFailure&) = delete;
    ScopedHugePageAdviceFailure& operator=(const ScopedHugePageAdviceFailure&) = delete;

    /// @brief Stops injection and reports whether the matching call failed.
    bool Stop() noexcept;

private:
    bool active_ = true;
};

} // namespace aethermind::test

#endif
