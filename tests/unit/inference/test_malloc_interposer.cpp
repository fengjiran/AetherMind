#include "test_malloc_interposer.h"
#include "aethermind/base/macros.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>

// Sanitizer runtimes own malloc-family interception. Forwarding directly to
// __libc_* bypasses their allocation bookkeeping and can crash before tests
// start. Compile only inert helpers in these builds; filtering test cases would
// leave the conflicting process-wide override symbols in the executable.
#if defined(AM_SANITIZE_ADDRESS) || defined(AM_SANITIZE_THREAD) || \
        defined(__SANITIZE_THREAD__) || AM_HAS_FEATURE(thread_sanitizer)
#define AETHERMIND_TEST_MALLOC_INTERPOSER_DISABLED 1
#endif

#if defined(__GLIBC__) && defined(__linux__) && \
        !defined(AETHERMIND_TEST_MALLOC_INTERPOSER_DISABLED)

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" void* __libc_malloc(size_t size) noexcept;
extern "C" void* __libc_calloc(size_t count, size_t size) noexcept;
extern "C" void* __libc_realloc(void* ptr, size_t size) noexcept;
extern "C" void* __libc_memalign(size_t alignment, size_t size) noexcept;
extern "C" void __libc_free(void* ptr) noexcept;

namespace aethermind::test {
namespace {

constinit std::atomic_bool g_count_malloc_calls{false};
constinit std::atomic_size_t g_malloc_calls{0};
constinit std::atomic_size_t g_calloc_calls{0};
constinit std::atomic_size_t g_realloc_calls{0};
constinit std::atomic_size_t g_free_calls{0};
constinit std::atomic_size_t g_aligned_alloc_calls{0};
constinit std::atomic_size_t g_posix_memalign_calls{0};
constinit std::atomic_size_t g_memalign_calls{0};

struct MallocFailureState {
    bool enabled = false;
    size_t allocation_size = 0;
    size_t skip_matching_allocations = 0;
    bool after_aligned_allocation = false;
    MallocFailureTarget target = MallocFailureTarget::kMalloc;
    void* last_aligned_allocation = nullptr;
    MallocFailureResult result{};
};

constinit thread_local MallocFailureState g_malloc_failure{};

struct HugePageAdviceFailureState {
    bool enabled = false;
    size_t expected_bytes = 0;
    bool failed = false;
};

constinit thread_local HugePageAdviceFailureState g_advice_failure{};

} // namespace

bool MallocInterposerAvailable() noexcept {
    return true;
}

ScopedMallocFailure::ScopedMallocFailure(size_t allocation_size,
                                         size_t skip_matching_allocations,
                                         bool after_aligned_allocation,
                                         MallocFailureTarget target) noexcept {
    g_malloc_failure = {.enabled = true,
                        .allocation_size = allocation_size,
                        .skip_matching_allocations = skip_matching_allocations,
                        .after_aligned_allocation = after_aligned_allocation,
                        .target = target};
}

ScopedMallocFailure::~ScopedMallocFailure() noexcept {
    Stop();
}

MallocFailureResult ScopedMallocFailure::Stop() noexcept {
    if (!active_) {
        return {};
    }
    active_ = false;
    const MallocFailureResult result = g_malloc_failure.result;
    g_malloc_failure = {};
    return result;
}

ScopedHugePageAdviceFailure::ScopedHugePageAdviceFailure(size_t expected_bytes) noexcept {
    g_advice_failure = {.enabled = true, .expected_bytes = expected_bytes};
}

ScopedHugePageAdviceFailure::~ScopedHugePageAdviceFailure() noexcept {
    Stop();
}

bool ScopedHugePageAdviceFailure::Stop() noexcept {
    if (!active_) {
        return false;
    }
    active_ = false;
    const bool failed = g_advice_failure.failed;
    g_advice_failure = {};
    return failed;
}

void BeginMallocCallCounting() noexcept {
    g_malloc_calls.store(0, std::memory_order_relaxed);
    g_calloc_calls.store(0, std::memory_order_relaxed);
    g_realloc_calls.store(0, std::memory_order_relaxed);
    g_free_calls.store(0, std::memory_order_relaxed);
    g_aligned_alloc_calls.store(0, std::memory_order_relaxed);
    g_posix_memalign_calls.store(0, std::memory_order_relaxed);
    g_memalign_calls.store(0, std::memory_order_relaxed);
    g_count_malloc_calls.store(true, std::memory_order_release);
}

MallocCallCounts EndMallocCallCounting() noexcept {
    g_count_malloc_calls.store(false, std::memory_order_release);
    MallocCallCounts counts{
            .malloc_calls = g_malloc_calls.load(std::memory_order_relaxed),
            .calloc_calls = g_calloc_calls.load(std::memory_order_relaxed),
            .realloc_calls = g_realloc_calls.load(std::memory_order_relaxed),
            .free_calls = g_free_calls.load(std::memory_order_relaxed),
            .aligned_alloc_calls = g_aligned_alloc_calls.load(std::memory_order_relaxed),
            .posix_memalign_calls = g_posix_memalign_calls.load(std::memory_order_relaxed),
            .memalign_calls = g_memalign_calls.load(std::memory_order_relaxed),
    };
    return counts;
}

void CountMallocCall(std::atomic_size_t& count) noexcept {
    if (g_count_malloc_calls.load(std::memory_order_acquire)) {
        count.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace aethermind::test

extern "C" void* malloc(size_t size) noexcept {
    aethermind::test::CountMallocCall(aethermind::test::g_malloc_calls);
    auto& failure = aethermind::test::g_malloc_failure;
    if (failure.enabled && failure.target == aethermind::test::MallocFailureTarget::kMalloc &&
        !failure.result.allocation_failed &&
        (!failure.after_aligned_allocation || failure.last_aligned_allocation != nullptr) &&
        (failure.allocation_size == 0 || size == failure.allocation_size)) {
        if (failure.skip_matching_allocations > 0) {
            --failure.skip_matching_allocations;
        } else {
            failure.result.allocation_failed = true;
            errno = ENOMEM;
            return nullptr;
        }
    }
    return __libc_malloc(size);
}

extern "C" void* calloc(size_t count, size_t size) noexcept {
    aethermind::test::CountMallocCall(aethermind::test::g_calloc_calls);
    return __libc_calloc(count, size);
}

extern "C" void* realloc(void* ptr, size_t size) noexcept {
    aethermind::test::CountMallocCall(aethermind::test::g_realloc_calls);
    return __libc_realloc(ptr, size);
}

extern "C" void free(void* ptr) noexcept {
    aethermind::test::CountMallocCall(aethermind::test::g_free_calls);
    auto& failure = aethermind::test::g_malloc_failure;
    if (failure.enabled && ptr != nullptr && ptr == failure.last_aligned_allocation) {
        failure.result.last_aligned_allocation_released = true;
    }
    __libc_free(ptr);
}

extern "C" void* aligned_alloc(size_t alignment, size_t size) noexcept {
    aethermind::test::CountMallocCall(aethermind::test::g_aligned_alloc_calls);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        size % alignment != 0) {
        errno = EINVAL;
        return nullptr;
    }
    return __libc_memalign(alignment, size);
}

extern "C" int posix_memalign(void** result, size_t alignment, size_t size) noexcept {
    aethermind::test::CountMallocCall(aethermind::test::g_posix_memalign_calls);
    if (result == nullptr || alignment % sizeof(void*) != 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
        return EINVAL;
    }
    auto& failure = aethermind::test::g_malloc_failure;
    if (failure.enabled && failure.target == aethermind::test::MallocFailureTarget::kPosixMemalign &&
        !failure.result.allocation_failed &&
        (!failure.after_aligned_allocation || failure.last_aligned_allocation != nullptr) &&
        (failure.allocation_size == 0 || size == failure.allocation_size)) {
        if (failure.skip_matching_allocations > 0) {
            --failure.skip_matching_allocations;
        } else {
            failure.result.allocation_failed = true;
            return ENOMEM;
        }
    }
    const int saved_errno = errno;
    void* const data = __libc_memalign(alignment, size);
    if (data == nullptr) {
        return ENOMEM;
    }
    errno = saved_errno;
    *result = data;
    if (failure.enabled) {
        ++failure.result.aligned_allocation_calls;
        failure.last_aligned_allocation = data;
        failure.result.last_aligned_allocation_released = false;
    }
    return 0;
}

extern "C" void* memalign(size_t alignment, size_t size) noexcept {
    aethermind::test::CountMallocCall(aethermind::test::g_memalign_calls);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        errno = EINVAL;
        return nullptr;
    }
    return __libc_memalign(alignment, size);
}

extern "C" int madvise(void* address, size_t length, int advice) noexcept {
    auto& failure = aethermind::test::g_advice_failure;
    if (failure.enabled && !failure.failed && advice == MADV_HUGEPAGE &&
        length == failure.expected_bytes) {
        failure.failed = true;
        errno = EINVAL;
        return -1;
    }
    return static_cast<int>(syscall(SYS_madvise, address, length, advice));
}

#else

namespace aethermind::test {

bool MallocInterposerAvailable() noexcept {
    return false;
}

ScopedMallocFailure::ScopedMallocFailure(size_t, size_t, bool, MallocFailureTarget) noexcept {}

ScopedMallocFailure::~ScopedMallocFailure() noexcept = default;

MallocFailureResult ScopedMallocFailure::Stop() noexcept {
    active_ = false;
    return {};
}

ScopedHugePageAdviceFailure::ScopedHugePageAdviceFailure(size_t) noexcept {}

ScopedHugePageAdviceFailure::~ScopedHugePageAdviceFailure() noexcept = default;

bool ScopedHugePageAdviceFailure::Stop() noexcept {
    active_ = false;
    return false;
}

void BeginMallocCallCounting() noexcept {}

MallocCallCounts EndMallocCallCounting() noexcept {
    return {};
}

} // namespace aethermind::test

#endif
