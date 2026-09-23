#include "test_malloc_interposer.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>

#if defined(__GLIBC__) && defined(__linux__)

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

} // namespace

bool MallocInterposerAvailable() noexcept {
    return true;
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
    const int saved_errno = errno;
    void* const data = __libc_memalign(alignment, size);
    if (data == nullptr) {
        return ENOMEM;
    }
    errno = saved_errno;
    *result = data;
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

#else

namespace aethermind::test {

bool MallocInterposerAvailable() noexcept {
    return false;
}

void BeginMallocCallCounting() noexcept {}

MallocCallCounts EndMallocCallCounting() noexcept {
    return {};
}

} // namespace aethermind::test

#endif
