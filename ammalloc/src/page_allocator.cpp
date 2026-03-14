/// Implementation of PageAllocator and internal HugePageCache.
///
/// Optimistic huge-page allocation strategy:
/// 1. Try exact-size mmap first; accept if already huge-page aligned.
/// 2. If misaligned, unmap and retry with over-allocation + trimming.
/// 3. Cache released 2MB huge pages to reduce future mmap overhead.

#include "ammalloc/page_allocator.h"
#include "ammalloc/config.h"
#include "utils/logging.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <sys/mman.h>

namespace {

// Zero-allocation lock-free dual-stack cache for 2MB huge pages.
//
// This cache eliminates the std::mutex bottleneck under high concurrency while
// strictly adhering to ammalloc's zero-allocation bootstrapping constraints
// (no new/delete, no std::vector).
//
// It maintains two lock-free stacks (free_head_ and used_head_) over a fixed
// array of Slots. ABA problem is prevented by packing a 16-bit index and a
// 48-bit tag into a single 64-bit atomic head.
class HugePageCache {
public:
    static HugePageCache& GetInstance() {
        // Use static storage + placement new to avoid singleton destruction order issues
        // and prevent recursive allocator calls during initialization.
        alignas(alignof(HugePageCache)) static char storage[sizeof(HugePageCache)];
        static auto* instance = new (storage) HugePageCache();
        return *instance;
    }

    // Pops a cached 2MB huge page from the used stack.
    // Returns nullptr if the cache is empty.
    void* Get() noexcept {
        uint16_t index = kInvalid;
        if (!Pop(used_head_, index)) {
            return nullptr;
        }

        void* ptr = slots_[index].ptr;
        Push(free_head_, index);
        return ptr;
    }

    // Pushes a 2MB huge page into the cache.
    // Returns false if the cache is full, in which case the caller must munmap.
    bool Put(void* ptr) noexcept {
        uint16_t index = kInvalid;
        if (!Pop(free_head_, index)) {
            return false;
        }

        slots_[index].ptr = ptr;
        Push(used_head_, index);
        return true;
    }

    void ReleaseAllForTesting() {
        while (void* ptr = Get()) {
            if (munmap(ptr, aethermind::SystemConfig::HUGE_PAGE_SIZE) != 0) {
                aethermind::PageAllocator::RecordMunmapFailure();
                spdlog::error("munmap failed in HugePageCache::ReleaseAll: ptr={}, errno={}",
                              ptr, errno);
            }
        }
    }

private:
    struct Slot {
        void* ptr{nullptr};
        std::atomic<uint16_t> next{kInvalid};
    };

    static constexpr uint16_t kInvalid = 0xFFFF;
    static constexpr size_t kCapacity = aethermind::PageConfig::HUGE_PAGE_CACHE_SIZE;
    static_assert(kCapacity > 0 && kCapacity < kInvalid);
    static_assert(std::atomic<uint64_t>::is_always_lock_free);

    static constexpr uint64_t kIndexMask = 0xFFFFull;
    static constexpr uint64_t kTagMask = (1ull << 48) - 1;

    alignas(aethermind::SystemConfig::CACHE_LINE_SIZE) std::atomic<uint64_t> free_head_{0};
    alignas(aethermind::SystemConfig::CACHE_LINE_SIZE) std::atomic<uint64_t> used_head_{0};
    Slot slots_[kCapacity]{};

    static constexpr uint64_t Pack(uint16_t index, uint64_t tag) noexcept {
        return ((tag & kTagMask) << 16) | index;
    }

    static constexpr uint16_t Index(uint64_t head) noexcept {
        return static_cast<uint16_t>(head & kIndexMask);
    }

    static constexpr uint64_t Tag(uint64_t head) noexcept {
        return head >> 16;
    }

    HugePageCache() noexcept {
        for (uint16_t i = 0; i < kCapacity - 1; ++i) {
            slots_[i].next.store(i + 1, std::memory_order_relaxed);
        }
        slots_[kCapacity - 1].next.store(kInvalid, std::memory_order_relaxed);
        free_head_.store(Pack(0, 0), std::memory_order_relaxed);
        used_head_.store(Pack(kInvalid, 0), std::memory_order_relaxed);
    }

    // Lock-free pop from the specified stack head.
    // Returns true and sets `out` to the popped index, or false if empty.
    bool Pop(std::atomic<uint64_t>& head, uint16_t& out) noexcept {
        uint64_t old_val = head.load(std::memory_order_acquire);
        while (true) {
            const uint16_t index = Index(old_val);
            if (index == kInvalid) {
                return false;
            }

            const uint16_t next = slots_[index].next.load(std::memory_order_relaxed);
            const uint64_t new_val = Pack(next, Tag(old_val) + 1);
            if (head.compare_exchange_weak(old_val, new_val,
                                           std::memory_order_acquire,
                                           std::memory_order_acquire)) {
                out = index;
                return true;
            }
        }
    }

    // Lock-free push of an index onto the specified stack head.
    void Push(std::atomic<uint64_t>& head, uint16_t index) noexcept {
        uint64_t old_val = head.load(std::memory_order_relaxed);
        while (true) {
            slots_[index].next.store(Index(old_val), std::memory_order_relaxed);
            const uint64_t new_val = Pack(index, Tag(old_val) + 1);
            if (head.compare_exchange_weak(old_val, new_val,
                                           std::memory_order_release,
                                           std::memory_order_relaxed)) {
                return;
            }
        }
    }
};

}// namespace

namespace aethermind {

#ifdef AMMALLOC_TEST
std::atomic<bool> g_mock_huge_alloc_fail{false};
#endif

#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif


void PageAllocator::ResetStats() {
    stats_.normal_alloc_count.store(0, std::memory_order_relaxed);
    stats_.normal_alloc_success.store(0, std::memory_order_relaxed);
    stats_.normal_alloc_bytes.store(0, std::memory_order_relaxed);
    stats_.huge_alloc_count.store(0, std::memory_order_relaxed);
    stats_.huge_alloc_success.store(0, std::memory_order_relaxed);
    stats_.huge_alloc_bytes.store(0, std::memory_order_relaxed);
    stats_.huge_align_waste_bytes.store(0, std::memory_order_relaxed);
    stats_.huge_cache_hit_count.store(0, std::memory_order_relaxed);
    stats_.huge_cache_miss_count.store(0, std::memory_order_relaxed);
    stats_.free_count.store(0, std::memory_order_relaxed);
    stats_.free_bytes.store(0, std::memory_order_relaxed);
    stats_.alloc_failed_count.store(0, std::memory_order_relaxed);
    stats_.huge_fallback_to_normal_count.store(0, std::memory_order_relaxed);
    stats_.normal_alloc_failed_count.store(0, std::memory_order_relaxed);
    stats_.huge_alloc_failed_count.store(0, std::memory_order_relaxed);
    stats_.munmap_failed_count.store(0, std::memory_order_relaxed);
    stats_.madvise_failed_count.store(0, std::memory_order_relaxed);
    stats_.mmap_enomem_count.store(0, std::memory_order_relaxed);
    stats_.mmap_other_error_count.store(0, std::memory_order_relaxed);
}

bool PageAllocator::SafeMunmap(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return true;
    }

    if (munmap(ptr, size) != 0) {
        stats_.munmap_failed_count.fetch_add(1, std::memory_order_relaxed);
        spdlog::error("munmap failed: ptr={}, size={}, errno={}, msg={}",
                      ptr, size, errno, strerror(errno));
        return false;
    }
    return true;
}

void* PageAllocator::AllocWithRetry(size_t size, int flags) {
    for (size_t i = 0; i < PageConfig::MAX_ALLOC_RETRIES; ++i) {
        if (void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
            ptr != MAP_FAILED) {
            return ptr;
        }

        if (errno == ENOMEM) {
            stats_.mmap_enomem_count.fetch_add(1, std::memory_order_relaxed);
            // ENOMEM may be transient under memory pressure; back off briefly
            // before retrying.
            spdlog::warn("mmap ENOMEM for size {}, retry {}/{}...",
                         size, i + 1, PageConfig::MAX_ALLOC_RETRIES);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            stats_.mmap_other_error_count.fetch_add(1, std::memory_order_relaxed);
            spdlog::error("mmap fatal error: errno={}, msg={}", errno, strerror(errno));
            break;
        }
    }

    return MAP_FAILED;
}

void PageAllocator::ApplyHugePageHint(void* ptr, size_t size) {
    if (madvise(ptr, size, MADV_HUGEPAGE) != 0) {
        stats_.madvise_failed_count.fetch_add(1, std::memory_order_relaxed);
        spdlog::debug("madvise MADV_HUGEPAGE failed (expected on non-THP systems).");
    }

    if (RuntimeConfig::GetInstance().UseMapPopulate()) {
        if (madvise(ptr, size, MADV_WILLNEED) != 0) {
            stats_.madvise_failed_count.fetch_add(1, std::memory_order_relaxed);
            spdlog::warn("madvise MADV_WILLNEED failed: ptr={}, size={}", ptr, size);
        }
    }
}

void* PageAllocator::AllocNormalPage(size_t size) {
    stats_.normal_alloc_count.fetch_add(1, std::memory_order_relaxed);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (RuntimeConfig::GetInstance().UseMapPopulate()) {
        flags |= MAP_POPULATE;
    }

    void* ptr = AllocWithRetry(size, flags);
    if (ptr == MAP_FAILED) {
        stats_.normal_alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    stats_.normal_alloc_success.fetch_add(1, std::memory_order_relaxed);
    stats_.normal_alloc_bytes.fetch_add(size, std::memory_order_relaxed);

    return ptr;
}

void* PageAllocator::AllocHugePageWithTrim(size_t size) {
    // Overflow guard: size + HUGE_PAGE_SIZE must not wrap
    // clang-format off
    if (size > (std::numeric_limits<size_t>::max() - SystemConfig::HUGE_PAGE_SIZE)) AM_UNLIKELY {
        spdlog::error("AllocHugePageWithTrim size overflow: {}", size);
        return nullptr;
    }
    // clang-format on

    size_t alloc_size = size + SystemConfig::HUGE_PAGE_SIZE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    void* ptr = AllocWithRetry(alloc_size, flags);
    if (ptr == MAP_FAILED) {
        stats_.huge_alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned_addr = (addr + SystemConfig::HUGE_PAGE_SIZE - 1) &
                                   ~(SystemConfig::HUGE_PAGE_SIZE - 1);
    AM_DCHECK((aligned_addr & (SystemConfig::HUGE_PAGE_SIZE - 1)) == 0);
    AM_DCHECK(aligned_addr >= addr);
    AM_DCHECK(aligned_addr + size <= addr + alloc_size);

    size_t waste = 0;
    const size_t head_gap = aligned_addr - addr;
    if (head_gap > 0) {
        SafeMunmap(ptr, head_gap);
        waste += head_gap;
    }

    if (const size_t tail_gap = alloc_size - head_gap - size; tail_gap > 0) {
        SafeMunmap(reinterpret_cast<void*>(aligned_addr + size), tail_gap);
        waste += tail_gap;
    }

    stats_.huge_align_waste_bytes.fetch_add(waste, std::memory_order_relaxed);

    auto* res = reinterpret_cast<void*>(aligned_addr);
    ApplyHugePageHint(res, size);
    stats_.huge_alloc_success.fetch_add(1, std::memory_order_relaxed);
    stats_.huge_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    return res;
}

// Optimistic huge-page allocation strategy:
// 1) First try exact-size mmap and accept it if already huge-page aligned.
// 2) If misaligned, unmap it and retry with over-allocation + trimming.
// This avoids extra VMA operations on the fast-success path.
void* PageAllocator::AllocHugePage(size_t size) {
    stats_.huge_alloc_count.fetch_add(1, std::memory_order_relaxed);

#ifdef AMMALLOC_TEST
    if (g_mock_huge_alloc_fail.load(std::memory_order_relaxed)) {
        stats_.huge_alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
#endif

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    void* ptr = AllocWithRetry(size, flags);
    if (ptr == MAP_FAILED) {
        stats_.huge_alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    auto addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & (SystemConfig::HUGE_PAGE_SIZE - 1)) == 0) {
        ApplyHugePageHint(ptr, size);
        stats_.huge_alloc_success.fetch_add(1, std::memory_order_relaxed);
        stats_.huge_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
        return ptr;
    }

    SafeMunmap(ptr, size);
    return AllocHugePageWithTrim(size);
}

void* PageAllocator::SystemAlloc(size_t page_num) {
    // clang-format off
    if (page_num == 0) AM_UNLIKELY {
        spdlog::warn("SystemAlloc called with page_num = 0");
        return nullptr;
    }

    // Overflow guard: ensure page_num << PAGE_SHIFT doesn't wrap
    if (page_num > (std::numeric_limits<size_t>::max() >> SystemConfig::PAGE_SHIFT)) AM_UNLIKELY {
        spdlog::error("SystemAlloc page_num overflow: {}", page_num);
        return nullptr;
    }

    const size_t size = page_num << SystemConfig::PAGE_SHIFT;
    // Requests below 1MB do not benefit enough from huge-page machinery.
    if (size < (SystemConfig::HUGE_PAGE_SIZE >> 1)) AM_LIKELY {
        void* ptr = AllocNormalPage(size);
        if (!ptr) {
            stats_.alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        }
        return ptr;
    }
    // clang-format on

    // Cache only exact-size huge pages (2MB). Variable-sized large mappings are
    // returned directly to the OS on free.
    if (size == SystemConfig::HUGE_PAGE_SIZE) {
        void* ptr = HugePageCache::GetInstance().Get();
        auto addr = reinterpret_cast<uintptr_t>(ptr);
        if (ptr && (addr & (SystemConfig::HUGE_PAGE_SIZE - 1)) == 0) {
            stats_.huge_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
            return ptr;
        }
        stats_.huge_cache_miss_count.fetch_add(1, std::memory_order_relaxed);
    }

    auto* ptr = AllocHugePage(size);
    // Keep availability-first semantics: fall back to normal pages when huge-page
    // allocation fails.
    if (!ptr) {
        stats_.huge_fallback_to_normal_count.fetch_add(1, std::memory_order_relaxed);
        ptr = AllocNormalPage(size);
        if (!ptr) {
            stats_.alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    return ptr;
}

void PageAllocator::SystemFree(void* ptr, size_t page_num) {
    if (!ptr || page_num == 0) {
        return;
    }

    // Overflow guard
    // clang-format off
    if (page_num > (std::numeric_limits<size_t>::max() >> SystemConfig::PAGE_SHIFT)) AM_UNLIKELY {
        spdlog::error("SystemFree page_num overflow: {}", page_num);
        return;
    }
    // clang-format on

    auto addr = reinterpret_cast<uintptr_t>(ptr);
    const size_t size = page_num << SystemConfig::PAGE_SHIFT;
    stats_.free_count.fetch_add(1, std::memory_order_relaxed);
    stats_.free_bytes.fetch_add(size, std::memory_order_relaxed);
    // For exact-size huge pages, prefer caching the VMA and dropping physical
    // backing (`MADV_DONTNEED`) to reduce future mmap/munmap overhead.
    if (size == SystemConfig::HUGE_PAGE_SIZE && (addr & (SystemConfig::HUGE_PAGE_SIZE - 1)) == 0) {
        if (madvise(ptr, size, MADV_DONTNEED) != 0) {
            stats_.madvise_failed_count.fetch_add(1, std::memory_order_relaxed);
            spdlog::debug("madvise MADV_DONTNEED failed in SystemFree: ptr={}, size={}", ptr, size);
        }
        if (HugePageCache::GetInstance().Put(ptr)) {
            return;
        }
    }
    SafeMunmap(ptr, size);
}

void PageAllocator::ReleaseHugePageCache() {
    HugePageCache::GetInstance().ReleaseAllForTesting();
}

}// namespace aethermind
