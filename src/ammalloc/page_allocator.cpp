//
// Created by richard on 2/7/26.
//
#include "ammalloc/page_allocator.h"
#include "utils/logging.h"

#include <cstring>
#include <sys/mman.h>

namespace aethermind {

namespace {

class HugePageCache {
public:
    static HugePageCache& GetInstance() {
        static HugePageCache instance;
        return instance;
    }

    ~HugePageCache() {
#ifdef AM_DEBUG
        ReleaseAll();// 测试模式下清理，为了让 ASan 通过
#endif
        // 生产模式下什么都不做
        // 依赖 OS 回收，避免析构顺序 crash
    }

    void* Get() {
        std::lock_guard lock(mtx_);
        if (cache_.empty()) {
            return nullptr;
        }

        void* ptr = cache_.back();
        cache_.pop_back();
        return ptr;
    }

    bool Put(void* ptr) {
        std::lock_guard lock(mtx_);
        static size_t huge_page_cache_size =
                RuntimeConfig::GetInstance().HugePageCacheSize();
        if (cache_.size() >= huge_page_cache_size) {
            return false;
        }
        cache_.push_back(ptr);
        return true;
    }

    void ReleaseAll() {
        std::lock_guard lock(mtx_);
        while (!cache_.empty()) {
            void* ptr = cache_.back();
            cache_.pop_back();
            munmap(ptr, SystemConfig::HUGE_PAGE_SIZE);
        }
    }

private:
    HugePageCache() {
        cache_.reserve(16);
    }

    std::mutex mtx_;
    std::vector<void*> cache_;
};

}// namespace

#ifdef PAGE_ALLOCATOR_TEST
std::atomic<bool> g_mock_huge_alloc_fail{false};
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

    if (const int ret = munmap(ptr, size); ret != 0) {
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

void* PageAllocator::AllocNormalPage(size_t size, bool is_fallback) {
    if (!is_fallback) {
        stats_.normal_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }

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

void* PageAllocator::AllocHugePageFallback(size_t size) {
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

/**
 * @brief 乐观的大页分配策略
 * 策略：
 * 1. 先尝试直接申请 size 大小（赌它刚好对齐）。
 * 2. 如果不对齐，munmap 掉，再走 "Over-allocate" 流程。
 * 收益：在内存碎片较少或 OS 激进 THP 时，避免了 2MB 的 VMA 浪费和额外的 munmap 调用。
 */
void* PageAllocator::AllocHugePage(size_t size) {
#ifdef PAGE_ALLOCATOR_TEST
    if (g_mock_huge_alloc_fail.load(std::memory_order_relaxed)) {
        stats_.huge_alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
#endif

    stats_.huge_alloc_count.fetch_add(1, std::memory_order_relaxed);
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
    return AllocHugePageFallback(size);
}

void* PageAllocator::SystemAlloc(size_t page_num) {
    if (page_num == 0) {
        spdlog::warn("SystemAlloc called with page_num=0");
        return nullptr;
    }

    const size_t size = page_num << SystemConfig::PAGE_SHIFT;

    // clang-format off
    if (size < (SystemConfig::HUGE_PAGE_SIZE >> 1)) AM_LIKELY {
        void* ptr = AllocNormalPage(size, false);
        if (!ptr) {
            stats_.alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        }
        return ptr;
    }
    // clang-format on

    // 如果请求的大小正好是大页大小，尝试走缓存
    // 只有标准大页大小才值得缓存，不定长的小内存直接走 mmap
    if (size == SystemConfig::HUGE_PAGE_SIZE) {
        if (void* ptr = HugePageCache::GetInstance().Get()) {
            stats_.huge_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
            return ptr;
        }
        // stats_.huge_alloc_count.fetch_add(1, std::memory_order_relaxed);
        // stats_.huge_alloc_success.fetch_add(1, std::memory_order_relaxed);
        // stats_.huge_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
        stats_.huge_cache_miss_count.fetch_add(1, std::memory_order_relaxed);
    }

    auto* ptr = AllocHugePage(size);
    // 大页分配失败，统计降级次数并降级到普通页
    if (ptr == nullptr) {
        stats_.huge_fallback_to_normal_count.fetch_add(1, std::memory_order_relaxed);
        // 标记为降级请求，不统计normal_alloc_count
        ptr = AllocNormalPage(size, true);
        if (ptr == nullptr) {
            stats_.alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    return ptr;
}

void PageAllocator::SystemFree(void* ptr, size_t page_num) {
    if (!ptr || page_num == 0) {
        return;
    }

    const size_t size = page_num << SystemConfig::PAGE_SHIFT;
    stats_.free_count.fetch_add(1, std::memory_order_relaxed);
    stats_.free_bytes.fetch_add(size, std::memory_order_relaxed);
    // 如果释放的是标准大页，尝试放入缓存
    // 调用 madvise(ptr, size, MADV_FREE / MADV_DONTNEED)
    // 告诉 OS 这块物理内存暂时不用了，但在缓存期间保留虚拟地址
    // 这样既节省物理内存，又避免了 munmap 的 VMA 销毁开销。
    // 下次复用时只有缺页中断开销，没有 mmap 锁开销
    if (size == SystemConfig::HUGE_PAGE_SIZE) {
        madvise(ptr, size, MADV_DONTNEED);
        if (HugePageCache::GetInstance().Put(ptr)) {
            return;
        }
    }
    SafeMunmap(ptr, size);
}

void PageAllocator::ReleaseHugePageCache() {
    HugePageCache::GetInstance().ReleaseAll();
}

}// namespace aethermind