//
// Created by richard on 3/2/26.
//

#include "ammalloc/page_heap_scavenger.h"
#include "ammalloc/page_cache.h"

#include <sys/mman.h>

namespace aethermind {

void PageHeapScavenger::Start() {
    // Ensure it won't be started twice
    if (!scavenge_thread_.joinable()) {
        // std::jthread automatically passes the internal stop_token to the bound function
        scavenge_thread_ = std::jthread(&PageHeapScavenger::ScavengeLoop, this);
        spdlog::info("PageHeapScavenger thread started.");
    }
}

void PageHeapScavenger::Stop() {
    if (scavenge_thread_.joinable()) {
        // Send stop signal, this will wake up the thread waiting in cv_.wait_for
        scavenge_thread_.request_stop();
        // Explicitly wait for the thread to stop, although jthread destructor will also join
        scavenge_thread_.join();
        spdlog::info("PageHeapScavenger thread stopped.");
    }
}

void PageHeapScavenger::ScavengeLoop(std::stop_token stoken) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Keep running until stop signal is received
    while (!stoken.stop_requested()) {
        // [Core Optimization]: Interruptible sleep
        // Use condition_variable_any together with stop_token.
        // If request_stop() is called during sleep, this wait_for
        // will wake up immediately and return true.
        // If it wakes up due to normal timeout, it returns false.
        bool stop_requested = cv_.wait_for(lock, stoken,
                                           std::chrono::milliseconds(kScavengeIntervalMs),
                                           [&stoken] { return stoken.stop_requested(); });
        if (stop_requested) {
            break;
        }

        // Sleep end, ready to work.
        // Unlock first to prevent blocking other threads using cv_
        lock.unlock();
        ScavengeOnePass();

        // After working, lock again to sleep again in the next round
        lock.lock();
    }
}

void PageHeapScavenger::ScavengeOnePass() {
    auto now = GetCurrentTimeMs();
    auto& page_cache = PageCache::GetInstance();
    size_t release_bytes = 0;

    for (size_t i = PageConfig::MAX_PAGE_NUM; i > 0; --i) {
        Span* head = nullptr;
        Span* tail = nullptr;
        {
            std::lock_guard<std::mutex> lock(page_cache.GetMutex());
            auto* cur = page_cache.span_lists_[i].begin();
            while (cur != page_cache.span_lists_[i].end()) {
                auto* next = cur->next;
                if (!cur->is_committed) {
                    cur = next;
                    continue;
                }

                if (now - cur->last_used_time_ms >= kIdleThresholdMs) {
                    page_cache.span_lists_[i].erase(cur);
                    if (!head) {
                        head = cur;
                        tail = cur;
                    } else {
                        tail->next = cur;// NOLINT
                        tail = cur;
                    }
                }
                cur = next;
            }
        }// PageCache Unlock

        // 在锁外执行耗时的 madvise
        auto* cur = head;
        while (cur) {
            void* star_ptr = cur->GetStartAddr();
            size_t size = cur->page_num << SystemConfig::PAGE_SHIFT;
            if (madvise(star_ptr, size, MADV_DONTNEED) == 0) {
                cur->is_committed = false;
                release_bytes += size;
            }
            cur = cur->next;
        }

        // 再次加锁，挂回 PageCache
        if (head) {
            std::lock_guard<std::mutex> lock(page_cache.GetMutex());
            cur = head;
            while (cur) {
                auto* next = cur->next;
                page_cache.span_lists_[i].push_back(cur);
                cur = next;
            }
        }
    }

    if (release_bytes > 0) {
        spdlog::debug("Scavenger released {} MB physical memory.", release_bytes >> 20);
    }
}

}// namespace aethermind
