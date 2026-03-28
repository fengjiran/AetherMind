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
        scavenge_thread_ = std::jthread([this](std::stop_token stoken) {
            ScavengeLoop(stoken);
        });
        spdlog::debug("PageHeapScavenger thread started.");
    }
}

void PageHeapScavenger::Stop() {
    if (scavenge_thread_.joinable()) {
        // Send stop signal, this will wake up the thread waiting in cv_.wait_for
        scavenge_thread_.request_stop();
        // Explicitly wait for the thread to stop, although jthread destructor will also join
        scavenge_thread_.join();
        spdlog::debug("PageHeapScavenger thread stopped.");
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
        auto& span_list = page_cache.GetSpanList(i);
        {
            std::lock_guard<std::mutex> lock(page_cache.GetMutex());
            auto* cur = span_list.begin();
            while (cur != span_list.end()) {
                auto* next = cur->next;
                if (cur->IsUsed()) {
                    spdlog::error("Scavenger: used span {} in free list.",
                                  static_cast<void*>(cur));
                    cur = next;
                    continue;
                }

                if (!cur->IsCommitted()) {
                    cur = next;
                    continue;
                }

                if (now - cur->last_used_time_ms >= kIdleThresholdMs) {
                    span_list.erase(cur);
                    // Mark as "in use" to prevent ReleaseSpan from merging
                    cur->SetUsed(true);

                    if (!head) {
                        head = cur;
                        tail = cur;
                    } else {
                        tail->next = cur;// NOLINT
                        tail = cur;
                        tail->next = nullptr;
                    }
                }
                cur = next;
            }
        }// PageCache Unlock

        // Perform time-consuming madvise outside the lock
        auto* cur = head;
        while (cur) {
            void* start_ptr = cur->GetPageBaseAddr();
            size_t size = cur->page_num << SystemConfig::PAGE_SHIFT;
            if (madvise(start_ptr, size, MADV_DONTNEED) == 0) {
                cur->SetCommitted(false);
                release_bytes += size;
            } else {
                spdlog::warn("madvise MADV_DONTNEED failed for span {}",
                             static_cast<void*>(cur));
            }
            cur = cur->next;
        }

        // Once again lock to put back to PageCache
        if (head) {
            std::lock_guard<std::mutex> lock(page_cache.GetMutex());
            cur = head;
            while (cur) {
                auto* next = cur->next;
                // Mark as unused to allow merging
                cur->SetUsed(false);
                cur->last_used_time_ms = GetCurrentTimeMs();
                span_list.push_back(cur);
                cur = next;
            }
        }
    }

    if (release_bytes > 0) {
        spdlog::debug("Scavenger released {} MB physical memory.", release_bytes >> 20);
    }
}

}// namespace aethermind
