//
// Created by richard on 3/2/26.
//

#include "ammalloc/page_heap_scavenger.h"
#include "ammalloc/page_cache.h"

#include <sys/mman.h>

#include <utility>

namespace aethermind {

void PageHeapScavenger::Start() {
    // 确保不会重复启动
    if (!scavenge_thread_.joinable()) {
        // std::jthread 会自动将内部的 stop_token 传递给绑定的函数
        scavenge_thread_ = std::jthread([this](std::stop_token stoken) {
            ScavengeLoop(std::move(stoken));
        });
        spdlog::info("PageHeapScavenger thread started.");
    }
}

void PageHeapScavenger::Stop() {
    if (scavenge_thread_.joinable()) {
        // 发送停止信号，这会唤醒正在 cv_.wait_for 中沉睡的线程
        scavenge_thread_.request_stop();
        // 显式等待线程结束 (虽然 jthread 析构时也会自动 join，但显式调用语义更清晰)
        scavenge_thread_.join();
        spdlog::info("PageHeapScavenger thread stopped.");
    }
}

void PageHeapScavenger::ScavengeLoop(std::stop_token stoken) {
    std::unique_lock<std::mutex> lock(mutex_);

    // 只要没有收到停止请求，就继续循环
    while (!stoken.stop_requested()) {
        // 【核心优化】：可中断的睡眠
        // 使用 condition_variable_any 配合 stop_token。
        // 如果在睡眠期间调用了 request_stop()，这个 wait_for 会立刻被唤醒并返回 true。
        // 如果是正常超时醒来，返回 false。
        bool stop_requested = cv_.wait_for(lock, stoken,
                                           std::chrono::milliseconds(kScavengeIntervalMs),
                                           [&stoken] { return stoken.stop_requested(); });
        if (stop_requested) {
            break;
        }

        // 睡眠结束，准备干活。先解锁，防止阻塞其他可能使用 cv_ 的逻辑
        lock.unlock();
        ScavengeOnePass();

        // 干完活重新加锁，准备进入下一次睡眠
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
