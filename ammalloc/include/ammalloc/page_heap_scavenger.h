//
// Created by richard on 3/2/26.
//

#ifndef AETHERMIND_AMMALLOC_PAGE_HEAP_SCAVENGER_H
#define AETHERMIND_AMMALLOC_PAGE_HEAP_SCAVENGER_H

#include <atomic>
#include <chrono>
#include <thread>

namespace aethermind {

class PageHeapScavenger {
public:
    static PageHeapScavenger& GetInstance() {
        static PageHeapScavenger instance;
        return instance;
    }

    PageHeapScavenger(const PageHeapScavenger&) = delete;
    PageHeapScavenger& operator=(const PageHeapScavenger&) = delete;

    void Start();
    void Stop();

    ~PageHeapScavenger() {
        Stop();
    }

private:
    PageHeapScavenger() = default;

    void ScavengeLoop();
    void ScavengeOnePass();

    std::jthread scavenge_thread_;
    std::atomic<bool> is_running_{false};

    // 清理间隔
    static constexpr uint64_t kScavengeIntervalMs = 500;
    // 闲置阈值
    static constexpr uint64_t kIdleThresholdMs = 5000;
};

}// namespace aethermind

#endif//AETHERMIND_AMMALLOC_PAGE_HEAP_SCAVENGER_H
