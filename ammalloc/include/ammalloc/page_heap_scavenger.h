//
// Created by richard on 3/2/26.
//

#ifndef AETHERMIND_AMMALLOC_PAGE_HEAP_SCAVENGER_H
#define AETHERMIND_AMMALLOC_PAGE_HEAP_SCAVENGER_H

#include <condition_variable>
#include <stop_token>
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

    void ScavengeLoop(std::stop_token stoken);
    static void ScavengeOnePass();

    std::jthread scavenge_thread_;
    std::condition_variable_any cv_;
    std::mutex mutex_;

    // Clean-up interval
    static constexpr uint64_t kScavengeIntervalMs = 1000;
    // Idle threshold
    static constexpr uint64_t kIdleThresholdMs = 10000;
};

}// namespace aethermind

#endif//AETHERMIND_AMMALLOC_PAGE_HEAP_SCAVENGER_H
