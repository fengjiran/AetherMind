//
// Created by richard on 2/24/26.
//

#ifndef AETHERMIND_MALLOC_SPIN_LOCK_H
#define AETHERMIND_MALLOC_SPIN_LOCK_H

#include "ammalloc/common.h"
#include "macros.h"

#include <atomic>

namespace aethermind {

/**
 * @brief 高性能用户态自旋锁 (TTAS 架构)
 *
 * 适用于临界区极短的场景（如 CentralCache 的桶锁）。
 * 采用 Test-and-Test-and-Set 策略，极大减少缓存行失效 (Cache Line Bouncing)。
 */
class SpinLock {
public:
    SpinLock() noexcept = default;
    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    void lock() noexcept {
        size_t spin_cnt = 0;
        while (true) {
            // 1. Test (乐观读)
            // 先用 relaxed 读，如果锁被占用，就在本地 Cache Line 里自旋读。
            // 这避免了疯狂调用 exchange 导致总线风暴和缓存一致性流量激增。
            if (!locked_.load(std::memory_order_relaxed)) {
                // 2. Test-and-Set (尝试抢占)
                // 发现锁可能空闲，尝试用 acquire 语义抢占
                if (!locked_.exchange(true, std::memory_order_acquire)) {
                    return;// 抢占成功
                }
            }

            // 3. 退避策略 (Backoff)
            // 提示 CPU 当前处于自旋等待状态，优化流水线并降低功耗
            details::CPUPause();
            ++spin_cnt;

            // 4. 深度退避 (Yield)
            // 如果自旋次数过多（说明持有锁的线程可能被 OS 调度走了），
            // 主动让出当前 CPU 时间片，防止死等。
            // 这里的 2000 可以作为 RuntimeConfig 的配置项 (spin_count)
            if (spin_cnt > 2000) AM_UNLIKELY {
                    std::this_thread::yield();
                    spin_cnt = 0;
                }
        }
    }

    // 同样先 Test 再 Set，优化失败时的开销
    bool try_lock() noexcept {
        return !locked_.load(std::memory_order_relaxed) && !locked_.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        // Release 语义：保证临界区内的读写操作在解锁前全部完成，并对下一个获取锁的线程可见
        locked_.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> locked_{false};
};

}// namespace aethermind

#endif//AETHERMIND_MALLOC_SPIN_LOCK_H
