//
// Created by richard on 1/29/26.
//
#include <atomic>
#include <chrono>
#include <functional>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

// 对比 std::thread，验证 jthread 在作用域结束时会自动等待线程完成
TEST(JThreadBasicTest, AutoJoinOnDestruction) {
    std::atomic<bool> task_completed{false};

    {
        // 创建一个 jthread
        std::jthread t([&task_completed] {
            std::this_thread::sleep_for(50ms);
            task_completed = true;
        });

        // 离开作用域时：
        // 1. t 的析构函数被调用
        // 2. 检查是否 joinable()
        // 3. 自动调用 request_stop()
        // 4. 自动调用 join() -> 阻塞直到 lambda 执行完毕
    }

    // 断言：由于自动 join，这里一定是 true
    EXPECT_TRUE(task_completed);
}

// 测试显式调用 request_stop()
TEST(JThreadStopTest, ExplicitStopRequest) {
    std::atomic<bool> stopped_gracefully{false};

    // 注意：lambda 的第一个参数接受 std::stop_token
    std::jthread t([&stopped_gracefully](std::stop_token stoken) {
        // 模拟一个循环任务
        while (!stoken.stop_requested()) {
            std::this_thread::sleep_for(1ms);
        }
        stopped_gracefully = true;
    });

    // 让线程跑一会儿
    std::this_thread::sleep_for(20ms);

    // 主线程显式请求停止
    EXPECT_FALSE(stopped_gracefully);// 此时应该还在跑
    t.request_stop();

    // 等待线程结束
    t.join();

    EXPECT_TRUE(stopped_gracefully);
}

}// namespace
