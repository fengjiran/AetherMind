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

// =========================================================================
// 测试组 1: 基础启动与 Lambda
// =========================================================================
TEST(ThreadBasicTest, LambdaExecution) {
    std::atomic<bool> is_executed{false};

    // 1. 创建线程，执行 Lambda
    std::thread t([&is_executed]() {
        // 模拟一些工作
        std::this_thread::sleep_for(10ms);
        is_executed = true;
    });

    // 断言：在 join 之前，我们无法确定它是否执行完（取决于调度）
    EXPECT_TRUE(t.joinable());

    // 2. 汇合 (Join)
    t.join();

    // 3. 验证结果
    EXPECT_FALSE(t.joinable());// join 之后变为不可 join
    EXPECT_TRUE(is_executed);  // 线程任务已完成
}

// =========================================================================
// 测试组 2: 参数传递 (值传递 vs 引用传递)
// =========================================================================
// 辅助函数：按值接收
void thread_by_value(int val, std::atomic<int>& output) {
    val += 100;// 修改的是副本
    output = val;
}

// 辅助函数：按引用接收
void thread_by_ref(int& val) {
    val = 100;// 修改的是本体
}

TEST(ThreadParamTest, PassByValue) {
    int input = 10;
    std::atomic<int> result{0};

    // input 按值拷贝传递给线程，result 用 std::ref 传引用以便回写结果
    std::thread t(thread_by_value, input, std::ref(result));
    t.join();

    EXPECT_EQ(result, 110);
    EXPECT_EQ(input, 10);// input 保持不变，证明是拷贝传递
}

TEST(ThreadParamTest, PassByRef) {
    int input = 10;
    std::thread t(thread_by_ref, std::ref(input));
    if (t.joinable()) {
        t.join();
    }
}

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
