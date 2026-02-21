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

// =========================================================================
// 测试组 3: 调用类成员函数
// =========================================================================

class Calculator {
public:
    void add(int a, int b, std::atomic<int>& res) {
        res = a + b;
    }
};

TEST(ThreadMemberTest, CallMemberFunction) {
    Calculator calc;
    std::atomic<int> result{0};

    // 语法：std::thread(&类名::函数名, 对象指针, 参数...)
    std::thread t(&Calculator::add, &calc, 10, 20, std::ref(result));
    t.join();

    EXPECT_EQ(result, 30);
}

// =========================================================================
// 测试组 4: 移动语义 (Move Semantics)
// =========================================================================
TEST(ThreadMoveTest, MoveOwnership) {
    // 1. 创建 t1
    std::thread t1([] { std::this_thread::sleep_for(10ms); });

    EXPECT_TRUE(t1.joinable());

    // 2. 移动构造：将 t1 的所有权转移给 t2
    std::thread t2 = std::move(t1);

    // 3. 验证状态
    EXPECT_FALSE(t1.joinable());// t1 变为空对象
    EXPECT_TRUE(t2.joinable()); // t2 接管了线程

    // 4. 只需 join t2
    t2.join();
}

// =========================================================================
// 测试组 5: Detach (分离线程)
// =========================================================================
TEST(ThreadDetachTest, DetachExecution) {
    std::atomic<bool> flag{false};

    {
        std::thread t([&flag]() {
            std::this_thread::sleep_for(50ms);
            flag = true;
        });

        EXPECT_TRUE(t.joinable());

        // 分离线程：让它在后台跑
        t.detach();

        EXPECT_FALSE(t.joinable());// detach 后不可再 join
        // t 在这里析构，但不会导致 terminate，因为已经 detach 了
    }

    // 此时 t 对象已销毁，但后台线程可能还在跑
    // 为了测试演示，我们主线程等它一会儿
    // 注意：实际开发中慎用 detach，因为很难确定它什么时候结束
    int retries = 0;
    while (!flag && retries < 10) {
        std::this_thread::sleep_for(20ms);
        retries++;
    }

    EXPECT_TRUE(flag);
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
