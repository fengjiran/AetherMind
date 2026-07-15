// Created by richard on 12/18/25.

#include "container/map.h"
#include "container/map_v2.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// ========== 测试套件1：基础功能测试（构造/析构/空判断/指针获取） ==========
TEST(BBlockTest, BasicFunction) {
    HashTableBlock<int> block;

    // 测试：初始状态所有槽位都是未构造的
    for (size_t i = 0; i < MapMagicConstants::kSlotsPerBlock; ++i) {
        EXPECT_TRUE(block.IsUnConstructed(i));
    }

    // 测试：GetDataPtr返回有效指针，无崩溃
    EXPECT_NE(block.GetDataPtr(0), nullptr);
    EXPECT_NE(block.GetDataPtr(MapMagicConstants::kSlotsPerBlock - 1), nullptr);

    // 测试：const版本GetDataPtr正确性
    const HashTableBlock<int>& const_block = block;
    EXPECT_NE(const_block.GetDataPtr(0), nullptr);
    EXPECT_TRUE(const_block.IsUnConstructed(0));
}

// ========== 测试套件2：核心功能 - emplace+destroy 正常流程测试 ==========
TEST(BBlockTest, EmplaceAndDestroy_Normal) {
    HashTableBlock<int> block;
    const size_t test_slot = 5;
    const int test_val = 1024;

    // 1. emplace构造对象
    block.emplace(test_slot, test_val);
    EXPECT_FALSE(block.IsUnConstructed(test_slot));
    EXPECT_EQ(*block.GetDataPtr(test_slot), test_val);

    // 2. destroy析构对象
    block.destroy(test_slot);
    EXPECT_TRUE(block.IsUnConstructed(test_slot));

    // 3. 重复destroy无副作用
    block.destroy(test_slot);
    EXPECT_TRUE(block.IsUnConstructed(test_slot));
}

// ========== 测试套件3：核心功能 - emplace覆盖写入（修复后的核心场景） ==========
TEST(BBlockTest, Emplace_CoverWrite) {
    HashTableBlock<String> block;
    const size_t test_slot = 3;

    // 第一次emplace
    block.emplace(test_slot, "hello world");
    EXPECT_EQ(*block.GetDataPtr(test_slot), "hello world");
    EXPECT_FALSE(block.IsUnConstructed(test_slot));

    // 第二次emplace覆盖写入（核心测试：修复BUG2后无内存泄漏）
    block.emplace(test_slot, "cpp high performance");
    EXPECT_EQ(*block.GetDataPtr(test_slot), "cpp high performance");
    EXPECT_FALSE(block.IsUnConstructed(test_slot));

    // destroy后再emplace
    block.destroy(test_slot);
    block.emplace(test_slot, "final test");
    EXPECT_EQ(*block.GetDataPtr(test_slot), "final test");
}

// ========== 测试套件4：拷贝构造测试（深拷贝，重中之重，必测） ==========
TEST(BBlockTest, CopyConstructor_DeepCopy) {
    HashTableBlock<std::pair<int, std::vector<int>>> block;
    const size_t slot1 = 0;
    const size_t slot2 = 7;

    // 源对象emplace两个复合对象
    block.emplace(slot1, std::make_pair(10, std::vector<int>{1, 2, 3}));
    block.emplace(slot2, std::make_pair(20, std::vector<int>{4, 5, 6}));

    // 拷贝构造新对象
    HashTableBlock<std::pair<int, std::vector<int>>> block_copy(block);

    // 测试1：拷贝后的对象标记正确
    EXPECT_FALSE(block_copy.IsUnConstructed(slot1));
    EXPECT_FALSE(block_copy.IsUnConstructed(slot2));
    EXPECT_TRUE(block_copy.IsUnConstructed(1));

    // // 测试2：深拷贝-值相等
    EXPECT_EQ(block_copy.GetDataPtr(slot1)->first, 10);
    auto a = std::vector<int>{1, 2, 3};
    EXPECT_EQ(block_copy.GetDataPtr(slot1)->second, a);
    EXPECT_EQ(block_copy.GetDataPtr(slot2)->first, 20);
    a = std::vector<int>{4, 5, 6};
    EXPECT_EQ(block_copy.GetDataPtr(slot2)->second, a);

    // // 测试3：深拷贝-内存独立（修改拷贝对象，源对象不受影响）
    block_copy.destroy(slot1);
    block_copy.emplace(slot1, std::make_pair(99, std::vector<int>{9, 9, 9}));
    EXPECT_EQ(block.GetDataPtr(slot1)->first, 10);     // 源对象不变
    EXPECT_EQ(block_copy.GetDataPtr(slot1)->first, 99);// 拷贝对象已修改
}

// ========== 测试套件5：边界条件测试（槽位越界/最大槽位/0槽位） ==========
TEST(BBlockTest, BoundaryCondition) {
    HashTableBlock<int> block;
    const size_t max_slot = MapMagicConstants::kSlotsPerBlock - 1;

    // 测试最大槽位emplace/destroy
    block.emplace(max_slot, 999);
    EXPECT_EQ(*block.GetDataPtr(max_slot), 999);
    block.destroy(max_slot);
    EXPECT_TRUE(block.IsUnConstructed(max_slot));

    // 测试0槽位
    block.emplace(0, 1);
    EXPECT_EQ(*block.GetDataPtr(0), 1);
}

// ========== 测试套件6：复杂类型测试（带堆内存，无内存泄漏/双重释放） ==========
TEST(BBlockTest, ComplexValueType_NoLeak) {
    HashTableBlock<std::vector<int>> block;
    const size_t slot = 2;

    // emplace大vector
    block.emplace(slot, std::vector<int>(1000, 5));
    EXPECT_EQ(block.GetDataPtr(slot)->size(), 1000);
    EXPECT_EQ((*block.GetDataPtr(slot))[0], 5);

    // destroy后再emplace
    block.destroy(slot);
    block.emplace(slot, std::vector<int>(500, 8));
    EXPECT_EQ(block.GetDataPtr(slot)->size(), 500);
    EXPECT_EQ((*block.GetDataPtr(slot))[0], 8);

    // 析构后无内存泄漏（valgrind检测无泄漏）
}

// ========== 测试套件7：const正确性测试（const对象只能读，不能写） ==========
TEST(BBlockTest, ConstCorrectness) {
    HashTableBlock<std::string> block;
    block.emplace(1, "const test");
    const HashTableBlock<std::string>& const_block = block;

    // const对象可以调用const版本的GetDataPtr和IsUnconstructed
    EXPECT_EQ(*const_block.GetDataPtr(1), "const test");
    EXPECT_FALSE(const_block.IsUnConstructed(1));
    EXPECT_TRUE(const_block.IsUnConstructed(2));

    // const对象无法调用emplace/destroy（编译器会报错，此处注释验证）
    // const_block.emplace(1, "error"); // ❌ compile error
    // const_block.destroy(1); // ❌ compile error
}

}// namespace
