//
// Created by richard on 12/18/25.
//
#include "container/map.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

TEST(MapTest, basic) {
    Map<int, int> dict;
    EXPECT_TRUE(dict.empty());
    EXPECT_EQ(dict.slots(), 2);
    dict.insert(1, 2);
    dict.insert(2, 3);
    EXPECT_EQ(dict.size(), 2);
    EXPECT_TRUE(dict.IsSmallMap());

    EXPECT_EQ(dict[1], 2);
    EXPECT_EQ(dict[2], 3);

    dict.insert(3, 4);
    dict.insert(4, 5);
    EXPECT_EQ(dict.size(), 4);
    EXPECT_TRUE(dict.IsSmallMap());
    EXPECT_EQ(dict.slots(), 4);

    dict.insert(5, 6);
    dict.insert(6, 7);
    EXPECT_EQ(dict.size(), 6);
    EXPECT_TRUE(!dict.IsSmallMap());
    auto it = dict.begin();
    EXPECT_EQ((*it).first, 1);
    EXPECT_EQ((*it).second, 2);
}

// 测试构造函数和赋值运算符
TEST(MAPTEST, constructors_and_assignment) {
    // 默认构造函数
    Map<int, String> map1;
    EXPECT_TRUE(map1.empty());
    EXPECT_EQ(map1.size(), 0);

    // 复制构造函数
    map1.insert(1, "one");
    map1.insert(2, "two");
    Map<int, String> map2(map1);
    EXPECT_EQ(map2.size(), 2);
    EXPECT_EQ(map2[1], "one");
    EXPECT_EQ(map2[2], "two");

    // 移动构造函数
    Map<int, String> map3(std::move(map1));
    EXPECT_TRUE(map1.empty());
    EXPECT_EQ(map3.size(), 2);
    EXPECT_EQ(map3[1], "one");

    // 复制赋值运算符
    Map<int, String> map4;
    map4 = map3;
    EXPECT_EQ(map4.size(), 2);
    EXPECT_EQ(map4[1], "one");

    // 移动赋值运算符
    Map<int, String> map5;
    map5 = std::move(map3);
    EXPECT_TRUE(map3.empty());
    EXPECT_EQ(map5.size(), 2);
    EXPECT_EQ(map5[1], "one");
}

// 测试插入和访问操作
TEST(MAPTEST, insert_and_access) {
    Map<String, int> map;

    // 插入元素
    map.insert("one", 1);
    map.insert("two", 2);
    map.insert("three", 3);

    // 测试size和empty
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map.size(), 3);

    // 测试at方法
    EXPECT_EQ(map.at("one"), 1);
    EXPECT_EQ(map.at("two"), 2);
    EXPECT_EQ(map.at("three"), 3);

    // 测试operator[]
    EXPECT_EQ(map["one"], 1);
    EXPECT_EQ(map["two"], 2);
    EXPECT_EQ(map["three"], 3);

    // 测试插入重复键（应该覆盖）
    map.insert("one", 10);
    EXPECT_EQ(map["one"], 10);
}

// 测试删除操作
TEST(MAPTEST, erase) {
    Map<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);
    map.insert(4, 40);
    map.insert(5, 50);

    EXPECT_EQ(map.size(), 5);

    // 删除一个元素
    map.erase(3);
    EXPECT_EQ(map.size(), 4);

    // 删除不存在的元素
    map.erase(10);
    EXPECT_EQ(map.size(), 4);

    // 删除所有元素
    map.erase(1);
    map.erase(2);
    map.erase(4);
    map.erase(5);
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

// 测试查找操作
TEST(MAPTEST, find) {
    Map<int, String> map;
    map.insert(1, "one");
    map.insert(2, "two");
    map.insert(3, "three");

    // 查找存在的键
    auto it = map.find(2);
    EXPECT_NE(it, map.end());
    EXPECT_EQ((*it).first, 2);
    EXPECT_EQ((*it).second, "two");

    // 查找不存在的键
    it = map.find(10);
    EXPECT_TRUE(it == map.end());
}

// 测试迭代器
TEST(MAPTEST, iterators) {
    Map<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    // 测试begin和end
    auto it = map.begin();
    EXPECT_NE(it, map.end());

    // 遍历所有元素
    int sum = 0;
    for (auto& pair: map) {
        sum += pair.second;
    }
    EXPECT_EQ(sum, 60);

    // 测试空地图的迭代器
    Map<int, int> empty_map;
    EXPECT_EQ(empty_map.begin(), empty_map.end());
}

// 测试清除操作
TEST(MAPTEST, clear) {
    Map<int, String> map;
    map.insert(1, "one");
    map.insert(2, "two");
    map.insert(3, "three");

    EXPECT_EQ(map.size(), 3);

    // 清除地图
    map.clear();
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_TRUE(map.IsSmallMap());

    // 清除后可以继续插入
    map.insert(4, "four");
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map[4], "four");
}

// 测试小地图到大地图的转换
TEST(MAPTEST, small_to_large_conversion) {
    Map<int, int> map;

    // 插入元素直到转换为大地图
    size_t small_slots = map.slots();
    EXPECT_TRUE(map.IsSmallMap());

    // 插入足够多的元素
    for (int i = 0; i < 10; ++i) {
        map.insert(i, i * 10);
    }

    EXPECT_FALSE(map.IsSmallMap());
    EXPECT_GT(map.slots(), small_slots);
}

// 测试边界条件
TEST(MAPTEST, edge_cases) {
    Map<int, int> map;

    // 单个元素
    map.insert(1, 10);
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map[1], 10);

    // 空键测试（如果支持的话）
    // 注意：这里假设int类型的0是合法键
    map.insert(0, 0);
    EXPECT_EQ(map[0], 0);

    // 大量元素
    for (int i = 0; i < 100; ++i) {
        map.insert(100 + i, 200 + i);
    }
    EXPECT_EQ(map.size(), 102);

    // 清除大量元素
    map.clear();
    EXPECT_TRUE(map.empty());
}

// 测试不同类型的键和值
TEST(MAPTEST, different_types) {
    // String作为键，int作为值
    Map<String, int> string_map;
    string_map.insert("apple", 1);
    string_map.insert("banana", 2);
    string_map.insert("cherry", 3);
    EXPECT_EQ(string_map["apple"], 1);

    // int作为键，Map作为值（嵌套Map）
    Map<int, Map<String, int>> nested_map;
    nested_map.insert(1, string_map);
    EXPECT_EQ(nested_map[1]["apple"], 1);

    // 使用const类型
    const Map<String, int> const_map = string_map;
    EXPECT_EQ(const_map.at("apple"), 1);
    EXPECT_EQ(const_map.size(), 3);
}

}// namespace