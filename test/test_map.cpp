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
    EXPECT_EQ(it->first, 1);
    EXPECT_EQ(it->second, 2);
    std::unordered_map<int, int> x;
    auto res = x.insert({1, 1});
    // auto res1 = x.insert_or_assign(1, 2);
    auto res1 = x.insert({1, 2});
    EXPECT_TRUE(!res1.second);
    EXPECT_EQ(res1.first->first, 1);
    EXPECT_EQ(x[1], 1);
}

// 测试构造函数和赋值运算符
TEST(MapTest, constructors_and_assignment) {
    // 默认构造函数
    Map<int, String> map1;
    EXPECT_TRUE(map1.empty());
    EXPECT_EQ(map1.size(), 0);
    EXPECT_TRUE(map1.unique());

    // 复制构造函数
    map1.insert(1, "one");
    map1.insert(2, "two");
    Map<int, String> map2(map1);
    EXPECT_EQ(map2.size(), 2);
    EXPECT_EQ(map2[1], "one");
    EXPECT_EQ(map2[2], "two");
    EXPECT_EQ(map2.use_count(), 2);
    map2.insert(3, "three");
    EXPECT_TRUE(map2.unique());

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
    EXPECT_EQ(map3.use_count(), 2);

    // 移动赋值运算符
    Map<int, String> map5;
    map5 = std::move(map3);
    EXPECT_TRUE(map3.empty());
    EXPECT_EQ(map5.size(), 2);
    EXPECT_EQ(map5[1], "one");
}

// 测试插入和访问操作
TEST(MapTest, insert_and_access) {
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
    EXPECT_EQ(map["one"], 1);
}

// 测试删除操作
TEST(MapTest, erase) {
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
TEST(MapTest, find) {
    Map<int, String> map;
    map.insert(1, "one");
    map.insert(2, "two");
    map.insert(3, "three");

    // 查找存在的键
    auto it = map.find(2);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->first, 2);
    EXPECT_EQ(it->second, "two");

    // 查找不存在的键
    it = map.find(10);
    EXPECT_TRUE(it == map.end());
}

// 测试迭代器
TEST(MapTest, iterators) {
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
        sum += static_cast<int>(pair.second);
    }
    EXPECT_EQ(sum, 60);

    // 测试空地图的迭代器
    Map<int, int> empty_map;
    EXPECT_EQ(empty_map.begin(), empty_map.end());
}

// 测试清除操作
TEST(MapTest, clear) {
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
TEST(MapTest, small_to_large_conversion) {
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
TEST(MapTest, edge_cases) {
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
TEST(MapTest, different_types) {
    // String作为键，int作为值
    Map<String, int> string_map;
    string_map.insert("apple", 1);
    string_map.insert("banana", 2);
    string_map.insert("cherry", 3);
    EXPECT_EQ(string_map["apple"], 1);

    // int作为键，Map作为值（嵌套Map）
    Map<int, Map<String, int>> nested_map;
    nested_map.insert(1, string_map);
    // EXPECT_EQ(nested_map[1]["apple"], 1);

    // 使用const类型
    const Map<String, int> const_map = string_map;
    EXPECT_EQ(const_map.at("apple"), 1);
    EXPECT_EQ(const_map.size(), 3);
}

// 测试所有insert方法
TEST(MapInsertTest, all_insert_methods) {
    Map<int, int> map;

    // 1. 测试键值对插入 insert(key_type, mapped_type)
    {
        auto [it, success] = map.insert(1, 10);
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, 1);
        EXPECT_EQ(it->second, 10);
        EXPECT_EQ(map.size(), 1);
    }

    // 2. 测试值类型插入 insert(const value_type&)
    {
        Map<int, int>::value_type pair(2, 20);
        auto [it, success] = map.insert(pair);
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, 2);
        EXPECT_EQ(it->second, 20);
        EXPECT_EQ(map.size(), 2);
    }

    // 3. 测试移动插入 insert(value_type&&)
    {
        auto [it, success] = map.insert(Map<int, int>::value_type(3, 30));
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, 3);
        EXPECT_EQ(it->second, 30);
        EXPECT_EQ(map.size(), 3);
    }

    // 4. 测试可构造对插入 insert(Pair&&)
    {
        struct CustomPair {
            int first = 4;
            int second = 40;
            operator Map<int, int>::value_type() const {
                return {first, second};
            }
        };
        CustomPair custom_pair;
        auto [it, success] = map.insert(custom_pair);
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, 4);
        EXPECT_EQ(it->second, 40);
        EXPECT_EQ(map.size(), 4);

        // 测试右值版本
        auto [it2, success2] = map.insert(custom_pair);
        EXPECT_EQ(it2->first, 4);
        EXPECT_EQ(it2->second, 40);
        EXPECT_FALSE(success2);  // 应该插入失败，因为键4已经存在
        EXPECT_EQ(map.size(), 4);// 大小应该保持不变
    }

    // 5. 测试范围插入 insert(Iter first, Iter last)
    {
        std::vector<Map<int, int>::value_type> pairs = {{5, 50}, {6, 60}, {7, 70}};
        map.insert(pairs.begin(), pairs.end());
        EXPECT_EQ(map.size(), 7);
        EXPECT_EQ(map[5], 50);
        EXPECT_EQ(map[6], 60);
        EXPECT_EQ(map[7], 70);
    }

    // 6. 测试初始化列表插入 insert(std::initializer_list<value_type>)
    {
        map.insert({{8, 80}, {9, 90}, {10, 100}});
        EXPECT_EQ(map.size(), 10);
        EXPECT_EQ(map[8], 80);
        EXPECT_EQ(map[9], 90);
        EXPECT_EQ(map[10], 100);
    }
}


// 测试插入重复键
TEST(MapInsertTest, insert_duplicate_keys) {
    Map<int, int> map;

    // 插入第一个元素
    auto [it1, success1] = map.insert(1, 10);
    EXPECT_TRUE(success1);
    EXPECT_EQ(map.size(), 1);

    // 插入重复键，应该失败
    auto [it2, success2] = map.insert(1, 20);
    EXPECT_FALSE(success2);
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(it2->second, 10);// 应该返回已存在的元素

    // 使用不同的insert方法测试重复键
    auto [it3, success3] = map.insert(Map<int, int>::value_type(1, 30));
    EXPECT_FALSE(success3);
    EXPECT_EQ(it3->second, 10);

    auto [it4, success4] = map.insert(std::move(Map<int, int>::value_type(1, 40)));
    EXPECT_FALSE(success4);
    EXPECT_EQ(it4->second, 10);
}

// 测试COW机制下的插入
TEST(MapInsertTest, insert_with_COW) {
    Map<int, int> map1;
    map1.insert(1, 10);
    map1.insert(2, 20);

    // 创建共享副本
    Map<int, int> map2 = map1;
    EXPECT_EQ(map1.use_count(), 2);
    EXPECT_EQ(map2.use_count(), 2);

    // 在副本上插入，应该触发COW
    auto [it, success] = map2.insert(3, 30);
    EXPECT_TRUE(success);
    EXPECT_EQ(map2.size(), 3);
    EXPECT_EQ(map1.size(), 2);// 原map应该保持不变
    EXPECT_EQ(map1.use_count(), 1);
    EXPECT_EQ(map2.use_count(), 1);
}

// 测试小地图到大地图转换时的插入
TEST(MapInsertTest, insert_small_to_large_conversion) {
    Map<int, int> map;
    EXPECT_TRUE(map.IsSmallMap());

    // 插入足够多的元素，触发从小地图到大地图的转换
    for (int i = 0; i < 10; ++i) {
        auto [it, success] = map.insert(i, i * 10);
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, i);
        EXPECT_EQ(it->second, i * 10);
    }

    EXPECT_FALSE(map.IsSmallMap());
    EXPECT_EQ(map.size(), 10);

    // 在大地图上继续插入
    auto [it, success] = map.insert(10, 100);
    EXPECT_TRUE(success);
    EXPECT_EQ(map.size(), 11);
}

// 测试不同类型的键和值
TEST(MapInsertTest, insert_different_types) {
    // String作为键，int作为值
    Map<String, int> string_map;
    auto [it1, success1] = string_map.insert("apple", 1);
    EXPECT_TRUE(success1);
    EXPECT_EQ(it1->first, "apple");
    EXPECT_EQ(it1->second, 1);

    // 使用移动语义插入String键
    String key = "banana";
    auto [it2, success2] = string_map.insert(std::move(key), 2);
    EXPECT_TRUE(success2);
    EXPECT_EQ(it2->first, "banana");
    EXPECT_EQ(it2->second, 2);

    // 测试复杂值类型
    Map<int, std::vector<int>> complex_map;
    std::vector<int> vec = {1, 2, 3};
    auto [it3, success3] = complex_map.insert(1, vec);
    EXPECT_TRUE(success3);
    EXPECT_EQ(static_cast<std::vector<int>>(it3->second), vec);

    // 测试初始化列表插入复杂类型
    complex_map.insert({{2, {4, 5, 6}}, {3, {7, 8, 9}}});
    EXPECT_EQ(complex_map.size(), 3);
}

// 测试范围插入的各种迭代器
TEST(MapInsertTest, insert_range_iterators) {
    Map<int, int> map;

    // 使用vector迭代器
    std::vector<std::pair<int, int>> vec = {{1, 10}, {2, 20}, {3, 30}};
    map.insert(vec.begin(), vec.end());
    EXPECT_EQ(map.size(), 3);

    // 使用数组迭代器
    std::pair<int, int> arr[] = {{4, 40}, {5, 50}, {6, 60}};
    map.insert(std::begin(arr), std::end(arr));
    EXPECT_EQ(map.size(), 6);

    // 使用Map的迭代器
    Map<int, int> map2;
    map2.insert(map.begin(), map.end());
    EXPECT_EQ(map2.size(), 6);

    // 使用const迭代器
    Map<int, int> map3;
    const Map<int, int>& const_map = map;
    map3.insert(const_map.begin(), const_map.end());
    EXPECT_EQ(map3.size(), 6);
}

// 测试插入空范围和单元素范围
TEST(MapInsertTest, insert_edge_cases) {
    Map<int, int> map;

    // 插入空范围
    std::vector<std::pair<int, int>> empty_vec;
    map.insert(empty_vec.begin(), empty_vec.end());
    EXPECT_TRUE(map.empty());

    // 插入单元素
    std::vector<std::pair<int, int>> single_vec = {{1, 10}};
    map.insert(single_vec.begin(), single_vec.end());
    EXPECT_EQ(map.size(), 1);

    // 插入空初始化列表
    map.insert({});
    EXPECT_EQ(map.size(), 1);

    // 插入包含重复键的初始化列表
    map.insert({{1, 20}, {2, 30}, {3, 40}});
    EXPECT_EQ(map.size(), 3);// 只有两个新元素被插入
    EXPECT_EQ(map[1], 10);   // 原元素保持不变
    EXPECT_EQ(map[2], 30);
    EXPECT_EQ(map[3], 40);
}

}// namespace

#ifdef TEST_MAP

namespace {
using namespace aethermind;


}// namespace

#endif
