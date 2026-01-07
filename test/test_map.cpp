//
// Created by richard on 12/18/25.
//
#include "container/map.h"
#include "container/map_depr.h"

#include <gtest/gtest.h>
#include <list>

namespace {
using namespace aethermind;

TEST(MapTest, basic) {
    Map<int, int> dict;
    EXPECT_TRUE(dict.empty());
    EXPECT_EQ(dict.slots(), 4);
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
    std::vector<std::pair<int, int>> data = {{1, 2}, {1, 3}, {1, 4}};
    x.insert(data.begin(), data.end());
    EXPECT_EQ(x.size(), 1);
    EXPECT_EQ(x[1], 2);
    auto [it1, success] = x.insert_or_assign(1, 3);
    EXPECT_TRUE(!success);
    EXPECT_EQ(x[1], 3);

    auto it2 = data.erase(data.begin());
    EXPECT_EQ(it2->first, 1);
    EXPECT_EQ(it2->second, 3);

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

// 测试 CreateFromRange 方法
TEST(MapTest, create_from_range) {
    // 测试从空范围创建
    std::vector<std::pair<int, int>> empty_vec;
    Map<int, int> empty_map(empty_vec.begin(), empty_vec.end());
    EXPECT_TRUE(empty_map.empty());
    EXPECT_EQ(empty_map.size(), 0);
    EXPECT_TRUE(empty_map.IsSmallMap());

    // 测试从小范围创建（<= kThreshold，应该使用SmallMapImpl）
    std::vector<std::pair<int, int>> small_vec = {{1, 10}, {2, 20}, {3, 30}};
    Map<int, int> small_map(small_vec.begin(), small_vec.end());
    EXPECT_FALSE(small_map.empty());
    EXPECT_EQ(small_map.size(), 3);
    EXPECT_TRUE(small_map.IsSmallMap());
    EXPECT_EQ(small_map[1], 10);
    EXPECT_EQ(small_map[2], 20);
    EXPECT_EQ(small_map[3], 30);

    // 测试从大范围创建（> kThreshold，应该使用DenseMapImpl）
    std::vector<std::pair<int, int>> large_vec = {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}};
    Map<int, int> large_map(large_vec.begin(), large_vec.end());
    EXPECT_FALSE(large_map.empty());
    EXPECT_EQ(large_map.size(), 5);
    EXPECT_FALSE(large_map.IsSmallMap());
    EXPECT_EQ(large_map[1], 10);
    EXPECT_EQ(large_map[2], 20);
    EXPECT_EQ(large_map[3], 30);
    EXPECT_EQ(large_map[4], 40);
    EXPECT_EQ(large_map[5], 50);

    // 测试包含重复键的范围（应该只保留第一个键）
    std::vector<std::pair<int, int>> duplicate_vec = {{1, 10}, {1, 20}, {2, 30}, {2, 40}};
    Map<int, int> duplicate_map(duplicate_vec.begin(), duplicate_vec.end());
    EXPECT_EQ(duplicate_map.size(), 2);
    EXPECT_EQ(duplicate_map[1], 10);// 应该保留第一个值
    EXPECT_EQ(duplicate_map[2], 30);// 应该保留第一个值

    // 测试从不同类型的容器创建
    std::list<std::pair<String, int>> list_data = {{"apple", 1}, {"banana", 2}, {"cherry", 3}};
    Map<String, int> list_map(list_data.begin(), list_data.end());
    EXPECT_EQ(list_map.size(), 3);
    EXPECT_EQ(list_map["apple"], 1);
    EXPECT_EQ(list_map["banana"], 2);
    EXPECT_EQ(list_map["cherry"], 3);

    // 测试从initializer_list创建（内部使用CreateFromRange）
    Map<int, int> init_map = {{1, 10}, {2, 20}, {3, 30}};
    EXPECT_EQ(init_map.size(), 3);
    EXPECT_EQ(init_map[1], 10);
    EXPECT_EQ(init_map[2], 20);
    EXPECT_EQ(init_map[3], 30);
}

// 测试CreateFromRange的边界条件
TEST(MapTest, create_from_range_edge_cases) {
    // 测试只有一个元素
    std::vector<std::pair<int, int>> single_vec = {{1, 10}};
    Map<int, int> single_map(single_vec.begin(), single_vec.end());
    EXPECT_EQ(single_map.size(), 1);
    EXPECT_EQ(single_map[1], 10);
    EXPECT_TRUE(single_map.IsSmallMap());

    // 测试刚好等于kThreshold个元素
    std::vector<std::pair<int, int>> threshold_vec = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
    Map<int, int> threshold_map(threshold_vec.begin(), threshold_vec.end());
    EXPECT_EQ(threshold_map.size(), 4);
    EXPECT_TRUE(threshold_map.IsSmallMap());// 应该还是SmallMapImpl

    // 测试刚好超过kThreshold个元素
    std::vector<std::pair<int, int>> over_threshold_vec = {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}};
    Map<int, int> over_threshold_map(over_threshold_vec.begin(), over_threshold_vec.end());
    EXPECT_EQ(over_threshold_map.size(), 5);
    EXPECT_FALSE(over_threshold_map.IsSmallMap());// 应该转换为DenseMapImpl

    // 测试负数大小（应该创建空Map）
    std::vector<std::pair<int, int>> vec = {{1, 10}, {2, 20}};
    Map<int, int> negative_map(vec.end(), vec.begin());
    EXPECT_TRUE(negative_map.empty());
    EXPECT_EQ(negative_map.size(), 0);
}

// 测试CreateFromRange与不同数据类型
TEST(MapTest, create_from_range_with_different_types) {
    // 使用String作为键
    std::vector<std::pair<String, String>> string_vec = {{"key1", "value1"}, {"key2", "value2"}};
    Map<String, String> string_map(string_vec.begin(), string_vec.end());
    EXPECT_EQ(string_map.size(), 2);
    EXPECT_EQ(string_map["key1"], "value1");
    EXPECT_EQ(string_map["key2"], "value2");

    // 使用嵌套容器
    std::vector<std::pair<int, std::vector<int>>> nested_vec = {{1, {1, 2, 3}}, {2, {4, 5, 6}}};
    // 注意：这里假设Map类支持vector<int>作为值类型
    Map<int, std::vector<int>> nested_map(nested_vec.begin(), nested_vec.end());
    EXPECT_EQ(nested_map.size(), 2);
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
        sum += pair.second;
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
    EXPECT_EQ(nested_map[1]["apple"], 1);

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

    auto [it4, success4] = map.insert(Map<int, int>::value_type(1, 40));
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
    EXPECT_EQ(it3->second, vec);

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

    // // 使用const迭代器
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

// 测试insert_or_assign方法的基本功能
TEST(MapInsertOrAssignTest, basic_functionality) {
    Map<int, int> map;

    // 1. 插入新元素（右值引用版本）
    {
        auto [it, success] = map.insert_or_assign(1, 10);
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, 1);
        EXPECT_EQ(it->second, 10);
        EXPECT_EQ(map.size(), 1);
    }

    // 2. 插入新元素（左值引用版本）
    {
        int key = 2;
        int value = 20;
        auto [it, success] = map.insert_or_assign(key, value);
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, 2);
        EXPECT_EQ(it->second, 20);
        EXPECT_EQ(map.size(), 2);
    }

    // 3. 更新已存在元素（右值引用版本）
    {
        auto [it, success] = map.insert_or_assign(1, 100);
        EXPECT_FALSE(success);
        EXPECT_EQ(it->first, 1);
        EXPECT_EQ(it->second, 100);
        EXPECT_EQ(map.size(), 2);// 大小保持不变
    }

    // 4. 更新已存在元素（左值引用版本）
    {
        int key = 2;
        int value = 200;
        auto [it, success] = map.insert_or_assign(key, value);
        EXPECT_FALSE(success);
        EXPECT_EQ(it->first, 2);
        EXPECT_EQ(it->second, 200);
        EXPECT_EQ(map.size(), 2);// 大小保持不变
    }
}

// 测试insert_or_assign与不同数据类型
TEST(MapInsertOrAssignTest, different_data_types) {
    // String作为键，int作为值
    Map<String, int> string_map;

    // 插入新元素
    auto [it1, success1] = string_map.insert_or_assign("apple", 1);
    EXPECT_TRUE(success1);
    EXPECT_EQ(it1->first, "apple");
    EXPECT_EQ(it1->second, 1);

    // 更新元素
    auto [it2, success2] = string_map.insert_or_assign("apple", 10);
    EXPECT_FALSE(success2);
    EXPECT_EQ(it2->second, 10);

    // 使用移动语义
    String key = "banana";
    auto [it3, success3] = string_map.insert_or_assign(std::move(key), 2);
    EXPECT_TRUE(success3);
    EXPECT_EQ(it3->first, "banana");

    // 复杂类型作为值
    Map<int, std::vector<int>> complex_map;
    std::vector<int> vec = {1, 2, 3};

    // 插入新元素
    auto [it4, success4] = complex_map.insert_or_assign(1, vec);
    EXPECT_TRUE(success4);
    EXPECT_EQ(it4->second, vec);

    // 更新元素
    std::vector<int> new_vec = {4, 5, 6};
    auto [it5, success5] = complex_map.insert_or_assign(1, new_vec);
    EXPECT_FALSE(success5);
    EXPECT_EQ(it5->second, new_vec);
}

// 测试COW机制下的insert_or_assign
TEST(MapInsertOrAssignTest, cow_mechanism) {
    Map<int, int> map1;
    map1.insert_or_assign(1, 10);
    map1.insert_or_assign(2, 20);

    // 创建共享副本
    Map<int, int> map2 = map1;
    EXPECT_EQ(map1.use_count(), 2);
    EXPECT_EQ(map2.use_count(), 2);

    // 在副本上更新元素，应该触发COW
    auto [it, success] = map2.insert_or_assign(1, 100);
    EXPECT_FALSE(success);
    EXPECT_EQ(map2.size(), 2);
    EXPECT_EQ(map1.size(), 2);// 原map应该保持不变
    EXPECT_EQ(map1[1], 10);   // 原map的元素保持不变
    EXPECT_EQ(map2[1], 100);  // 副本的元素被更新
    EXPECT_EQ(map1.use_count(), 1);
    EXPECT_EQ(map2.use_count(), 1);

    // 在副本上插入新元素，应该继续使用独立的资源
    auto [it2, success2] = map2.insert_or_assign(3, 30);
    EXPECT_TRUE(success2);
    EXPECT_EQ(map2.size(), 3);
    EXPECT_EQ(map1.size(), 2);// 原map应该保持不变
}

// 测试小地图到大地图转换时的insert_or_assign
TEST(MapInsertOrAssignTest, small_to_large_conversion) {
    Map<int, int> map;
    EXPECT_TRUE(map.IsSmallMap());

    // 插入足够多的元素，触发从小地图到大地图的转换
    for (int i = 0; i < 10; ++i) {
        auto [it, success] = map.insert_or_assign(i, i * 10);
        EXPECT_TRUE(success);
        EXPECT_EQ(it->first, i);
        EXPECT_EQ(it->second, i * 10);
    }

    EXPECT_FALSE(map.IsSmallMap());
    EXPECT_EQ(map.size(), 10);

    // 在大地图上更新元素
    auto [it1, success1] = map.insert_or_assign(5, 500);
    EXPECT_FALSE(success1);
    EXPECT_EQ(it1->second, 500);

    // 在大地图上插入新元素
    auto [it2, success2] = map.insert_or_assign(10, 100);
    EXPECT_TRUE(success2);
    EXPECT_EQ(map.size(), 11);
}

// 测试边界条件
TEST(MapInsertOrAssignTest, edge_cases) {
    Map<int, int> map;

    // 插入单个元素
    auto [it1, success1] = map.insert_or_assign(1, 10);
    EXPECT_TRUE(success1);
    EXPECT_EQ(map.size(), 1);

    // 空键测试（如果支持的话）
    auto [it2, success2] = map.insert_or_assign(0, 0);
    EXPECT_TRUE(success2);
    EXPECT_EQ(map[0], 0);

    // 大量元素
    for (int i = 0; i < 100; ++i) {
        map.insert_or_assign(100 + i, 200 + i);
    }
    EXPECT_EQ(map.size(), 102);

    // 更新大量元素
    for (int i = 0; i < 100; ++i) {
        map.insert_or_assign(100 + i, 300 + i);
    }
    EXPECT_EQ(map.size(), 102);
    EXPECT_EQ(map[100], 300);
    EXPECT_EQ(map[199], 399);
}

// 测试与其他方法的交互
TEST(MapInsertOrAssignTest, interaction_with_other_methods) {
    Map<int, int> map;

    // 与insert方法结合使用
    map.insert(1, 10);
    map.insert(2, 20);

    // 使用insert_or_assign更新insert插入的元素
    auto [it1, success1] = map.insert_or_assign(1, 100);
    EXPECT_FALSE(success1);
    EXPECT_EQ(it1->second, 100);

    // 与find方法结合使用
    auto it = map.find(2);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second, 20);

    // 更新后再次查找
    map.insert_or_assign(2, 200);
    it = map.find(2);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second, 200);

    // 与erase方法结合使用
    map.erase(1);
    EXPECT_EQ(map.size(), 1);

    // 在已删除的键上插入
    auto [it2, success2] = map.insert_or_assign(1, 1000);
    EXPECT_TRUE(success2);
    EXPECT_EQ(map.size(), 2);
}

// 测试不同类型的键和值
TEST(MapInsertOrAssignTest, different_key_value_types) {
    // String作为键，String作为值
    Map<String, String> string_map;
    string_map.insert_or_assign("name", "apple");
    EXPECT_EQ(string_map["name"], "apple");

    // 更新字符串值
    string_map.insert_or_assign("name", "banana");
    EXPECT_EQ(string_map["name"], "banana");

    // 混合类型
    Map<String, int> mixed_map;
    mixed_map.insert_or_assign("count", 5);
    mixed_map.insert_or_assign("value", 100);

    // 更新不同类型的值
    mixed_map.insert_or_assign("count", 10);
    EXPECT_EQ(mixed_map["count"], 10);
}

TEST(MapEraseTest, erase_by_iterator) {
    // 测试SmallMap的情况
    Map<int, int> small_map = {{1, 10}, {2, 20}, {3, 30}};
    EXPECT_TRUE(small_map.IsSmallMap());
    EXPECT_EQ(small_map.size(), 3);

    // 删除第一个元素
    auto it = small_map.begin();
    auto next_it = small_map.erase(it);
    EXPECT_EQ(small_map.size(), 2);
    EXPECT_EQ(next_it->first, 2);
    EXPECT_EQ(next_it->second, 20);
    EXPECT_FALSE(small_map.contains(1));
    EXPECT_TRUE(small_map.contains(2));
    EXPECT_TRUE(small_map.contains(3));

    // 删除中间元素
    next_it = small_map.erase(next_it);
    EXPECT_EQ(small_map.size(), 1);
    EXPECT_EQ(next_it->first, 3);
    EXPECT_EQ(next_it->second, 30);
    EXPECT_TRUE(small_map.contains(3));

    // 删除最后一个元素
    next_it = small_map.erase(next_it);
    EXPECT_TRUE(small_map.empty());
    EXPECT_EQ(small_map.size(), 0);
    EXPECT_EQ(next_it, small_map.end());

    // 测试DenseMap的情况
    Map<int, int> dense_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}};
    EXPECT_FALSE(dense_map.IsSmallMap());
    EXPECT_EQ(dense_map.size(), 5);


    // // 删除中间元素
    it = dense_map.begin();
    ++it;// 指向键2
    ++it;// 指向键3
    next_it = dense_map.erase(it);

    for (const auto& [fst, snd]: dense_map) {
        std::cout << fst << ": " << snd << std::endl;
    }
    EXPECT_EQ(dense_map.size(), 4);
    EXPECT_EQ(next_it->first, 4);
    EXPECT_EQ(next_it->second, 40);
    EXPECT_FALSE(dense_map.contains(3));

    // 删除第一个元素
    it = dense_map.begin();
    next_it = dense_map.erase(it);
    EXPECT_EQ(dense_map.size(), 3);
    EXPECT_EQ(next_it->first, 2);
    EXPECT_EQ(next_it->second, 20);
    EXPECT_FALSE(dense_map.contains(1));
}

TEST(MapEraseTest, erase_by_const_iterator) {
    Map<String, int> map = {{"one", 1}, {"two", 2}, {"three", 3}};
    EXPECT_EQ(map.size(), 3);

    // 使用const_iterator删除元素
    const Map<String, int>& const_map = map;
    auto it = const_map.find("two");
    EXPECT_NE(it, const_map.end());

    // 转换为非const_map并使用const_iterator删除
    Map<String, int> non_const_map = const_map;
    auto next_it = non_const_map.erase(it);
    EXPECT_EQ(non_const_map.size(), 2);
    EXPECT_EQ(next_it->first, "three");
    EXPECT_EQ(next_it->second, 3);
    EXPECT_FALSE(non_const_map.contains("two"));
}

TEST(MapEraseTest, erase_range) {
    // 测试SmallMap的情况
    Map<int, int> small_map = {{1, 10}, {2, 20}, {3, 30}};
    EXPECT_TRUE(small_map.IsSmallMap());
    EXPECT_EQ(small_map.size(), 3);

    // 删除所有元素
    auto it = small_map.begin();
    auto end_it = small_map.end();
    auto result = small_map.erase(it, end_it);
    EXPECT_TRUE(small_map.empty());
    EXPECT_EQ(result, small_map.end());

    // 重新插入元素，删除部分范围
    small_map.insert({1, 10});
    small_map.insert({2, 20});
    small_map.insert({3, 30});
    EXPECT_EQ(small_map.size(), 3);

    it = small_map.begin();
    ++it;// 指向键2
    result = small_map.erase(small_map.begin(), it);
    EXPECT_EQ(small_map.size(), 2);
    EXPECT_EQ(result->first, 2);
    EXPECT_EQ(small_map.contains(1), false);
    EXPECT_EQ(small_map.contains(2), true);
    EXPECT_EQ(small_map.contains(3), true);

    // 测试DenseMap的情况
    Map<int, int> dense_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}};
    EXPECT_FALSE(dense_map.IsSmallMap());
    EXPECT_EQ(dense_map.size(), 5);

    // 删除中间范围
    it = dense_map.begin();
    ++it;// 指向键2
    auto mid_it = it;
    ++mid_it;// 指向键3
    ++mid_it;// 指向键4
    result = dense_map.erase(it, mid_it);
    EXPECT_EQ(dense_map.size(), 3);
    EXPECT_EQ(result->first, 4);
    EXPECT_EQ(dense_map.contains(2), false);
    EXPECT_EQ(dense_map.contains(3), false);
    EXPECT_TRUE(dense_map.contains(1));
    EXPECT_TRUE(dense_map.contains(4));
    EXPECT_TRUE(dense_map.contains(5));

    // 删除空范围
    result = dense_map.erase(dense_map.begin(), dense_map.begin());
    EXPECT_EQ(dense_map.size(), 3);
    EXPECT_EQ(result, dense_map.begin());
}

TEST(MapEraseTest, erase_by_key) {
    // 测试SmallMap的情况
    Map<int, int> small_map = {{1, 10}, {2, 20}, {3, 30}};
    EXPECT_TRUE(small_map.IsSmallMap());
    EXPECT_EQ(small_map.size(), 3);

    // 删除存在的键
    size_t erased_count = small_map.erase(2);
    EXPECT_EQ(erased_count, 1);
    EXPECT_EQ(small_map.size(), 2);
    EXPECT_FALSE(small_map.contains(2));

    // 删除不存在的键
    erased_count = small_map.erase(4);
    EXPECT_EQ(erased_count, 0);
    EXPECT_EQ(small_map.size(), 2);

    // 删除所有键
    erased_count = small_map.erase(1);
    EXPECT_EQ(erased_count, 1);
    erased_count = small_map.erase(3);
    EXPECT_EQ(erased_count, 1);
    EXPECT_TRUE(small_map.empty());

    // 测试DenseMap的情况
    Map<int, int> dense_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}};
    EXPECT_FALSE(dense_map.IsSmallMap());
    EXPECT_EQ(dense_map.size(), 5);

    // 删除多个键
    erased_count = dense_map.erase(3);
    EXPECT_EQ(erased_count, 1);
    EXPECT_EQ(dense_map.size(), 4);
    EXPECT_FALSE(dense_map.contains(3));

    erased_count = dense_map.erase(5);
    EXPECT_EQ(erased_count, 1);
    EXPECT_EQ(dense_map.size(), 3);
    EXPECT_FALSE(dense_map.contains(5));

    // 删除剩余键
    erased_count = dense_map.erase(1);
    EXPECT_EQ(erased_count, 1);
    erased_count = dense_map.erase(2);
    EXPECT_EQ(erased_count, 1);
    erased_count = dense_map.erase(4);
    EXPECT_EQ(erased_count, 1);
    EXPECT_TRUE(dense_map.empty());
}

TEST(MapEraseTest, erase_edge_cases) {
    // 删除空map的迭代器（不应该崩溃）
    Map<int, int> empty_map;
    EXPECT_TRUE(empty_map.empty());
    auto it = empty_map.end();
    auto result = empty_map.erase(it);
    EXPECT_EQ(result, empty_map.end());

    // 从空map删除键
    size_t erased_count = empty_map.erase(1);
    EXPECT_EQ(erased_count, 0);

    // 删除单元素map
    Map<int, int> single_map = {{1, 10}};
    EXPECT_EQ(single_map.size(), 1);

    it = single_map.begin();
    result = single_map.erase(it);
    EXPECT_TRUE(single_map.empty());
    EXPECT_EQ(result, single_map.end());

    // 使用键删除单元素map
    single_map.insert({1, 10});
    EXPECT_EQ(single_map.size(), 1);
    erased_count = single_map.erase(1);
    EXPECT_EQ(erased_count, 1);
    EXPECT_TRUE(single_map.empty());

    // 删除超出范围的迭代器
    Map<int, int> map = {{1, 10}, {2, 20}};
    it = map.end();
    result = map.erase(it);
    EXPECT_EQ(result, map.end());
    EXPECT_EQ(map.size(), 2);
}

TEST(MapEraseTest, erase_with_cow) {
    // 测试引用计数和COW
    Map<int, int> map1 = {{1, 10}, {2, 20}, {3, 30}};
    Map<int, int> map2 = map1;// 共享数据

    EXPECT_EQ(map1.use_count(), 2);
    EXPECT_EQ(map2.use_count(), 2);

    // map1执行删除操作，应该触发COW
    map1.erase(1);
    EXPECT_EQ(map1.size(), 2);
    EXPECT_EQ(map1.use_count(), 1);// 现在应该是唯一的
    EXPECT_EQ(map2.size(), 3);     // map2应该不受影响
    EXPECT_EQ(map2.use_count(), 1);// map2现在也是唯一的

    // 验证map1和map2现在是独立的
    map2.erase(2);
    EXPECT_EQ(map1.size(), 2);
    EXPECT_EQ(map2.size(), 2);
    EXPECT_TRUE(map1.contains(2));
    EXPECT_FALSE(map2.contains(2));
}

TEST(MapEraseTest, erase_with_different_types) {
    // 测试String作为键
    Map<String, String> string_map = {{"key1", "value1"}, {"key2", "value2"}, {"key3", "value3"}};
    EXPECT_EQ(string_map.size(), 3);

    // 使用键删除
    size_t erased_count = string_map.erase("key2");
    EXPECT_EQ(erased_count, 1);
    EXPECT_EQ(string_map.size(), 2);
    EXPECT_FALSE(string_map.contains("key2"));

    // 使用迭代器删除
    auto it = string_map.find("key1");
    auto result = string_map.erase(it);
    EXPECT_EQ(string_map.size(), 1);
    EXPECT_EQ(result->first, "key3");
    EXPECT_EQ(result->second, "value3");

    // 测试嵌套Map
    Map<int, Map<String, int>> nested_map = {
            {1, {{"a", 10}, {"b", 20}}},
            {2, {{"c", 30}, {"d", 40}}}};
    EXPECT_EQ(nested_map.size(), 2);

    EXPECT_EQ(nested_map[1]["a"], 10);
    EXPECT_EQ(nested_map[1]["b"], 20);
    EXPECT_EQ(nested_map[2]["c"], 30);
    EXPECT_EQ(nested_map[2]["d"], 40);

    auto m = nested_map[1]["a"];

    // 删除嵌套Map的元素
    nested_map[1].erase("a");
    EXPECT_EQ(nested_map[1].size(), 1);
    EXPECT_FALSE(nested_map[1].contains("a"));

    // 删除整个嵌套Map
    nested_map.erase(2);
    EXPECT_EQ(nested_map.size(), 1);
    EXPECT_FALSE(nested_map.contains(2));
}

TEST(MapEraseTest, erase_and_iterate_safety) {
    Map<int, int> map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}};

    // 安全的迭代和删除方式
    for (auto it = map.begin(); it != map.end();) {
        if (it->first % 2 == 0) {
            it = map.erase(it);// 正确的方式：使用erase返回的迭代器
        } else {
            ++it;
        }
    }

    EXPECT_EQ(map.size(), 3);
    EXPECT_TRUE(map.contains(1));
    EXPECT_TRUE(map.contains(3));
    EXPECT_TRUE(map.contains(5));
    EXPECT_FALSE(map.contains(2));
    EXPECT_FALSE(map.contains(4));
}
}// namespace

#ifdef TEST_MAP


#endif
