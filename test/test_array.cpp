//
// Created by 赵丹 on 2025/9/2.
//
#include "container/array.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Array, init) {
    Array<int> vec1(5, 42);
    EXPECT_TRUE(vec1.defined());
    EXPECT_TRUE(vec1.unique());
    EXPECT_EQ(vec1.size(), 5);
    EXPECT_EQ(vec1.capacity(), 5);
    EXPECT_EQ(vec1.front(), 42);
    EXPECT_EQ(*vec1.begin(), 42);
    EXPECT_EQ(vec1.back(), 42);
    EXPECT_EQ(*(vec1.end() - 1), 42);

    Array<int> vec2 = std::vector<int>{1, 2, 3, 4, 5};
    EXPECT_TRUE(vec2.defined());
    EXPECT_TRUE(vec2.unique());
    EXPECT_EQ(vec2.size(), 5);
    EXPECT_EQ(vec2.capacity(), 5);
    EXPECT_EQ(vec2.front(), 1);
    EXPECT_EQ(vec2.back(), 5);
    auto vec21 = vec2;
    EXPECT_EQ(vec21.use_count(), 2);

    Array<int> vec3 = {4, 5, 6, 7, 8, 9};
    EXPECT_TRUE(vec3.defined());
    EXPECT_TRUE(vec3.unique());
    EXPECT_EQ(vec3.size(), 6);
    EXPECT_EQ(vec3.capacity(), 6);
    EXPECT_EQ(vec3.front(), 4);
    EXPECT_EQ(vec3.back(), 9);
}

TEST(Array, DefaultConstructor) {
    Array<int> arr;
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
    EXPECT_EQ(arr.capacity(), 0);
}

TEST(Array, SizeValueConstructor) {
    const size_t size = 5;
    const int value = 42;
    Array<int> arr(size, value);

    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.size(), size);
    EXPECT_EQ(arr.capacity(), size);

    for (const auto& elem: arr) {
        EXPECT_EQ(elem, value);
    }
}

TEST(Array, VectorConstructor) {
    std::vector<std::string> vec = {"hello", "world", "test"};
    Array<std::string> arr(vec);

    EXPECT_EQ(arr.size(), vec.size());
    EXPECT_EQ(arr.capacity(), arr.size());
    EXPECT_TRUE(std::equal(arr.begin(), arr.end(), vec.begin()));
}

TEST(Array, InitializerListConstructor) {
    Array<double> arr = {1.5, 2.5, 3.5, 4.5};

    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 1.5);
    EXPECT_EQ(arr[2], 3.5);
    EXPECT_EQ(arr.back(), 4.5);
}

TEST(Array, ElementAccess) {
    Array<int> arr = {10, 20, 30, 40};

    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[3], 40);
    EXPECT_EQ(arr.front(), 10);
    EXPECT_EQ(arr.back(), 40);
}

TEST(Array, Iterators) {
    Array<int> arr = {1, 2, 3, 4, 5};
    std::vector<int> expected = {1, 2, 3, 4, 5};

    // Test forward iterators
    EXPECT_TRUE(std::equal(arr.begin(), arr.end(), expected.begin()));

    // Test reverse iterators
    std::reverse(expected.begin(), expected.end());
    EXPECT_TRUE(std::equal(arr.rbegin(), arr.rend(), expected.begin()));
}

TEST(ArrayTest, EmptyArrayExceptions) {
    Array<int> arr;

    EXPECT_THROW({
        try {
            UNUSED(arr.front());
        } catch (const Error&) {
            throw;
        } }, Error);

    EXPECT_THROW({try {
            UNUSED(arr.back());
        } catch (const Error&) {
            throw;
        } }, Error);

    EXPECT_THROW({try {
            UNUSED(arr[1]);
        } catch (const Error&) {
            throw;
        } }, Error);
}

TEST(Array, CopyAndMoveOperations) {
    // Test copy constructor
    Array<int> arr1 = {1, 2, 3};
    Array<int> arr2(arr1);
    EXPECT_EQ(arr1.size(), arr2.size());
    EXPECT_TRUE(std::equal(arr1.begin(), arr1.end(), arr2.begin()));
    EXPECT_EQ(arr1.use_count(), 2);

    // Test move constructor
    Array<int> arr3(std::move(arr2));
    EXPECT_EQ(arr3.size(), 3);
    EXPECT_TRUE(arr2.empty());
    EXPECT_EQ(arr3.use_count(), 2);

    // Test copy assignment
    Array<int> arr4;
    arr4 = arr1;
    EXPECT_EQ(arr4.size(), arr1.size());
    EXPECT_EQ(arr4.use_count(), 3);

    // Test move assignment
    Array<int> arr5;
    arr5 = std::move(arr4);
    EXPECT_EQ(arr5.size(), 3);
    EXPECT_TRUE(arr4.empty());
    EXPECT_EQ(arr5.use_count(), 3);
}

TEST(ArrayTest, SwapOperation) {
    Array<int> arr1 = {1, 2, 3};
    Array<int> arr2 = {4, 5};

    arr1.swap(arr2);

    EXPECT_EQ(arr1.size(), 2);
    EXPECT_EQ(arr2.size(), 3);
    EXPECT_EQ(arr1[0], 4);
    EXPECT_EQ(arr2[0], 1);
}

TEST(Array, PushBackAndEmplaceBack) {
    Array<int> arr;

    // Test push_back
    arr.push_back(1);
    EXPECT_EQ(arr.size(), 1);
    EXPECT_EQ(arr[0], 1);
    EXPECT_TRUE(arr.unique());

    arr.push_back(2);
    EXPECT_EQ(arr.size(), 2);
    EXPECT_EQ(arr[1], 2);

    // Test emplace_back
    arr.emplace_back(3);
    EXPECT_EQ(arr.size(), 3);
    EXPECT_EQ(arr[2], 3);

    arr.emplace_back(4);
    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[3], 4);
}

TEST(Array, OutOfBoundsException) {
    Array<int> arr = {1, 2, 3};

    // Test negative index
    EXPECT_THROW({
        try {
            UNUSED(arr[-1]);
        } catch (const Error&) {
            throw;
        } }, Error);

    // Test index beyond size
    EXPECT_THROW({
        try {
            UNUSED(arr[5]);
        } catch (const Error&) {
            throw;
        } }, Error);
}

TEST(Array, UseCountAndUnique) {
    Array<int> arr1 = {1, 2, 3};
    EXPECT_TRUE(arr1.unique());
    EXPECT_EQ(arr1.use_count(), 1);

    // Test copy increases use count
    Array<int> arr2 = arr1;
    EXPECT_EQ(arr1.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);
    EXPECT_FALSE(arr1.unique());
    EXPECT_FALSE(arr2.unique());

    // Test move resets use count
    Array<int> arr3 = std::move(arr1);
    EXPECT_TRUE(arr1.empty());
    EXPECT_EQ(arr3.use_count(), 2);
}

TEST(Array, DefinedAndEmpty) {
    Array<int> arr1;
    EXPECT_FALSE(arr1.defined());
    EXPECT_TRUE(arr1.empty());

    Array<int> arr2 = {1};
    EXPECT_TRUE(arr2.defined());
    EXPECT_FALSE(arr2.empty());

    Array<int> arr3(3, 0);
    EXPECT_TRUE(arr3.defined());
    EXPECT_FALSE(arr3.empty());
}

TEST(Array, IteratorValidity) {
    Array<int> arr = {1, 2, 3, 4, 5};

    // Test iterator arithmetic
    auto it = arr.begin();
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(*(it + 1), 2);
    EXPECT_EQ(*(it + 4), 5);

    // Test reverse iterator
    auto rit = arr.rbegin();
    EXPECT_EQ(*rit, 5);
    EXPECT_EQ(*(rit + 1), 4);

    // Test const iterators
    const Array<int>& carr = arr;
    auto cit = carr.begin();
    EXPECT_EQ(*cit, 1);
}

TEST(Array, CapacityManagement) {
    Array<int> arr;

    // Initial capacity should be 0
    EXPECT_EQ(arr.capacity(), 0);

    // Adding elements should increase capacity
    arr.push_back(1);
    EXPECT_GE(arr.capacity(), 1);

    arr.push_back(2);
    arr.push_back(3);
    arr.push_back(4);
    EXPECT_GE(arr.capacity(), 4);
}

TEST(Array, ComplexTypes) {
    // Test with std::string
    Array<std::string> str_arr = {"hello", "world", "test"};
    EXPECT_EQ(str_arr.size(), 3);
    EXPECT_EQ(str_arr[0], "hello");
    EXPECT_EQ(str_arr[2], "test");

    // // Test with custom types
    // struct TestStruct {
    //     int id;
    //     std::string name;
    //     bool operator==(const TestStruct& other) const {
    //         return id == other.id && name == other.name;
    //     }
    // };
    //
    // Array<TestStruct> custom_arr = {{1, "first"}, {2, "second"}};
    // EXPECT_EQ(custom_arr.size(), 2);
    // EXPECT_EQ(custom_arr[0].id, 1);
    // EXPECT_EQ(custom_arr[1].name, "second");
}

TEST(Array, CopyOnWriteSemantics) {
    Array<int> arr1 = {1, 2, 3};
    Array<int> arr2 = arr1;

    // Both should share the same underlying data
    EXPECT_EQ(arr1.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // Modifying arr2 should trigger copy-on-write
    arr2.push_back(4);
    EXPECT_EQ(arr1.size(), 3);  // arr1 unchanged
    EXPECT_EQ(arr2.size(), 4);  // arr2 modified
    EXPECT_TRUE(arr1.unique());  // Now they have separate data
    EXPECT_TRUE(arr2.unique());
}

}// namespace