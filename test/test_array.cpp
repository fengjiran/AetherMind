//
// Created by 赵丹 on 2025/9/2.
//
#include "container/array.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Array, init) {
    ObjectPtr<ArrayImpl> impl;
    EXPECT_TRUE(impl.get() == NullTypeOf<ArrayImpl>::singleton());
    EXPECT_TRUE(IsNullTypePtr(impl.get()));
    Array<int> vec1(5, 42);
    auto it = vec1.begin();
    *it = 43;

    EXPECT_TRUE(vec1.defined());
    EXPECT_TRUE(vec1.unique());
    EXPECT_EQ(vec1.size(), 5);
    EXPECT_EQ(vec1.capacity(), 5);
    EXPECT_EQ(vec1.front(), 43);
    EXPECT_EQ(*vec1.begin(), 43);
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
    AnyV1 t1 = vec21;
    AnyV1 t2 = vec21;
    t1 = t2;
    EXPECT_EQ(vec21.use_count(), 4);

    Array<int> vec3 = {4, 5, 6, 7, 8, 9};
    EXPECT_TRUE(vec3.defined());
    EXPECT_TRUE(vec3.unique());
    EXPECT_EQ(vec3.size(), 6);
    EXPECT_EQ(vec3.capacity(), 6);
    EXPECT_EQ(vec3.front(), 4);
    EXPECT_EQ(vec3.back(), 9);
    vec3[0] = 8.1;
    EXPECT_EQ(vec3.front(), 8);
    vec3.front() = 9;
    EXPECT_EQ(vec3.front(), 9);

    vec3.back() = 10.5;
    EXPECT_EQ(vec3.back(), 10);

    Any t = vec3;
    auto x = t.operator[]<Array<int>>(0);
    EXPECT_EQ(x, 9);
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

TEST(Array, CopyConstructor) {
    Array<int> original = {1, 2, 3};
    Array<int> copy(original);

    EXPECT_EQ(copy.size(), original.size());
    for (size_t i = 0; i < copy.size(); ++i) {
        EXPECT_EQ(copy[i], original[i]);
    }

    // // 验证深拷贝
    if (copy.size() > 0) {
        copy[0] = 999;
        EXPECT_FALSE(copy[0] == original[0]);
    }
}

// 测试移动构造函数
TEST(Array, MoveConstructor) {
    Array<int> original = {1, 2, 3};
    size_t original_size = original.size();

    Array<int> moved(std::move(original));

    EXPECT_EQ(moved.size(), original_size);
    EXPECT_TRUE(original.empty());// 原对象应该为空

    // 验证移动后的数据
    if (moved.size() >= 3) {
        EXPECT_EQ(moved[0], 1);
        EXPECT_EQ(moved[1], 2);
        EXPECT_EQ(moved[2], 3);
    }
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

TEST(Array, EmptyArrayExceptions) {
    Array<int> arr;

    EXPECT_THROW(arr.front(), Error);

    EXPECT_THROW({try {
            UNUSED(arr.back());
        } catch (const Error&) {
            throw;
        } }, Error);

    EXPECT_THROW({try {
            UNUSED(arr.at(1));
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

TEST(Array, SwapOperation) {
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
            UNUSED(arr.at(-1));
        } catch (const Error&) {
            throw;
        } }, Error);

    // Test index beyond size
    EXPECT_THROW({
        try {
            UNUSED(arr.at(5));
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
    EXPECT_EQ(arr1.size(), 3); // arr1 unchanged
    EXPECT_EQ(arr2.size(), 4); // arr2 modified
    EXPECT_TRUE(arr1.unique());// Now they have separate data
    EXPECT_TRUE(arr2.unique());
}

TEST(Array, ClearMethod) {
    // Test clearing a non-empty array
    Array<int> arr = {1, 2, 3, 4, 5};
    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.size(), 5);

    arr.clear();
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
    EXPECT_EQ(arr.capacity(), 5);// Capacity should remain unchanged

    // Test that elements are actually removed
    EXPECT_THROW({
        try {
            UNUSED(arr.at(0));
        } catch (const Error&) {
            throw;
        } }, Error);

    EXPECT_THROW({
        try {
            UNUSED(arr.front());
        } catch (const Error&) {
            throw;
        } }, Error);

    EXPECT_THROW({
        try {
            UNUSED(arr.back());
        } catch (const Error&) {
            throw;
        } }, Error);
}

TEST(Array, ClearEmptyArray) {
    // Test clearing an already empty array
    Array<int> arr;
    EXPECT_TRUE(arr.empty());

    arr.clear();// Should not throw any exception
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
}

TEST(Array, ClearAfterModifications) {
    // Test clear after various operations
    Array<std::string> arr;

    // Add some elements
    arr.push_back("hello");
    arr.push_back("world");
    arr.emplace_back("test");

    EXPECT_EQ(arr.size(), 3);

    // Clear and verify
    arr.clear();
    EXPECT_TRUE(arr.empty());

    // Add elements again after clear
    arr.push_back("new");
    arr.emplace_back("data");

    EXPECT_EQ(arr.size(), 2);
    EXPECT_EQ(arr[0], "new");
    EXPECT_EQ(arr[1], "data");

    // Clear again
    arr.clear();
    EXPECT_TRUE(arr.empty());
}

TEST(Array, ClearWithSharedData) {
    // Test clear behavior with shared data (copy-on-write)
    Array<int> arr1 = {1, 2, 3};
    Array<int> arr2 = arr1;// Shared data

    EXPECT_EQ(arr1.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // Clearing one array should not affect the other
    arr1.clear();
    EXPECT_TRUE(arr1.empty());
    EXPECT_EQ(arr2.size(), 3);// arr2 should remain unchanged
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);

    // Now they should have separate data
    EXPECT_TRUE(arr1.unique());
    EXPECT_TRUE(arr2.unique());
}

TEST(Array, ClearPreservesCapacity) {
    // Test that clear preserves capacity
    Array<int> arr;

    // Add multiple elements to increase capacity
    for (int i = 0; i < 10; ++i) {
        arr.push_back(i);
    }

    size_t original_capacity = arr.capacity();
    EXPECT_GE(original_capacity, 10);

    arr.clear();
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
    EXPECT_EQ(arr.capacity(), original_capacity);// Capacity should be preserved

    // Adding elements again should use existing capacity
    arr.push_back(42);
    EXPECT_EQ(arr.size(), 1);
    EXPECT_EQ(arr.capacity(), original_capacity);
}

TEST(Array, PopBackMethod) {
    // Test popping from non-empty array
    Array<int> arr = {1, 2, 3, 4, 5};
    size_t original_size = arr.size();

    arr.pop_back();
    EXPECT_EQ(arr.size(), original_size - 1);
    EXPECT_EQ(arr.back(), 4);
    EXPECT_EQ(arr[arr.size() - 1], 4);

    // Test multiple pops
    arr.pop_back();
    EXPECT_EQ(arr.size(), 3);
    EXPECT_EQ(arr.back(), 3);

    arr.pop_back();
    EXPECT_EQ(arr.size(), 2);
    EXPECT_EQ(arr.back(), 2);

    // Test popping until empty
    arr.pop_back();
    arr.pop_back();
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
}

TEST(Array, PopBackEmptyArrayException) {
    // Test popping from empty array throws exception
    Array<int> arr;

    EXPECT_THROW({
        try {
            arr.pop_back();
        } catch (const Error& e) {
            // Verify it's the correct exception type
            EXPECT_NE(std::string(e.what()).find("Cannot pop back an empty array"), std::string::npos);
            throw;
        } }, Error);

    // Test popping from array that becomes empty
    Array<int> arr2 = {42};
    arr2.pop_back();

    EXPECT_THROW({
        try {
            arr2.pop_back();
        } catch (const Error& e) {
            EXPECT_NE(std::string(e.what()).find("Cannot pop back an empty array"), std::string::npos);
            throw;
        } }, Error);
}

TEST(Array, PopBackWithCopyOnWrite) {
    // Test pop_back behavior with shared data (copy-on-write)
    Array<int> arr1 = {1, 2, 3, 4, 5};
    Array<int> arr2 = arr1;// Shared data

    EXPECT_EQ(arr1.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // Popping from one array should trigger copy-on-write
    arr1.pop_back();

    // Arrays should now have separate data
    EXPECT_TRUE(arr1.unique());
    EXPECT_TRUE(arr2.unique());

    EXPECT_EQ(arr1.size(), 4);// arr1 modified
    EXPECT_EQ(arr1.back(), 4);

    EXPECT_EQ(arr2.size(), 5);// arr2 unchanged
    EXPECT_EQ(arr2.back(), 5);
    EXPECT_EQ(arr2[4], 5);
}

TEST(Array, PopBackPreservesCapacity) {
    // Test that pop_back preserves capacity
    Array<int> arr;

    // Add multiple elements to increase capacity
    for (int i = 0; i < 10; ++i) {
        arr.push_back(i);
    }

    size_t original_capacity = arr.capacity();
    EXPECT_GE(original_capacity, 10);

    // Pop some elements
    arr.pop_back();
    arr.pop_back();
    arr.pop_back();

    EXPECT_EQ(arr.size(), 7);
    EXPECT_EQ(arr.capacity(), original_capacity);// Capacity should be preserved

    // Pop all elements
    while (!arr.empty()) {
        arr.pop_back();
    }

    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
    EXPECT_EQ(arr.capacity(), original_capacity);// Capacity should still be preserved
}

TEST(Array, PopBackWithComplexTypes) {
    // Test pop_back with std::string
    Array<std::string> str_arr = {"hello", "world", "test", "example"};

    EXPECT_EQ(str_arr.size(), 4);
    EXPECT_EQ(str_arr.back(), "example");

    str_arr.pop_back();
    EXPECT_EQ(str_arr.size(), 3);
    EXPECT_EQ(str_arr.back(), "test");

    str_arr.pop_back();
    EXPECT_EQ(str_arr.size(), 2);
    EXPECT_EQ(str_arr.back(), "world");

    // Test pop_back preserves remaining elements
    EXPECT_EQ(str_arr[0], "hello");
    EXPECT_EQ(str_arr[1], "world");
}

TEST(Array, ResizeMethod) {
    // Test resizing to larger size
    Array<int> arr = {1, 2, 3};
    EXPECT_EQ(arr.size(), 3);

    arr.resize(5);
    EXPECT_EQ(arr.size(), 5);
    EXPECT_GE(arr.capacity(), 5);

    // Original elements should be preserved
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);

    // New elements should be default initialized (but behavior depends on implementation)
    // For int, new elements might be uninitialized or zero-initialized
}

TEST(Array, ResizeToSmallerSize) {
    // Test resizing to smaller size
    Array<int> arr = {1, 2, 3, 4, 5};
    EXPECT_EQ(arr.size(), 5);

    arr.resize(3);
    EXPECT_EQ(arr.size(), 3);

    // Remaining elements should be preserved
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);

    // Capacity should be preserved
    size_t original_capacity = arr.capacity();
    EXPECT_GE(original_capacity, 5);
}

TEST(Array, ResizeToSameSize) {
    // Test resizing to the same size
    Array<int> arr = {1, 2, 3};
    size_t original_size = arr.size();
    size_t original_capacity = arr.capacity();

    arr.resize(3);
    EXPECT_EQ(arr.size(), original_size);
    EXPECT_EQ(arr.capacity(), original_capacity);

    // Elements should be unchanged
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
}

TEST(Array, ResizeEmptyArray) {
    // Test resizing an empty array
    Array<int> arr;
    EXPECT_TRUE(arr.empty());

    arr.resize(3);
    EXPECT_EQ(arr.size(), 3);
    EXPECT_GE(arr.capacity(), 3);

    // Array should now be initialized with default values
}

TEST(Array, ResizeToZero) {
    // Test resizing to zero
    Array<int> arr = {1, 2, 3};
    EXPECT_EQ(arr.size(), 3);

    arr.resize(0);
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);

    // Capacity should be preserved
    EXPECT_GE(arr.capacity(), 3);
}

TEST(Array, ResizeNegativeSizeException) {
    // Test that resizing to negative size throws exception
    Array<int> arr = {1, 2, 3};

    EXPECT_THROW({
        try {
            arr.resize(-1);
        } catch (const Error& e) {
            // Verify it's the correct exception type and message
            EXPECT_NE(std::string(e.what()).find("Cannot resize an array to a negative size"), std::string::npos);
            throw;
        } }, Error);
}

TEST(Array, ResizeWithCopyOnWrite) {
    // Test resize behavior with shared data (copy-on-write)
    Array<int> arr1 = {1, 2, 3, 4, 5};
    Array<int> arr2 = arr1;// Shared data

    EXPECT_EQ(arr1.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // Resizing one array should trigger copy-on-write
    arr1.resize(3);

    // Arrays should now have separate data
    EXPECT_TRUE(arr1.unique());
    EXPECT_TRUE(arr2.unique());

    EXPECT_EQ(arr1.size(), 3);// arr1 resized
    EXPECT_EQ(arr2.size(), 5);// arr2 unchanged

    // Verify arr1 elements
    EXPECT_EQ(arr1[0], 1);
    EXPECT_EQ(arr1[1], 2);
    EXPECT_EQ(arr1[2], 3);

    // Verify arr2 elements
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
    EXPECT_EQ(arr2[3], 4);
    EXPECT_EQ(arr2[4], 5);
}

TEST(Array, ResizeMultipleTimes) {
    // Test multiple resize operations
    Array<int> arr;

    // Grow
    arr.resize(5);
    EXPECT_EQ(arr.size(), 5);

    // Shrink
    arr.resize(2);
    EXPECT_EQ(arr.size(), 2);

    // Grow again
    arr.resize(8);
    EXPECT_EQ(arr.size(), 8);

    // Shrink to zero
    arr.resize(0);
    EXPECT_TRUE(arr.empty());

    // Grow from zero
    arr.resize(4);
    EXPECT_EQ(arr.size(), 4);
}

TEST(Array, ResizeWithComplexTypes) {
    // Test resize with std::string
    Array<std::string> str_arr = {"hello", "world", "test"};
    EXPECT_EQ(str_arr.size(), 3);

    // Resize to larger
    str_arr.resize(5);
    EXPECT_TRUE(str_arr.unique());

    // Original elements should be preserved
    EXPECT_EQ(str_arr[0], "hello");
    EXPECT_EQ(str_arr[1], "world");
    EXPECT_EQ(str_arr[2], "test");

    // New elements should be default initialized (empty strings)
    EXPECT_EQ(str_arr[3], "");
    EXPECT_EQ(str_arr[4], "");

    // Resize to smaller
    str_arr.resize(2);
    EXPECT_EQ(str_arr.size(), 2);
    EXPECT_EQ(str_arr[0], "hello");
    EXPECT_EQ(str_arr[1], "world");
}

TEST(Array, ReserveMethod) {
    // Test reserving capacity for empty array
    Array<int> arr;
    EXPECT_EQ(arr.capacity(), 0);

    arr.reserve(10);
    EXPECT_EQ(arr.capacity(), 10);
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
}

TEST(Array, ReserveLargerCapacity) {
    // Test reserving larger capacity than current
    Array<int> arr = {1, 2, 3};
    size_t original_capacity = arr.capacity();
    size_t original_size = arr.size();

    arr.reserve(20);
    EXPECT_EQ(arr.capacity(), 20);
    EXPECT_EQ(arr.size(), original_size);

    // Original elements should be preserved
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
}

TEST(Array, ReserveSmallerCapacity) {
    // Test reserving smaller capacity than current (should be no-op)
    Array<int> arr;
    arr.reserve(10);
    EXPECT_EQ(arr.capacity(), 10);

    size_t current_capacity = arr.capacity();
    arr.reserve(5);
    EXPECT_EQ(arr.capacity(), current_capacity);// Capacity should remain unchanged
    EXPECT_TRUE(arr.empty());
}

TEST(Array, ReserveSameCapacity) {
    // Test reserving the same capacity as current
    Array<int> arr = {1, 2, 3};
    size_t original_capacity = arr.capacity();
    size_t original_size = arr.size();

    arr.reserve(original_capacity);
    EXPECT_EQ(arr.capacity(), original_capacity);
    EXPECT_EQ(arr.size(), original_size);

    // Elements should be unchanged
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
}

TEST(Array, ReserveWithCopyOnWrite) {
    // Test reserve behavior with shared data (copy-on-write)
    Array<int> arr1 = {1, 2, 3};
    Array<int> arr2 = arr1;// Shared data

    EXPECT_EQ(arr1.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // Reserving capacity in one array should trigger copy-on-write
    arr1.reserve(20);

    // Arrays should now have separate data
    EXPECT_TRUE(arr1.unique());
    EXPECT_TRUE(arr2.unique());

    EXPECT_EQ(arr1.capacity(), 20);
    EXPECT_EQ(arr2.capacity(), 3);// arr2 capacity unchanged

    // Both arrays should have the same elements
    EXPECT_EQ(arr1.size(), 3);
    EXPECT_EQ(arr2.size(), 3);
    EXPECT_EQ(arr1[0], 1);
    EXPECT_EQ(arr1[1], 2);
    EXPECT_EQ(arr1[2], 3);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
}

TEST(Array, ReserveMultipleTimes) {
    // Test multiple reserve operations
    Array<int> arr;

    arr.reserve(5);
    EXPECT_EQ(arr.capacity(), 5);

    arr.reserve(10);
    EXPECT_EQ(arr.capacity(), 10);

    arr.reserve(15);
    EXPECT_EQ(arr.capacity(), 15);

    // Adding elements after reserving
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);

    EXPECT_EQ(arr.size(), 3);
    EXPECT_EQ(arr.capacity(), 15);
}

TEST(Array, ReserveWithComplexTypes) {
    // Test reserve with std::string
    Array<std::string> str_arr = {"hello", "world", "test"};
    size_t original_size = str_arr.size();

    str_arr.reserve(20);
    EXPECT_EQ(str_arr.capacity(), 20);
    EXPECT_EQ(str_arr.size(), original_size);

    // Original elements should be preserved
    EXPECT_EQ(str_arr[0], "hello");
    EXPECT_EQ(str_arr[1], "world");
    EXPECT_EQ(str_arr[2], "test");

    // Adding new elements should work correctly
    str_arr.push_back("new");
    str_arr.emplace_back("element");

    EXPECT_EQ(str_arr.size(), 5);
    EXPECT_EQ(str_arr.capacity(), 20);
    EXPECT_EQ(str_arr[3], "new");
    EXPECT_EQ(str_arr[4], "element");
}

TEST(Array, ReserveZeroCapacity) {
    // Test reserving zero capacity (should be no-op for non-empty arrays)
    Array<int> arr = {1, 2, 3};
    size_t original_capacity = arr.capacity();

    arr.reserve(0);
    EXPECT_EQ(arr.capacity(), original_capacity);// Should remain unchanged
    EXPECT_EQ(arr.size(), 3);

    // Test with empty array
    Array<int> empty_arr;
    empty_arr.reserve(0);
    EXPECT_EQ(empty_arr.capacity(), 0);
    EXPECT_TRUE(empty_arr.empty());
}

TEST(Array, ReserveAndThenAddElements) {
    // Test that reserved capacity is actually used when adding elements
    Array<int> arr;

    // Reserve capacity first
    arr.reserve(100);
    EXPECT_EQ(arr.capacity(), 100);
    EXPECT_TRUE(arr.empty());

    // Add elements - should not trigger reallocation
    for (int i = 0; i < 100; ++i) {
        arr.push_back(i);
        EXPECT_EQ(arr.capacity(), 100);// Capacity should remain constant
    }

    EXPECT_EQ(arr.size(), 100);
    EXPECT_EQ(arr.capacity(), 100);

    // Adding one more element should trigger reallocation
    arr.push_back(100);
    EXPECT_GT(arr.capacity(), 100);// Capacity should increase
    EXPECT_EQ(arr.size(), 101);
}

TEST(Array, InsertSingleElement) {
    // Test inserting single element at beginning
    Array<int> arr = {2, 3, 4};
    arr.insert(arr.begin(), 1);

    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);

    // Test inserting single element at middle
    Array<int> arr2 = {1, 3, 4};
    arr2.insert(arr2.begin() + 1, 2);

    EXPECT_EQ(arr2.size(), 4);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
    EXPECT_EQ(arr2[3], 4);

    // Test inserting single element at end
    Array<int> arr3 = {1, 2, 3};
    arr3.insert(arr3.end(), 4);

    EXPECT_EQ(arr3.size(), 4);
    EXPECT_EQ(arr3[0], 1);
    EXPECT_EQ(arr3[1], 2);
    EXPECT_EQ(arr3[2], 3);
    EXPECT_EQ(arr3[3], 4);
}

TEST(Array, InsertMultipleElements) {
    // Test inserting multiple elements at beginning
    Array<int> arr = {4, 5, 6};
    std::vector<int> elements = {1, 2, 3};
    arr.insert(arr.begin(), elements.begin(), elements.end());

    EXPECT_EQ(arr.size(), 6);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
    EXPECT_EQ(arr[4], 5);
    EXPECT_EQ(arr[5], 6);

    // Test inserting multiple elements at middle
    Array<int> arr2 = {1, 5, 6};
    std::vector<int> elements2 = {2, 3, 4};
    arr2.insert(arr2.begin() + 1, elements2.begin(), elements2.end());

    EXPECT_EQ(arr2.size(), 6);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
    EXPECT_EQ(arr2[3], 4);
    EXPECT_EQ(arr2[4], 5);
    EXPECT_EQ(arr2[5], 6);

    // Test inserting multiple elements at end
    Array<int> arr3 = {1, 2, 3};
    std::vector<int> elements3 = {4, 5, 6};
    arr3.insert(arr3.end(), elements3.begin(), elements3.end());

    EXPECT_EQ(arr3.size(), 6);
    EXPECT_EQ(arr3[0], 1);
    EXPECT_EQ(arr3[1], 2);
    EXPECT_EQ(arr3[2], 3);
    EXPECT_EQ(arr3[3], 4);
    EXPECT_EQ(arr3[4], 5);
    EXPECT_EQ(arr3[5], 6);
}

TEST(Array, InsertIntoEmptyArray) {
    // Test inserting into empty array
    Array<int> arr;
    arr.insert(arr.begin(), 42);

    EXPECT_EQ(arr.size(), 1);
    EXPECT_EQ(arr[0], 42);

    // Test inserting multiple elements into empty array
    Array<int> arr2;
    std::vector<int> elements = {1, 2, 3};
    arr2.insert(arr2.begin(), elements.begin(), elements.end());

    EXPECT_EQ(arr2.size(), 3);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
}

TEST(Array, InsertWithCopyOnWrite) {
    // Test insert behavior with shared data (copy-on-write)
    Array<int> arr1 = {1, 2, 3};
    Array<int> arr2 = arr1;// Shared data

    EXPECT_EQ(arr1.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // Inserting into one array should trigger copy-on-write
    arr1.insert(arr1.begin() + 1, 99);

    // Arrays should now have separate data
    EXPECT_TRUE(arr1.unique());
    EXPECT_TRUE(arr2.unique());

    // arr1 should be modified
    EXPECT_EQ(arr1.size(), 4);
    EXPECT_EQ(arr1[0], 1);
    EXPECT_EQ(arr1[1], 99);
    EXPECT_EQ(arr1[2], 2);
    EXPECT_EQ(arr1[3], 3);

    // arr2 should remain unchanged
    EXPECT_EQ(arr2.size(), 3);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
}

TEST(Array, InsertWithComplexTypes) {
    // Test insert with std::string
    Array<std::string> str_arr = {"world", "!"};

    // Insert at beginning
    str_arr.insert(str_arr.begin(), "hello");
    EXPECT_EQ(str_arr.size(), 3);
    EXPECT_EQ(str_arr[0], "hello");
    EXPECT_EQ(str_arr[1], "world");
    EXPECT_EQ(str_arr[2], "!");

    // Insert multiple strings
    Array<std::string> str_arr2 = {"hello", "!"};
    std::vector<std::string> words = {"beautiful", "world"};
    str_arr2.insert(str_arr2.begin() + 1, words.begin(), words.end());

    EXPECT_EQ(str_arr2.size(), 4);
    EXPECT_EQ(str_arr2[0], "hello");
    EXPECT_EQ(str_arr2[1], "beautiful");
    EXPECT_EQ(str_arr2[2], "world");
    EXPECT_EQ(str_arr2[3], "!");
}

TEST(Array, InsertEmptyRange) {
    // Test inserting empty range (should be no-op)
    Array<int> arr = {1, 2, 3};
    size_t original_size = arr.size();

    std::vector<int> empty_vec;
    arr.insert(arr.begin() + 1, empty_vec.begin(), empty_vec.end());

    EXPECT_EQ(arr.size(), original_size);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
}

TEST(Array, InsertPreservesCapacity) {
    // Test that insert operations preserve capacity appropriately
    Array<int> arr;
    arr.reserve(10);
    size_t original_capacity = arr.capacity();

    // Insert elements within reserved capacity
    for (int i = 0; i < 5; ++i) {
        arr.insert(arr.begin(), i);
        EXPECT_EQ(arr.capacity(), original_capacity);
    }

    EXPECT_EQ(arr.size(), 5);
    EXPECT_EQ(arr.capacity(), original_capacity);
}

TEST(Array, InsertIteratorValidity) {
    // Test that insert operations maintain iterator validity
    Array<int> arr = {1, 3, 5};

    // Get iterator before insert
    auto it = arr.begin() + 1;
    EXPECT_EQ(*it, 3);

    // Insert at position
    arr.insert(arr.begin() + 1, 2);

    // // Verify array contents
    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 5);
}

TEST(Array, EraseSingleElement) {
    Array<int> arr = {1, 2, 3, 4, 5};

    // 删除中间元素
    auto it = arr.begin() + 2;// 指向元素3
    arr.erase(it);

    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 5);
}

// 测试删除第一个元素
TEST(Array, EraseFirstElement) {
    Array<int> arr = {1, 2, 3, 4, 5};

    arr.erase(arr.begin());

    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 2);
    EXPECT_EQ(arr[1], 3);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 5);
}

// 测试删除最后一个元素
TEST(Array, EraseLastElement) {
    Array<int> arr = {1, 2, 3, 4, 5};

    arr.erase(arr.end() - 1);

    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
}

// 测试删除范围元素
TEST(Array, EraseRange) {
    Array<int> arr = {1, 2, 3, 4, 5, 6, 7};

    // 删除中间范围的元素 [2, 3, 4, 5]
    auto first = arr.begin() + 1;// 指向元素2
    auto last = arr.begin() + 5; // 指向元素6
    arr.erase(first, last);

    EXPECT_EQ(arr.size(), 3);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 6);
    EXPECT_EQ(arr[2], 7);
}

// 测试删除整个数组范围
TEST(Array, EraseEntireRange) {
    Array<int> arr = {1, 2, 3, 4, 5};

    arr.erase(arr.begin(), arr.end());

    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
}

// 测试删除空范围（应该没有变化）
TEST(Array, EraseEmptyRange) {
    Array<int> arr = {1, 2, 3, 4, 5};
    size_t original_size = arr.size();

    // 删除空范围
    arr.erase(arr.begin(), arr.begin());

    EXPECT_EQ(arr.size(), original_size);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
    EXPECT_EQ(arr[4], 5);
}

// 测试删除字符串元素
TEST(Array, EraseStringElements) {
    Array<std::string> arr = {"apple", "banana", "cherry", "date", "elderberry"};

    // 删除中间元素
    arr.erase(arr.begin() + 2);// 删除"cherry"

    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], "apple");
    EXPECT_EQ(arr[1], "banana");
    EXPECT_EQ(arr[2], "date");
    EXPECT_EQ(arr[3], "elderberry");
}

TEST(Array, EraseWithComplexTypes) {
    // Test erase with std::string
    Array<std::string> str_arr = {"hello", "world", "test", "example"};

    // Erase single element
    str_arr.erase(str_arr.begin() + 1);
    EXPECT_EQ(str_arr.size(), 3);
    EXPECT_EQ(str_arr[0], "hello");
    EXPECT_EQ(str_arr[1], "test");
    EXPECT_EQ(str_arr[2], "example");

    // Erase range of elements
    Array<std::string> str_arr2 = {"hello", "beautiful", "world", "!"};
    str_arr2.erase(str_arr2.begin() + 1, str_arr2.begin() + 3);

    EXPECT_EQ(str_arr2.size(), 2);
    EXPECT_EQ(str_arr2[0], "hello");
    EXPECT_EQ(str_arr2[1], "!");
}

// 测试删除后的迭代器有效性
TEST(Array, IteratorValidityAfterErase) {
    Array<int> arr = {1, 2, 3, 4, 5};

    auto it = arr.begin() + 2;// 指向元素3
    arr.erase(it);

    // 迭代器应该仍然有效，但指向被删除元素后的位置
    EXPECT_EQ(*it, 4);// 现在指向元素4

    // 继续遍历剩余元素
    int expected = 4;
    for (; it != arr.end(); ++it) {
        EXPECT_EQ(*it, expected++);
    }
}

// 测试空数组的erase操作（应该抛出异常）
TEST(Array, EraseFromEmptyArray) {
    Array<int> empty_arr;

    EXPECT_THROW({try {
      empty_arr.erase(empty_arr.begin());
      } catch(const Error&) {
      throw;
      } }, Error);
}

TEST(Array, SetMethod) {
    // 测试基本Set功能
    Array<int> arr = {1, 2, 3, 4, 5};

    // 设置中间元素
    arr.Set(2, 99);
    EXPECT_EQ(arr[2], 99);
    EXPECT_EQ(arr.size(), 5);

    // 设置第一个元素
    arr.Set(0, 100);
    EXPECT_EQ(arr[0], 100);

    // 设置最后一个元素
    arr.Set(4, 200);
    EXPECT_EQ(arr[4], 200);

    // 验证其他元素未被修改
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[3], 4);
}

TEST(Array, SetWithComplexTypes) {
    // 测试复杂类型的Set操作
    Array<std::string> arr = {"hello", "world", "test"};

    arr.Set(1, "modified");
    EXPECT_EQ(arr[1], "modified");
    EXPECT_EQ(arr[0], "hello");
    EXPECT_EQ(arr[2], "test");

    // 测试长字符串
    arr.Set(0, "this is a very long string for testing purposes");
    EXPECT_EQ(arr[0], "this is a very long string for testing purposes");
}

TEST(Array, SetCopyOnWriteBehavior) {
    // 测试Set操作的写时复制行为
    Array<int> arr1 = {1, 2, 3, 4, 5};
    Array<int> arr2 = arr1;// 共享数据

    EXPECT_FALSE(arr1.unique());
    EXPECT_FALSE(arr2.unique());

    // 在arr2上执行Set操作，应该触发写时复制
    arr2.Set(2, 999);

    // 验证写时复制发生
    EXPECT_TRUE(arr1.unique());
    EXPECT_TRUE(arr2.unique());

    // 验证数据正确性
    EXPECT_EQ(arr1[2], 3);  // 原始数据未改变
    EXPECT_EQ(arr2[2], 999);// 新数据已设置

    // 验证其他元素未被修改
    for (int i = 0; i < 5; ++i) {
        if (i != 2) {
            EXPECT_EQ(arr1[i], arr2[i]);
        }
    }
}

TEST(Array, SetOutOfBounds) {
    // 测试越界Set操作（应该抛出异常）
    Array<int> arr = {1, 2, 3};

    // 负索引
    EXPECT_THROW(arr.Set(-1, 99), Error);

    // 超出大小的索引
    EXPECT_THROW(arr.Set(3, 99), Error);
    EXPECT_THROW(arr.Set(100, 99), Error);

    // 空数组的Set操作
    Array<int> empty_arr;
    EXPECT_THROW(empty_arr.Set(0, 1), Error);
}

TEST(Array, SetMultipleOperations) {
    // 测试多次Set操作
    Array<int> arr = {0, 0, 0, 0, 0};

    // 多次设置不同位置的元素
    for (int i = 0; i < 5; ++i) {
        arr.Set(i, i * 10);
    }

    // 验证所有设置都正确
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(arr[i], i * 10);
    }

    // 再次修改某些元素
    arr.Set(0, 100);
    arr.Set(2, 200);
    arr.Set(4, 400);

    EXPECT_EQ(arr[0], 100);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], 200);
    EXPECT_EQ(arr[3], 30);
    EXPECT_EQ(arr[4], 400);
}

TEST(Array, SetPreservesCapacity) {
    // 测试Set操作不会改变容量
    Array<int> arr;
    arr.reserve(10);
    arr = {1, 2, 3, 4, 5};// 填充一些数据

    size_t original_capacity = arr.capacity();

    // 执行Set操作
    arr.Set(2, 99);

    // 验证容量保持不变
    EXPECT_EQ(arr.capacity(), original_capacity);
    EXPECT_EQ(arr.size(), 5);
    EXPECT_EQ(arr[2], 99);
}

TEST(Array, SetWithMoveSemantics) {
    // 测试Set操作支持移动语义
    Array<std::string> arr = {"original1", "original2", "original3"};

    std::string moved_string = "moved_value";
    arr.Set(1, std::move(moved_string));

    EXPECT_EQ(arr[1], "moved_value");
    // moved_string 可能处于有效但未指定的状态

    // int a = 10;
    // static_assert(std::is_same_v<decltype((a)), int&>);
}

}// namespace