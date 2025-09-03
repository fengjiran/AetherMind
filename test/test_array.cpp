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

}// namespace