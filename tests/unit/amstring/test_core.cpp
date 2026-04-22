// test_core.cpp - Unit tests for basic_string_core
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "amstring/core.hpp"

namespace aethermind {
namespace test {

// Convenience alias for the default core type used in Phase 1
using core_type = basic_string_core<
    char,
    std::char_traits<char>,
    std::allocator<char>,
    StableLayoutPolicy<char>,
    default_growth_policy>;

// ============================================================
// Default construction
// ============================================================

TEST(CoreTest, DefaultConstruct) {
    core_type c;
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
    EXPECT_STREQ(c.data(), "");
    EXPECT_STREQ(c.c_str(), "");
}

TEST(CoreTest, DefaultConstructWithAllocator) {
    std::allocator<char> alloc;
    core_type c(alloc);
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
}

// ============================================================
// Construction from pointer + length
// ============================================================

TEST(CoreTest, ConstructSmallFromString) {
    core_type c("hello", 5);
    EXPECT_EQ(c.size(), 5u);
    EXPECT_FALSE(c.empty());
    EXPECT_STREQ(c.data(), "hello");
}

TEST(CoreTest, ConstructEmptyString) {
    core_type c("", 0);
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
}

TEST(CoreTest, ConstructHeapFromString) {
    // Create a string longer than SSO capacity
    std::string long_str(100, 'x');
    core_type c(long_str.data(), long_str.size());
    EXPECT_EQ(c.size(), 100u);
    EXPECT_EQ(c.data()[0], 'x');
    EXPECT_EQ(c.data()[99], 'x');
}

// ============================================================
// Copy and move
// ============================================================

TEST(CoreTest, CopyConstructSmall) {
    core_type original("abc", 3);
    core_type copy(original);
    EXPECT_EQ(copy.size(), 3u);
    EXPECT_STREQ(copy.data(), "abc");
}

TEST(CoreTest, CopyConstructHeap) {
    std::string long_str(100, 'y');
    core_type original(long_str.data(), long_str.size());
    core_type copy(original);
    EXPECT_EQ(copy.size(), 100u);
    EXPECT_EQ(copy.data()[0], 'y');
}

TEST(CoreTest, MoveConstructSmall) {
    core_type original("mov", 3);
    core_type moved(std::move(original));
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_STREQ(moved.data(), "mov");
    // Moved-from object must be in valid state
    EXPECT_TRUE(original.empty() || original.size() == 0);
}

TEST(CoreTest, MoveConstructHeap) {
    std::string long_str(100, 'z');
    core_type original(long_str.data(), long_str.size());
    core_type moved(std::move(original));
    EXPECT_EQ(moved.size(), 100u);
    EXPECT_EQ(moved.data()[0], 'z');
}

// ============================================================
// Assignment
// ============================================================

TEST(CoreTest, CopyAssignSmallToSmall) {
    core_type a("aaa", 3);
    core_type b("bbb", 3);
    a = b;
    EXPECT_STREQ(a.data(), "bbb");
}

TEST(CoreTest, MoveAssign) {
    core_type a("old", 3);
    core_type b("newvalue", 8);
    a = std::move(b);
    EXPECT_STREQ(a.data(), "newvalue");
}

// ============================================================
// Basic accessors
// ============================================================

TEST(CoreTest, CapacitySmall) {
    core_type c("hi", 2);
    EXPECT_GE(c.capacity(), 2u);
}

TEST(CoreTest, CapacityHeap) {
    std::string long_str(100, 'a');
    core_type c(long_str.data(), long_str.size());
    EXPECT_GE(c.capacity(), 100u);
}

// ============================================================
// Mutation basics
// ============================================================

TEST(CoreTest, Clear) {
    core_type c("hello", 5);
    c.clear();
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
}

TEST(CoreTest, AppendSmall) {
    core_type c("hello", 5);
    c.append(" world", 6);
    EXPECT_EQ(c.size(), 11u);
    EXPECT_STREQ(c.data(), "hello world");
}

TEST(CoreTest, AssignSmall) {
    core_type c("hello", 5);
    c.assign("bye", 3);
    EXPECT_EQ(c.size(), 3u);
    EXPECT_STREQ(c.data(), "bye");
}

TEST(CoreTest, Reserve) {
    core_type c("abc", 3);
    c.reserve(200);
    EXPECT_GE(c.capacity(), 200u);
    EXPECT_STREQ(c.data(), "abc");
}

TEST(CoreTest, ResizeGrow) {
    core_type c("ab", 2);
    c.resize(5, 'x');
    EXPECT_EQ(c.size(), 5u);
    EXPECT_EQ(c.data()[2], 'x');
    EXPECT_EQ(c.data()[4], 'x');
}

TEST(CoreTest, ResizeShrink) {
    core_type c("hello", 5);
    c.resize(2);
    EXPECT_EQ(c.size(), 2u);
    EXPECT_STREQ(c.data(), "he");
}

// ============================================================
// Invariant check
// ============================================================

TEST(CoreTest, CheckInvariantsDefault) {
    core_type c;
    c.check_invariants();
}

TEST(CoreTest, CheckInvariantsSmall) {
    core_type c("test", 4);
    c.check_invariants();
}

TEST(CoreTest, CheckInvariantsHeap) {
    std::string long_str(100, 'q');
    core_type c(long_str.data(), long_str.size());
    c.check_invariants();
}

}// namespace test
}// namespace aethermind
