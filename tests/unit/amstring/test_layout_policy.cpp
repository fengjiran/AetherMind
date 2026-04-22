// test_layout_policy.cpp - Unit tests for stable_layout_policy
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#include "amstring/compact_layout_policy.hpp"
#include "amstring/stable_layout_policy.hpp"
#include <gtest/gtest.h>

namespace aethermind {
namespace test {

// ============================================================
// stable_layout_policy<char> tests
// ============================================================

using StableChar = StableLayoutPolicy<char>;

TEST(StableLayoutPolicyChar, InitEmpty) {
    StableChar::StorageType storage;
    StableChar::init_empty(storage);

    EXPECT_EQ(StableChar::size(storage), 0u);
    EXPECT_EQ(StableChar::capacity(storage), StableChar::kSmallCapacity);
    EXPECT_TRUE(StableChar::is_small(storage));
    EXPECT_FALSE(StableChar::is_heap(storage));
    EXPECT_STREQ(StableChar::data(storage), "");
}

TEST(StableLayoutPolicyChar, InitSmallOneChar) {
    StableChar::StorageType storage;
    StableChar::init_small(storage, "a", 1);

    EXPECT_EQ(StableChar::size(storage), 1u);
    EXPECT_TRUE(StableChar::is_small(storage));
    EXPECT_STREQ(StableChar::data(storage), "a");
}

TEST(StableLayoutPolicyChar, InitSmallLiteral) {
    StableChar::StorageType storage;
    const char* hello = "hello";
    StableChar::init_small(storage, hello, 5);

    EXPECT_EQ(StableChar::size(storage), 5u);
    EXPECT_TRUE(StableChar::is_small(storage));
    EXPECT_STREQ(StableChar::data(storage), "hello");
}

TEST(StableLayoutPolicyChar, InitSmallMaxSize) {
    // NOTE: stable_layout_policy metadata encoding = kSmallCapacity - size.
    // When size == kSmallCapacity, metadata == 0 == CharT{}, which collides
    // with the heap sentinel. The safe maximum small size is kSmallCapacity - 1.
    // This is a known boundary bug to be fixed in Milestone 2.
    StableChar::StorageType storage;
    constexpr auto safe_max = StableChar::kSmallCapacity - 1;
    std::string s(safe_max, 'x');
    StableChar::init_small(storage, s.data(), s.size());

    EXPECT_EQ(StableChar::size(storage), safe_max);
    EXPECT_TRUE(StableChar::is_small(storage));
}

TEST(StableLayoutPolicyChar, InitSmallFullCapacityIsBroken) {
    // When size == kSmallCapacity, metadata = 0 collides with heap sentinel.
    // This documents the known encoding boundary bug.
    StableChar::StorageType storage;
    std::string s(StableChar::kSmallCapacity, 'x');
    StableChar::init_small(storage, s.data(), s.size());

    EXPECT_FALSE(StableChar::is_small(storage));
}

TEST(StableLayoutPolicyChar, InitSmallEmbeddedNull) {
    StableChar::StorageType storage;
    const char data[] = {'a', '\0', 'b'};
    StableChar::init_small(storage, data, 3);

    EXPECT_EQ(StableChar::size(storage), 3u);
    EXPECT_EQ(StableChar::data(storage)[0], 'a');
    EXPECT_EQ(StableChar::data(storage)[1], '\0');
    EXPECT_EQ(StableChar::data(storage)[2], 'b');
}

TEST(StableLayoutPolicyChar, InitHeap) {
    StableChar::StorageType storage;
    const std::size_t cap = 64;
    char* ptr = new char[cap + 1];
    std::memcpy(ptr, "heap string test", 16);
    ptr[16] = '\0';

    StableChar::init_heap(storage, ptr, 16, cap);

    EXPECT_EQ(StableChar::size(storage), 16u);
    EXPECT_EQ(StableChar::capacity(storage), cap);
    EXPECT_TRUE(StableChar::is_heap(storage));
    EXPECT_FALSE(StableChar::is_small(storage));
    EXPECT_STREQ(StableChar::data(storage), "heap string test");

    delete[] ptr;
}

TEST(StableLayoutPolicyChar, SetSizeSmall) {
    // set_size only updates metadata; caller must write null terminator
    // to maintain data()[size()] == CharT{} invariant.
    StableChar::StorageType storage;
    StableChar::init_small(storage, "hello", 5);

    StableChar::set_size(storage, 3);
    storage.small[3] = '\0';
    EXPECT_EQ(StableChar::size(storage), 3u);
    EXPECT_EQ(StableChar::data(storage)[3], '\0');
}

TEST(StableLayoutPolicyChar, DestroyHeap) {
    StableChar::StorageType storage;
    char* ptr = new char[65];
    StableChar::init_heap(storage, ptr, 10, 64);

    StableChar::destroy_heap(storage);
    EXPECT_EQ(StableChar::heap_ptr(storage), nullptr);
    EXPECT_EQ(StableChar::size(storage), 0u);

    delete[] ptr;
}

// ============================================================
// stable_layout_policy<char16_t> tests
// ============================================================

using StableChar16 = StableLayoutPolicy<char16_t>;

TEST(StableLayoutPolicyChar16, InitEmpty) {
    StableChar16::StorageType storage;
    StableChar16::init_empty(storage);

    EXPECT_EQ(StableChar16::size(storage), 0u);
    EXPECT_TRUE(StableChar16::is_small(storage));
}

TEST(StableLayoutPolicyChar16, InitSmall) {
    StableChar16::StorageType storage;
    const char16_t data[] = {u'h', u'i', u'\0'};
    StableChar16::init_small(storage, data, 2);

    EXPECT_EQ(StableChar16::size(storage), 2u);
    EXPECT_TRUE(StableChar16::is_small(storage));
    EXPECT_EQ(StableChar16::data(storage)[0], u'h');
    EXPECT_EQ(StableChar16::data(storage)[1], u'i');
}

// ============================================================
// stable_layout_policy<char32_t> tests
// ============================================================

using StableChar32 = StableLayoutPolicy<char32_t>;

TEST(StableLayoutPolicyChar32, InitEmpty) {
    StableChar32::StorageType storage;
    StableChar32::init_empty(storage);

    EXPECT_EQ(StableChar32::size(storage), 0u);
    EXPECT_TRUE(StableChar32::is_small(storage));
}

TEST(StableLayoutPolicyChar32, InitSmall) {
    StableChar32::StorageType storage;
    const char32_t data[] = {U'A', U'B', U'\0'};
    StableChar32::init_small(storage, data, 2);

    EXPECT_EQ(StableChar32::size(storage), 2u);
    EXPECT_EQ(StableChar32::data(storage)[0], U'A');
    EXPECT_EQ(StableChar32::data(storage)[1], U'B');
}

// ============================================================
// stable_layout_policy<wchar_t> tests
// ============================================================

using StableWChar = StableLayoutPolicy<wchar_t>;

TEST(StableLayoutPolicyWChar, InitEmpty) {
    StableWChar::StorageType storage;
    StableWChar::init_empty(storage);

    EXPECT_EQ(StableWChar::size(storage), 0u);
    EXPECT_TRUE(StableWChar::is_small(storage));
}

TEST(StableLayoutPolicyWChar, InitSmall) {
    StableWChar::StorageType storage;
    const wchar_t data[] = {L'w', L'z', L'\0'};
    StableWChar::init_small(storage, data, 2);

    EXPECT_EQ(StableWChar::size(storage), 2u);
    EXPECT_EQ(StableWChar::data(storage)[0], L'w');
    EXPECT_EQ(StableWChar::data(storage)[1], L'z');
}

// ============================================================
// compact_layout_policy<char> - placeholder (Milestone 12)
// ============================================================

TEST(CompactLayoutPolicyPlaceholder, Placeholder) {
    GTEST_SKIP() << "Milestone 12: Implement compact_layout_policy tests";
}

}// namespace test
}// namespace aethermind
