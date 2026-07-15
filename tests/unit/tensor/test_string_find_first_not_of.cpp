//
// Created by 赵丹 on 2025/8/22.
//
#include "any.h"
#include "container/string.h"

#include <deque>
#include <gtest/gtest.h>
#include <list>

namespace {

using namespace aethermind;

// 测试 find_first_not_of(value_type c, size_type pos = 0) const noexcept
TEST(StringFindFirstNotOf, SingleChar) {
    // 基本功能测试
    String s1 = "aaaaabaaaa";
    EXPECT_EQ(s1.find_first_not_of('a'), 5);
    EXPECT_EQ(s1.find_first_not_of('a', 0), 5);
    EXPECT_EQ(s1.find_first_not_of('a', 5), 5);
    EXPECT_EQ(s1.find_first_not_of('a', 6), String::npos);
    EXPECT_EQ(s1.find_first_not_of('b'), 0);
    EXPECT_EQ(s1.find_first_not_of('x'), 0);

    // 空字符串测试
    String s2;
    EXPECT_EQ(s2.find_first_not_of('a'), String::npos);

    // 全部相同字符
    String s3(10, 'c');
    EXPECT_EQ(s3.find_first_not_of('c'), String::npos);

    // 特殊字符
    String s4 = "   \t\n   ";
    EXPECT_EQ(s4.find_first_not_of(' '), 3);
    EXPECT_EQ(s4.find_first_not_of('\t'), 0);

    // // 非ASCII字符
    // String s5 = "你好你好世界";
    // EXPECT_EQ(s5.find_first_not_of("你"[0]), 2);
}

// 测试 find_first_not_of(const_pointer str, size_type pos = 0) const noexcept
TEST(StringFindFirstNotOf, CString) {
    // 基本功能测试
    String s1 = "abcabcabc";
    EXPECT_EQ(s1.find_first_not_of("abc"), String::npos);
    EXPECT_EQ(s1.find_first_not_of("ab"), 2);
    EXPECT_EQ(s1.find_first_not_of("bc"), 0);
    EXPECT_EQ(s1.find_first_not_of("xyz"), 0);

    // 带有pos参数
    EXPECT_EQ(s1.find_first_not_of("ab", 3), 5);
    EXPECT_EQ(s1.find_first_not_of("abc", 10), String::npos);

    // 空字符串测试
    String s2;
    EXPECT_EQ(s2.find_first_not_of("abc"), String::npos);

    // 空查找字符串
    String s3 = "hello";
    EXPECT_EQ(s3.find_first_not_of(""), String::npos);

    // 特殊字符
    String s4 = "abc!@#def";
    EXPECT_EQ(s4.find_first_not_of("abc"), 3);
    EXPECT_EQ(s4.find_first_not_of("!@#"), 0);
    EXPECT_EQ(s4.find_first_not_of("abc!@#"), 6);
}

// 测试 find_first_not_of(const String& str, size_type pos = 0) const noexcept
TEST(StringFindFirstNotOf, String) {
    // 基本功能测试
    String s1 = "abcabcabc";
    String s2 = "abc";
    String s3 = "ab";
    String s4 = "xyz";

    EXPECT_EQ(s1.find_first_not_of(s2), String::npos);
    EXPECT_EQ(s1.find_first_not_of(s3), 2);
    EXPECT_EQ(s1.find_first_not_of(s4), 0);

    // 带有pos参数
    EXPECT_EQ(s1.find_first_not_of(s3, 3), 5);
    EXPECT_EQ(s1.find_first_not_of(s2, 10), String::npos);

    // 空字符串测试
    String s5;
    EXPECT_EQ(s5.find_first_not_of(s2), String::npos);

    // 空查找字符串
    String s6 = "hello";
    String s7;
    EXPECT_EQ(s6.find_first_not_of(s7), String::npos);

    // 长字符串
    String s8(1000, 'a');
    String s9(500, 'a');
    EXPECT_EQ(s8.find_first_not_of(s9), String::npos);
}

// 测试 find_first_not_of(const_pointer s, size_type pos, size_type n) const noexcept
TEST(StringFindFirstNotOf, CStringWithLength) {
    // 基本功能测试
    String s1 = "abcabcabc";
    EXPECT_EQ(s1.find_first_not_of("abcxyz", 0, 3), String::npos);
    EXPECT_EQ(s1.find_first_not_of("abcxyz", 0, 2), 2);
    EXPECT_EQ(s1.find_first_not_of("xyzabc", 0, 3), 0);

    // 带有pos参数
    EXPECT_EQ(s1.find_first_not_of("abxyz", 3, 2), 5);
    EXPECT_EQ(s1.find_first_not_of("abcxyz", 10, 3), String::npos);

    // n=0 测试
    EXPECT_EQ(s1.find_first_not_of("abc", 0, 0), String::npos);

    // 部分匹配
    String s2 = "abcdef";
    EXPECT_EQ(s2.find_first_not_of("abcxyz", 0, 5), 3);

    // 重叠字符
    String s3 = "123456789";
    EXPECT_EQ(s3.find_first_not_of("12345", 0, 5), 5);
    EXPECT_EQ(s3.find_first_not_of("12345", 3, 5), 5);
}

// 测试边界条件和特殊情况
TEST(StringFindFirstNotOf, EdgeCases) {
    // pos 超出范围
    String s1 = "hello";
    EXPECT_EQ(s1.find_first_not_of('h', 10), String::npos);
    EXPECT_EQ(s1.find_first_not_of("hello", 10), String::npos);

    // 单个字符字符串
    String s2 = "x";
    EXPECT_EQ(s2.find_first_not_of('x'), String::npos);
    EXPECT_EQ(s2.find_first_not_of('y'), 0);

    // 混合字符
    String s3 = "a1b2c3";
    EXPECT_EQ(s3.find_first_not_of("abc"), 1);
    EXPECT_EQ(s3.find_first_not_of("123"), 0);

    // 只有一个不同字符在末尾
    String s4 = "aaaaab";
    EXPECT_EQ(s4.find_first_not_of('a'), 5);

    // 只有一个不同字符在开头
    String s5 = "aabbbb";
    EXPECT_EQ(s5.find_first_not_of('b'), 0);
}

}  // namespace
