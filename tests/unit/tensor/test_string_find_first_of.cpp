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

// 测试 find_first_of(const_pointer s, size_type pos, size_type n) const noexcept
TEST(StringFindFirstOfTest, FindFirstOfConstPointerWithSize) {
    String str("hello world");

    // 基本功能测试
    EXPECT_EQ(1U, str.find_first_of("aeiou", 0, 5));        // 第一个元音字母'e'在位置1
    EXPECT_EQ(4U, str.find_first_of("od", 0, 2));           // 'o'在位置4，但'd'在位置3
    EXPECT_EQ(String::npos, str.find_first_of("xyz", 0, 3));// 没有匹配，返回npos

    // pos参数测试
    EXPECT_EQ(4U, str.find_first_of("aeiou", 2, 5));// 从位置2开始查找，找到'o'在位置4
    EXPECT_EQ(7U, str.find_first_of("aeiou", 6, 5));// 从位置6开始查找，没有找到元音

    // n=0的特殊情况
    EXPECT_EQ(String::npos, str.find_first_of("any", 0, 0));

    // pos等于size()
    EXPECT_EQ(String::npos, str.find_first_of("a", str.size(), 1));

    // 多个匹配字符，应该返回第一个出现的位置
    EXPECT_EQ(1U, str.find_first_of("elo", 0, 3));// 'e'在位置1，是第一个匹配

    // 重复字符测试
    String repeats("aaaaa");
    EXPECT_EQ(0U, repeats.find_first_of("a", 0, 1));
    EXPECT_EQ(2U, repeats.find_first_of("a", 2, 1));
}

// 测试 find_first_of(const String& str, size_type pos = 0) const noexcept
TEST(StringFindFirstOfTest, FindFirstOfStringOverload) {
    String str("hello world");
    String chars1("aeiou");
    String chars2("od");
    String chars3("xyz");

    // 基本功能测试
    EXPECT_EQ(1U, str.find_first_of(chars1, 0));          // 第一个元音字母'e'在位置1
    EXPECT_EQ(4U, str.find_first_of(chars2, 0));          // 'o'在位置4，但'd'在位置3
    EXPECT_EQ(String::npos, str.find_first_of(chars3, 0));// 没有匹配，返回npos

    // pos参数测试
    EXPECT_EQ(4U, str.find_first_of(chars1, 2));// 从位置2开始查找，找到'o'在位置4
    EXPECT_EQ(7U, str.find_first_of(chars1, 6));// 从位置6开始查找，没有找到元音

    // 默认pos参数
    EXPECT_EQ(1U, str.find_first_of(chars1));// 使用默认pos=0

    // 空字符串参数
    String empty;
    EXPECT_EQ(String::npos, str.find_first_of(empty, 0));

    // 字符串等于自身中的字符集
    EXPECT_EQ(0U, str.find_first_of(str, 0));
}

// 测试 find_first_of(const_pointer str, size_type pos = 0) const noexcept
TEST(StringFindFirstOfTest, FindFirstOfConstPointerOverload) {
    String str("hello world");

    // 基本功能测试
    EXPECT_EQ(1U, str.find_first_of("aeiou", 0));        // 第一个元音字母'e'在位置1
    EXPECT_EQ(4U, str.find_first_of("od", 0));           // 'o'在位置4，但'd'在位置3
    EXPECT_EQ(String::npos, str.find_first_of("xyz", 0));// 没有匹配，返回npos

    // pos参数测试
    EXPECT_EQ(4U, str.find_first_of("aeiou", 2));// 从位置2开始查找，找到'o'在位置4
    EXPECT_EQ(7U, str.find_first_of("aeiou", 6));// 从位置6开始查找，没有找到元音

    // 默认pos参数
    EXPECT_EQ(1U, str.find_first_of("aeiou"));// 使用默认pos=0

    // 空字符串参数
    EXPECT_EQ(String::npos, str.find_first_of("", 0));

    // 部分匹配测试
    EXPECT_EQ(1U, str.find_first_of("aeiouxyz", 0));// 只要有一个字符匹配即可
}

// 测试 find_first_of(value_type c, size_type pos = 0) const noexcept
TEST(StringFindFirstOfTest, FindFirstOfCharOverload) {
    String str("hello world");

    // 基本功能测试
    EXPECT_EQ(1U, str.find_first_of('e', 0));          // 'e'在位置1
    EXPECT_EQ(4U, str.find_first_of('o', 0));          // 'o'在位置4
    EXPECT_EQ(String::npos, str.find_first_of('x', 0));// 没有匹配，返回npos

    // pos参数测试
    EXPECT_EQ(7U, str.find_first_of('o', 5));          // 从位置5开始查找，找到第二个'o'在位置7
    EXPECT_EQ(String::npos, str.find_first_of('h', 1));// 从位置1开始查找，找不到'h'

    // 默认pos参数
    EXPECT_EQ(1U, str.find_first_of('e'));// 使用默认pos=0

    // 查找重复字符
    String repeats("aaaaa");
    EXPECT_EQ(0U, repeats.find_first_of('a', 0));
    EXPECT_EQ(2U, repeats.find_first_of('a', 2));
    EXPECT_EQ(4U, repeats.find_first_of('a', 4));
}

// 边界情况测试
TEST(StringFindFirstOfTest, BoundaryCases) {
    // 空字符串
    String empty;
    String nonEmpty("test");

    // 空字符串作为被查找对象
    EXPECT_EQ(String::npos, empty.find_first_of("test", 0, 4));
    EXPECT_EQ(String::npos, empty.find_first_of(nonEmpty, 0));
    EXPECT_EQ(String::npos, empty.find_first_of("test", 0));
    EXPECT_EQ(String::npos, empty.find_first_of('t', 0));

    // 单字符字符串
    String single("a");
    EXPECT_EQ(0U, single.find_first_of("a", 0, 1));
    EXPECT_EQ(0U, single.find_first_of(String("a"), 0));
    EXPECT_EQ(0U, single.find_first_of("a", 0));
    EXPECT_EQ(0U, single.find_first_of('a', 0));
    EXPECT_EQ(String::npos, single.find_first_of("b", 0, 1));
}

// 特殊字符测试
TEST(StringFindFirstOfTest, SpecialCharacters) {
    // 包含空格、制表符、换行符
    String str("space\ttab\nnewline");

    EXPECT_EQ(5U, str.find_first_of(" \t\n", 0, 3));// 空格在位置5
    EXPECT_EQ(5U, str.find_first_of(String(" \t\n"), 0));
    EXPECT_EQ(5U, str.find_first_of(" \t\n", 0));
    EXPECT_EQ(String::npos, str.find_first_of(' ', 0));

    // 非ASCII字符
    // String unicode("你好，世界！");
    //
    // EXPECT_EQ(4U, str.find_first_of("，！", 0, 2));// 逗号在位置4（假设UTF-8编码）
    // EXPECT_EQ(4U, str.find_first_of(String("，！"), 0));
    // EXPECT_EQ(4U, str.find_first_of("，！", 0));
}

// 功能一致性测试
TEST(StringFindFirstOfTest, ConsistencyBetweenOverloads) {
    String str("hello world");
    String chars("aeiou");

    // 验证四个重载版本对相同输入返回相同结果
    EXPECT_EQ(str.find_first_of("aeiou", 0, 5), str.find_first_of(chars, 0));
    EXPECT_EQ(str.find_first_of("aeiou", 0, 5), str.find_first_of("aeiou", 0));
    EXPECT_EQ(str.find_first_of('e', 0), str.find_first_of("e", 0, 1));
    EXPECT_EQ(str.find_first_of('e', 0), str.find_first_of("e", 0));

    // 从不同位置开始查找
    EXPECT_EQ(str.find_first_of("aeiou", 2, 5), str.find_first_of(chars, 2));
    EXPECT_EQ(str.find_first_of("aeiou", 2, 5), str.find_first_of("aeiou", 2));
}

// 性能相关测试 - 长字符串中的查找
TEST(StringFindFirstOfTest, PerformanceRelated) {
    // 创建一个较长的字符串
    String longStr;
    longStr.reserve(1000);
    for (int i = 0; i < 500; ++i) {
        longStr.append("a");
    }
    longStr.append("b");
    for (int i = 0; i < 500; ++i) {
        longStr.append("a");
    }

    // 应该快速找到第一个非'a'字符
    EXPECT_EQ(500U, longStr.find_first_of("b", 0, 1));
    EXPECT_EQ(500U, longStr.find_first_of(String("b"), 0));
    EXPECT_EQ(500U, longStr.find_first_of("b", 0));
    EXPECT_EQ(500U, longStr.find_first_of('b', 0));

    // 从中间位置查找
    EXPECT_EQ(500U, longStr.find_first_of("b", 499, 1));
}

// 组合操作测试
TEST(StringFindFirstOfTest, CombinedOperations) {
    String str("abcdefghijklmnopqrstuvwxyz");

    // 查找后插入
    auto pos = str.find_first_of("jkl", 0, 3);
    str.insert(pos, "INSERTED");
    EXPECT_NE(pos, str.find_first_of("jkl", 0, 3));

    // 查找后删除
    pos = str.find_first_of("def", 0, 3);
    str.erase(pos, 3);

    // 查找后替换
    pos = str.find_first_of("g", 0, 1);
    str.replace(pos, 1, "XYZ");
}

// 多字符集测试
TEST(StringFindFirstOfTest, MultiCharSet) {
    String str("The quick brown fox jumps over the lazy dog");

    // 查找任何元音字母
    EXPECT_EQ(2U, str.find_first_of("aeiouAEIOU", 0, 10));// 'e'在位置2

    // 查找任何标点符号或空格
    EXPECT_EQ(3U, str.find_first_of(" .,;!", 0, 5));// 空格在位置3

    // 查找任何数字（应该找不到）
    EXPECT_EQ(String::npos, str.find_first_of("0123456789", 0, 10));
}

}  // namespace
