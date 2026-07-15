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

// 测试 rfind(const_pointer s, size_type pos, size_type n) const noexcept
TEST(StringRFindTest, RFindConstPointerWithSize) {
    String str("hello world hello");

    // 基本功能测试
    EXPECT_EQ(12U, str.rfind("hello", String::npos, 5));        // 找到第二个"hello"
    EXPECT_EQ(0U, str.rfind("hello", 5, 5));                    // 从位置5向前查找，找到第一个"hello"
    EXPECT_EQ(6U, str.rfind("world", String::npos, 5));         // 找到"world"
    EXPECT_EQ(String::npos, str.rfind("test", String::npos, 4));// 找不到

    // pos参数测试
    EXPECT_EQ(0U, str.rfind("hello", 10, 5)); // 从位置10向前查找，找到第一个"hello"
    EXPECT_EQ(12U, str.rfind("hello", 15, 5));// 从位置15向前查找，找到第二个"hello"

    // n参数测试
    EXPECT_EQ(12U, str.rfind("helloworld", String::npos, 5));// 只比较前5个字符

    // n大于字符串长度
    EXPECT_EQ(String::npos, str.rfind("hello", String::npos, 10));// n=10大于字符串长度

    // 空字符串参数(n=0)
    EXPECT_EQ(17, str.rfind("", String::npos, 0));

    // 子串位于开头
    EXPECT_EQ(12U, str.rfind("he", String::npos, 2));

    // 子串位于结尾
    EXPECT_EQ(15U, str.rfind("lo", String::npos, 2));
}

// 测试 rfind(const String& str, size_type pos = npos) const noexcept
TEST(StringRFindTest, RFindStringOverload) {
    String str("hello world hello");
    String substr1("hello");
    String substr2("world");
    String substr3("test");

    // 基本功能测试
    EXPECT_EQ(12U, str.rfind(substr1, String::npos));         // 找到第二个"hello"
    EXPECT_EQ(0U, str.rfind(substr1, 5));                     // 从位置5向前查找，找到第一个"hello"
    EXPECT_EQ(6U, str.rfind(substr2, String::npos));          // 找到"world"
    EXPECT_EQ(String::npos, str.rfind(substr3, String::npos));// 找不到

    // 默认pos参数
    EXPECT_EQ(12U, str.rfind(substr1));// 使用默认pos=npos

    // 空字符串参数
    String empty;
    EXPECT_EQ(17, str.rfind(empty, String::npos));

    // 字符串等于自身
    EXPECT_EQ(0U, str.rfind(str, String::npos));// 完整匹配，应该返回0
}

// 测试 rfind(const_pointer str, size_type pos = npos) const noexcept
TEST(StringRFindTest, RFindConstPointerOverload) {
    String str("hello world hello");

    // 基本功能测试
    EXPECT_EQ(12U, str.rfind("hello", String::npos));        // 找到第二个"hello"
    EXPECT_EQ(0U, str.rfind("hello", 5));                    // 从位置5向前查找，找到第一个"hello"
    EXPECT_EQ(6U, str.rfind("world", String::npos));         // 找到"world"
    EXPECT_EQ(String::npos, str.rfind("test", String::npos));// 找不到

    // 默认pos参数
    EXPECT_EQ(12U, str.rfind("hello"));// 使用默认pos=npos

    // 空字符串参数
    EXPECT_EQ(17, str.rfind("", String::npos));

    // 部分匹配测试
    EXPECT_EQ(String::npos, str.rfind("hello123", String::npos));// 前缀匹配
}

// 测试 rfind(value_type c, size_type pos = npos) const noexcept
TEST(StringRFindTest, RFindCharOverload) {
    String str("hello world hello");

    // 基本功能测试
    EXPECT_EQ(12U, str.rfind('h', String::npos));         // 找到第二个'h'
    EXPECT_EQ(7U, str.rfind('o', 10));                    // 从位置10向前查找，找到第一个'o'
    EXPECT_EQ(16U, str.rfind('o', String::npos));         // 找到最后一个'o'
    EXPECT_EQ(String::npos, str.rfind('x', String::npos));// 找不到

    // 默认pos参数
    EXPECT_EQ(12U, str.rfind('h'));// 使用默认pos=npos

    // 查找重复字符
    String repeats("aaaaa");
    EXPECT_EQ(4U, repeats.rfind('a', String::npos));
    EXPECT_EQ(2U, repeats.rfind('a', 2));
    EXPECT_EQ(0U, repeats.rfind('a', 0));
}

// 边界情况测试
TEST(StringRFindTest, BoundaryCases) {
    // 空字符串
    String empty;
    String nonEmpty("test");

    // 空字符串作为被查找对象
    EXPECT_EQ(String::npos, empty.rfind("test", String::npos, 4));
    EXPECT_EQ(String::npos, empty.rfind(nonEmpty, String::npos));
    EXPECT_EQ(String::npos, empty.rfind("test", String::npos));
    EXPECT_EQ(String::npos, empty.rfind('t', String::npos));

    // 单字符字符串
    String single("a");
    EXPECT_EQ(0U, single.rfind("a", String::npos, 1));
    EXPECT_EQ(0U, single.rfind(String("a"), String::npos));
    EXPECT_EQ(0U, single.rfind("a", String::npos));
    EXPECT_EQ(0U, single.rfind('a', String::npos));
    EXPECT_EQ(String::npos, single.rfind("b", String::npos, 1));

    // pos等于size()-1
    String str("test");
    EXPECT_EQ(3U, str.rfind('t', 3));

    // pos小于0的情况（通过unsigned的环绕处理）
    // 注意：由于pos是unsigned类型，-1会变成非常大的值
    EXPECT_EQ(3U, str.rfind('t', static_cast<size_t>(-1)));
}

// 特殊字符测试
TEST(StringRFindTest, SpecialCharacters) {
    // 包含空格、制表符、换行符
    String str("space\ttab\nnewline");

    EXPECT_EQ(String::npos, str.rfind(" \t\n", String::npos, 3));// 换行符在位置9
    EXPECT_EQ(String::npos, str.rfind(String(" \t\n"), String::npos));
    EXPECT_EQ(String::npos, str.rfind(" \t\n", String::npos));
    EXPECT_EQ(9U, str.rfind('\n', String::npos));

    // 非ASCII字符
    // String unicode("你好，世界！你好");
    //
    // EXPECT_EQ(10U, unicode.rfind("你好", String::npos, 4));// 第二个"你好"在位置10（假设UTF-8编码）
    // EXPECT_EQ(10U, unicode.rfind(String("你好"), String::npos));
    // EXPECT_EQ(10U, unicode.rfind("你好", String::npos));
}

// 功能一致性测试
TEST(StringRFindTest, ConsistencyBetweenOverloads) {
    String str("hello world hello");
    String substr("hello");

    // 验证四个重载版本对相同输入返回相同结果
    EXPECT_EQ(str.rfind("hello", String::npos, 5), str.rfind(substr, String::npos));
    EXPECT_EQ(str.rfind("hello", String::npos, 5), str.rfind("hello", String::npos));
    EXPECT_EQ(str.rfind('h', String::npos), str.rfind("h", String::npos, 1));
    EXPECT_EQ(str.rfind('h', String::npos), str.rfind("h", String::npos));

    // 从不同位置开始查找
    EXPECT_EQ(str.rfind("hello", 10, 5), str.rfind(substr, 10));
    EXPECT_EQ(str.rfind("hello", 10, 5), str.rfind("hello", 10));
}

// 性能相关测试 - 长字符串中的从右查找
TEST(StringRFindTest, PerformanceRelated) {
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

    // 应该快速找到最后一个非'a'字符
    EXPECT_EQ(500U, longStr.rfind("b", String::npos, 1));
    EXPECT_EQ(500U, longStr.rfind(String("b"), String::npos));
    EXPECT_EQ(500U, longStr.rfind("b", String::npos));
    EXPECT_EQ(500U, longStr.rfind('b', String::npos));

    // 从中间位置向前查找
    EXPECT_EQ(500U, longStr.rfind("b", 500, 1));
}

// 组合操作测试
TEST(StringRFindTest, CombinedOperations) {
    String str("abcdefghijklmnopqrstuvwxyz");

    // 查找后插入
    auto pos = str.rfind("jkl", String::npos, 3);
    str.insert(pos, "INSERTED");
    EXPECT_NE(pos, str.rfind("jkl", String::npos, 3));

    // 查找后删除
    pos = str.rfind("def", String::npos, 3);
    str.erase(pos, 3);

    // 查找后替换
    pos = str.rfind("g", String::npos, 1);
    str.replace(pos, 1, "XYZ");
}

// 多重复子串测试
TEST(StringRFindTest, MultipleOccurrences) {
    String str("ababababab");

    // 查找重复子串
    EXPECT_EQ(8U, str.rfind("ab", String::npos, 2));// 最后一个"ab"
    EXPECT_EQ(6U, str.rfind("ab", 7, 2));           // 从位置7向前查找
    EXPECT_EQ(4U, str.rfind("ab", 5, 2));           // 从位置5向前查找

    // 查找更长的重复子串
    EXPECT_EQ(6U, str.rfind("aba", String::npos, 3));// 最后一个"aba"
    EXPECT_EQ(4U, str.rfind("aba", 5, 3));           // 从位置5向前查找
}

}  // namespace
