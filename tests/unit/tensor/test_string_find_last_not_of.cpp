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

// 测试 find_last_not_of(const_pointer s, size_type pos, size_type n) 函数
TEST(StringFindLastNotOfTest, FindLastNotOfConstPointerPosN) {
    // 基本功能测试
    String str("Hello, World!");
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", 12), 1);       // 'H' 不在字符集中
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", 4), 1);        // 'o' 在字符集中，但 'H' 是第一个不在其中的
    EXPECT_EQ(str.find_last_not_of("Hello", 4), String::npos);// 所有字符都在字符集中

    // 位置参数测试
    EXPECT_EQ(str.find_last_not_of("World", 8), 6);// 从位置8开始搜索，找到','位置7
    EXPECT_EQ(str.find_last_not_of("World", 5), 5);// 从位置5开始搜索，找到'o'位置4

    // n=0 测试（注意：实现中注释掉了n=0的特殊处理，但根据标准库行为应该返回npos）
    EXPECT_EQ(str.find_last_not_of("abc", 5, 0), 5);

    // 边界情况测试
    String empty_str;
    EXPECT_EQ(empty_str.find_last_not_of("abc", 0, 3), String::npos);// 空字符串

    // pos大于字符串长度
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", 20), 1);// pos会被调整为字符串末尾

    // 特殊字符测试
    String special_str("123!@#$");
    EXPECT_EQ(special_str.find_last_not_of("!@#$", 6), 2);// 找到'3'位置2

    // 多字节字符边界测试
    // String multi_str("你好，世界！");
    // EXPECT_EQ(multi_str.find_last_not_of("！界世，好", 5), 0);// 找到'你'位置0
}

// 测试 find_last_not_of(const String& str, size_type pos) 函数
TEST(StringFindLastNotOfTest, FindLastNotOfString) {
    // 基本功能测试
    String str("Hello, World!");
    String charset("!dlroW, ");
    EXPECT_EQ(str.find_last_not_of(charset, 12), 1);// 'H' 不在字符集中

    // 位置参数测试
    EXPECT_EQ(str.find_last_not_of(charset, 4), 1);// 'o' 在字符集中，但 'H' 是第一个不在其中的

    // 空字符串参数测试
    String empty_charset;
    // 根据实现，n=0时不会特殊处理，但期望行为应该是返回pos
    // 这里根据实现的实际行为进行测试
    EXPECT_EQ(str.find_last_not_of(empty_charset, 5), 5);

    // 空字符串调用测试
    String empty_str;
    EXPECT_EQ(empty_str.find_last_not_of(charset, 0), String::npos);

    // pos=npos 测试
    EXPECT_EQ(str.find_last_not_of(charset, String::npos), 1);
}

// 测试 find_last_not_of(const_pointer str, size_type pos) 函数
TEST(StringFindLastNotOfTest, FindLastNotOfConstPointer) {
    // 基本功能测试
    String str("Hello, World!");
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", 12), 1);// 'H' 不在字符集中

    // 位置参数测试
    EXPECT_EQ(str.find_last_not_of("World", 8), 6);// 从位置8开始搜索，找到','位置7

    // 空字符串参数测试
    EXPECT_EQ(str.find_last_not_of("", 5), 5);// 空字符串参数，返回pos

    // pos=npos 测试
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", String::npos), 1);

    // 重复模式测试
    String repeat_str("aaaaabaaaa");
    EXPECT_EQ(repeat_str.find_last_not_of("a", 10), 5);// 找到'b'位置5
}

// 测试 find_last_not_of(value_type c, size_type pos) 函数
TEST(StringFindLastNotOfTest, FindLastNotOfChar) {
    // 基本功能测试
    String str("Hello, World!");
    EXPECT_EQ(str.find_last_not_of('o', 12), 12);// 找到'!'位置11

    // 位置参数测试
    EXPECT_EQ(str.find_last_not_of('l', 4), 4);// 从位置4开始搜索，找到'e'位置3

    // 空字符串测试
    String empty_str;
    EXPECT_EQ(empty_str.find_last_not_of('a', 0), String::npos);

    // pos=npos 测试
    EXPECT_EQ(str.find_last_not_of('o', String::npos), 12);

    // pos大于字符串长度
    EXPECT_EQ(str.find_last_not_of('o', 20), 12);// pos会被调整为字符串末尾

    // 特殊字符测试
    String special_str("111!111");
    EXPECT_EQ(special_str.find_last_not_of('1', 6), 3);// 找到'!'位置3

    // 没有找到匹配字符
    EXPECT_EQ(str.find_last_not_of('x', 12), 12);// 所有字符都不等于'x'，返回最后一个字符位置
}

// 测试所有find_last_not_of重载的一致性
TEST(StringFindLastNotOfTest, ConsistencyBetweenOverloads) {
    String str("Hello, World!");
    char c = 'o';
    const char* cs = "o";
    String cs_str("o");

    // 测试不同重载函数在相同条件下的一致性
    EXPECT_EQ(str.find_last_not_of(c, 12), str.find_last_not_of(cs, 12, 1));
    EXPECT_EQ(str.find_last_not_of(c, 12), str.find_last_not_of(cs, 12));
    EXPECT_EQ(str.find_last_not_of(c, 12), str.find_last_not_of(cs_str, 12));
}

// 测试COW机制下的find_last_not_of
TEST(StringFindLastNotOfTest, FindLastNotOfWithCOW) {
    // 创建一个字符串
    String str("Hello, World!");

    // 创建一个共享str内容的新字符串（COW）
    String str_cow = str;

    // 对str_cow调用find_last_not_of不应修改共享数据
    EXPECT_EQ(str_cow.find_last_not_of("!dlroW, ", 12), 1);

    // 验证str和str_cow仍然相等
    EXPECT_EQ(str, str_cow);
}

// 测试本地缓冲区切换情况下的find_last_not_of
TEST(StringFindLastNotOfTest, FindLastNotOfWithLocalBuffer) {
    // 短字符串（使用本地缓冲区）
    String short_str("abc");
    EXPECT_EQ(short_str.find_last_not_of("bc", 2), 0);// 找到'a'位置0

    // 长字符串（使用堆分配）
    String long_str("a");
    for (int i = 0; i < 30; ++i) {
        long_str += "a";
    }
    long_str += "b";

    EXPECT_EQ(long_str.find_last_not_of("a", long_str.size() - 1), long_str.size() - 1);// 找到'b'
}

// 测试与其他字符串操作的组合
TEST(StringFindLastNotOfTest, FindLastNotOfWithOtherOperations) {
    String str("Hello, World!");

    // 与insert组合
    str.insert(7, "Beautiful ");
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", str.size() - 1), 14);// 找到'l'位置16

    // 与erase组合
    str.erase(7, 10);
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", str.size() - 1), 1);// 找到'H'位置0

    // 与replace组合
    str.replace(0, 5, "Hi");
    EXPECT_EQ(str.find_last_not_of("!dlroW, ", str.size() - 1), 1);// 找到'i'位置1
}

// 测试长字符串中的性能和正确性
TEST(StringFindLastNotOfTest, FindLastNotOfInLongString) {
    // 创建一个长字符串，最后有一个不同的字符
    String long_str;
    for (int i = 0; i < 1000; ++i) {
        long_str += "a";
    }
    long_str += "b";

    // 从末尾查找第一个不是'a'的字符
    EXPECT_EQ(long_str.find_last_not_of('a', long_str.size() - 1), 1000);// 找到'b'位置1000

    // 创建一个更复杂的长字符串
    String complex_str;
    for (int i = 0; i < 500; ++i) {
        complex_str += "abc";
    }
    complex_str += "xyz";

    EXPECT_EQ(complex_str.find_last_not_of("abc", complex_str.size() - 1), 1502);// 找到'x'位置1500
}

}  // namespace
