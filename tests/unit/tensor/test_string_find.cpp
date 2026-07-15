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

// 基本功能测试
TEST(StringFindTest, BasicFunctionality) {
    // 基本查找功能
    String str("hello world");
    EXPECT_EQ(0, str.find("hello", 0, 5));
    EXPECT_EQ(6, str.find("world", 0, 5));
    EXPECT_EQ(3, str.find("lo", 0, 2));
    EXPECT_EQ(String::npos, str.find("test", 0, 4));
}

// n=0的特殊情况测试
TEST(StringFindTest, EmptyNeedle) {
    String str("hello");
    // 当n=0时，只要pos有效就返回pos
    EXPECT_EQ(0U, str.find("anything", 0, 0));
    EXPECT_EQ(2U, str.find("anything", 2, 0));
    EXPECT_EQ(5U, str.find("anything", 5, 0));          // pos等于size()
    EXPECT_EQ(String::npos, str.find("anything", 6, 0));// pos越界
}

// pos参数测试
TEST(StringFindTest, PositionParameter) {
    String str("abababa");
    // 从不同位置开始查找
    EXPECT_EQ(0U, str.find("aba", 0, 3));
    EXPECT_EQ(2U, str.find("aba", 1, 3));
    EXPECT_EQ(4U, str.find("aba", 3, 3));
    EXPECT_EQ(String::npos, str.find("aba", 5, 3));

    // pos等于size()
    EXPECT_EQ(String::npos, str.find("a", 7, 1));

    // pos超过size()
    EXPECT_EQ(String::npos, str.find("a", 10, 1));
}

// 部分匹配测试
TEST(StringFindTest, PartialMatch) {
    String str("abcdefg");
    // 查找子串的一部分
    EXPECT_EQ(2U, str.find("cde", 0, 3));
    EXPECT_EQ(2U, str.find("cdefxyz", 0, 3));// 只比较前3个字符
}

// 边界情况测试
TEST(StringFindTest, BoundaryCases) {
    // 空字符串
    String empty;
    EXPECT_EQ(0U, empty.find("", 0, 0));// n=0特殊情况
    EXPECT_EQ(String::npos, empty.find("a", 0, 1));

    // 子串位于字符串开头
    String str1("start middle end");
    EXPECT_EQ(0U, str1.find("start", 0, 5));

    // 子串位于字符串结尾
    EXPECT_EQ(13U, str1.find("end", 0, 3));

    // 子串等于整个字符串
    String str2("exact match");
    EXPECT_EQ(0U, str2.find("exact match", 0, 11));
}

// 特殊字符测试
TEST(StringFindTest, SpecialCharacters) {
    // 包含空格、制表符、换行符等
    String str("space\ttab\nnewline");
    EXPECT_EQ(0U, str.find("space", 0, 5));
    EXPECT_EQ(5U, str.find("\t", 0, 1));
    EXPECT_EQ(9U, str.find("\n", 0, 1));
    EXPECT_EQ(6U, str.find("tab", 0, 3));

    // 包含非ASCII字符
    // String unicode("你好，世界！");
    // EXPECT_EQ(0U, unicode.find("你好", 0, 4));// UTF-8中"你好"占4字节
    // EXPECT_EQ(4U, unicode.find("，", 0, 3));  // UTF-8中"，"占3字节
}

// 重复模式测试
TEST(StringFindTest, RepeatedPatterns) {
    String str("aaaaa");
    EXPECT_EQ(0U, str.find("aa", 0, 2));
    EXPECT_EQ(1U, str.find("aa", 1, 2));
    EXPECT_EQ(2U, str.find("aa", 2, 2));
    EXPECT_EQ(3U, str.find("aa", 3, 2));
    EXPECT_EQ(String::npos, str.find("aa", 4, 2));

    String alternating("abababab");
    EXPECT_EQ(0U, alternating.find("abab", 0, 4));
    EXPECT_EQ(2U, alternating.find("abab", 2, 4));
}

// COW机制测试
TEST(StringFindTest, CopyOnWrite) {
    String original("test string for COW");
    String copy = original;// 应该共享数据

    // 验证find操作不会触发COW
    EXPECT_EQ(5U, copy.find("string", 0, 6));

    // 验证数据仍然共享（通过比较内部缓冲区地址）
    // 注意：这里假设可以通过某种方式验证缓冲区共享
    // 实际测试中可能需要使用特定的内部接口或实现细节
}

// 本地缓冲区切换测试
TEST(StringFindTest, LocalBufferSwitching) {
    // 创建一个超过本地缓冲区大小的字符串
    String large(100, 'x');// 假设本地缓冲区大小小于100

    // 在不同位置查找
    EXPECT_EQ(0U, large.find("x", 0, 1));
    EXPECT_EQ(50U, large.find("x", 50, 1));

    // 缩小字符串并验证本地缓冲区切换
    large.resize(10);// 假设这会切换回本地缓冲区
    EXPECT_EQ(5U, large.find("x", 5, 1));
}

// 异常处理测试
TEST(StringFindTest, ExceptionHandling) {
    String str("test");

    // 注意：find函数本身不会抛出异常，但它调用的compare可能会
    // 这里测试pos参数的边界情况
    EXPECT_NO_THROW(UNUSED(str.find("t", 5, 1)));
}

// 多字节字符边界测试
TEST(StringFindTest, MultiByteCharacterBoundaries) {
    String utf8("Hello, 世界！");// 包含ASCII和UTF-8字符

    // 确保在多字节字符边界上正确查找
    EXPECT_EQ(7U, utf8.find("世", 0, 3)); // UTF-8中"世"占3字节
    EXPECT_EQ(10U, utf8.find("界", 0, 3));// UTF-8中"界"占3字节
}

// 组合操作测试
TEST(StringFindTest, CombinedOperations) {
    String str("abcdefghijklmnopqrstuvwxyz");

    // 查找后插入
    auto pos = str.find("jkl", 0, 3);
    str.insert(pos, "INSERTED_");
    EXPECT_EQ(pos, str.find("INSERTED_", 0, 9));
    EXPECT_EQ(pos + 9, str.find("jkl", 0, 3));

    // 查找后删除
    pos = str.find("def", 0, 3);
    str.erase(pos, 3);
    EXPECT_EQ(String::npos, str.find("def", 0, 3));
}

// 测试 find(const String& str, size_type pos = 0)
TEST(StringFindTest, FindStringOverload) {
    // 基本功能测试
    String str("hello world");
    String substr1("hello");
    String substr2("world");
    String substr3("lo");
    String substr4("test");

    EXPECT_EQ(0U, str.find(substr1, 0));
    EXPECT_EQ(6U, str.find(substr2, 0));
    EXPECT_EQ(3U, str.find(substr3, 0));
    EXPECT_EQ(String::npos, str.find(substr4, 0));

    // pos参数测试
    EXPECT_EQ(6U, str.find(substr2, 6));
    EXPECT_EQ(String::npos, str.find(substr2, 7));
    EXPECT_EQ(String::npos, str.find(substr1, 1));

    // pos等于size()
    EXPECT_EQ(String::npos, str.find(substr1, str.size()));

    // 空字符串参数
    String empty;
    EXPECT_EQ(0U, str.find(empty, 0));// 查找空字符串应该返回pos
    EXPECT_EQ(5U, str.find(empty, 5));

    // 字符串等于自身
    EXPECT_EQ(0U, str.find(str, 0));
}

// 测试 find(const_pointer str, size_type pos = 0)
TEST(StringFindTest, FindConstPointerOverload) {
    // 基本功能测试
    String str("hello world");

    EXPECT_EQ(0U, str.find("hello", 0));
    EXPECT_EQ(6U, str.find("world", 0));
    EXPECT_EQ(3U, str.find("lo", 0));
    EXPECT_EQ(String::npos, str.find("test", 0));

    // pos参数测试
    EXPECT_EQ(6U, str.find("world", 6));
    EXPECT_EQ(String::npos, str.find("world", 7));
    EXPECT_EQ(String::npos, str.find("hello", 1));

    // pos等于size()
    EXPECT_EQ(String::npos, str.find("hello", str.size()));

    // 空字符串参数
    EXPECT_EQ(0U, str.find("", 0));// 查找空字符串应该返回pos
    EXPECT_EQ(5U, str.find("", 5));

    // 部分匹配测试
    EXPECT_EQ(String::npos, str.find("hello123", 0));// 前缀匹配
}

// 测试 find(value_type c, size_type pos = 0)
TEST(StringFindTest, FindCharOverload) {
    // 基本功能测试
    String str("hello world");

    EXPECT_EQ(0U, str.find('h', 0));
    EXPECT_EQ(4U, str.find('o', 0));
    EXPECT_EQ(6U, str.find('w', 0));
    EXPECT_EQ(String::npos, str.find('x', 0));

    // pos参数测试
    EXPECT_EQ(7U, str.find('o', 5));          // 找到第二个'o'
    EXPECT_EQ(String::npos, str.find('h', 1));// 从位置1开始找不到'h'

    // pos等于size()
    EXPECT_EQ(String::npos, str.find('h', str.size()));

    // 查找重复字符
    String repeats("aaaaa");
    EXPECT_EQ(0U, repeats.find('a', 0));
    EXPECT_EQ(2U, repeats.find('a', 2));
    EXPECT_EQ(4U, repeats.find('a', 4));
    EXPECT_EQ(String::npos, repeats.find('a', 5));

    // 特殊字符测试
    String special("\t\n\r");
    EXPECT_EQ(0U, special.find('\t', 0));
    EXPECT_EQ(1U, special.find('\n', 0));
    EXPECT_EQ(2U, special.find('\r', 0));
}

// 边界情况测试
TEST(StringFindTest, BoundaryCasesOverloadSet) {
    // 空字符串
    String empty;
    String nonEmpty("test");

    // 空字符串作为被查找对象
    EXPECT_EQ(String::npos, empty.find(nonEmpty, 0));
    EXPECT_EQ(String::npos, empty.find("test", 0));
    EXPECT_EQ(String::npos, empty.find('t', 0));

    // 空字符串作为查找参数
    EXPECT_EQ(0U, nonEmpty.find(String(), 0));
    EXPECT_EQ(0U, nonEmpty.find("", 0));

    // 单字符字符串
    String single("a");
    EXPECT_EQ(0U, single.find(String("a"), 0));
    EXPECT_EQ(0U, single.find("a", 0));
    EXPECT_EQ(0U, single.find('a', 0));
    EXPECT_EQ(String::npos, single.find(String("aa"), 0));
    EXPECT_EQ(String::npos, single.find("aa", 0));
}

// 特殊字符测试
TEST(StringFindTest, SpecialCharactersOverloadSet) {
    // 包含空格、制表符、换行符
    String str("space\ttab\nnewline");
    String spaceStr("space");
    String tabStr("\ttab");

    EXPECT_EQ(0U, str.find(spaceStr, 0));
    EXPECT_EQ(5U, str.find(tabStr, 0));
    EXPECT_EQ(0U, str.find("space", 0));
    EXPECT_EQ(5U, str.find("\ttab", 0));
    EXPECT_EQ(5U, str.find('\t', 0));
    EXPECT_EQ(9U, str.find('\n', 0));

    // 非ASCII字符
    // String unicode("你好，世界！");
    // String hello("你好");
    // String world("世界");
    //
    // EXPECT_EQ(0U, unicode.find(hello, 0));
    // EXPECT_EQ(4U, unicode.find(world, 0));// UTF-8中"你好"占4字节
    // EXPECT_EQ(0U, unicode.find("你好", 0));
    // EXPECT_EQ(4U, unicode.find("世界", 0));
}

// 异常处理测试
TEST(StringFindTest, ExceptionHandlingOverloadSet) {
    String str("test");

    // 测试pos参数越界
    EXPECT_NO_THROW(UNUSED(str.find(String("t"), 5)));
    EXPECT_NO_THROW(UNUSED(str.find("t", 5)));
    EXPECT_NO_THROW(UNUSED(str.find('t', 5)));
}

// COW机制测试
TEST(StringFindTest, CopyOnWriteOverloadSet) {
    String original("test string for COW");
    String copy = original;// 应该共享数据

    // 验证find操作不会触发COW
    EXPECT_EQ(5U, copy.find(String("string"), 0));
    EXPECT_EQ(5U, copy.find("string", 0));
    EXPECT_EQ(7U, copy.find('r', 0));
}

// 本地缓冲区测试
TEST(StringFindTest, LocalBuffer) {
    // 创建一个超过本地缓冲区大小的字符串
    String large(100, 'x');
    String small("x");

    // 在不同位置查找
    EXPECT_EQ(0U, large.find(String("x"), 0));
    EXPECT_EQ(50U, large.find("x", 50));
    EXPECT_EQ(99U, large.find('x', 99));

    // 小字符串测试
    EXPECT_EQ(0U, small.find(String("x"), 0));
    EXPECT_EQ(0U, small.find("x", 0));
    EXPECT_EQ(0U, small.find('x', 0));
}

// 功能一致性测试
TEST(StringFindTest, ConsistencyBetweenOverloads) {
    String str("hello world");
    String substr("world");

    // 验证三个重载版本对相同输入返回相同结果
    EXPECT_EQ(str.find(substr, 0), str.find("world", 0));
    EXPECT_EQ(str.find('o', 0), str.find("o", 0));

    // 从不同位置开始查找
    EXPECT_EQ(str.find(substr, 6), str.find("world", 6));
    EXPECT_EQ(str.find('o', 5), str.find("o", 5));
}

// 组合操作测试
TEST(StringFindTest, CombinedOperationsOverloadSet) {
    String str("abcdefghijklmnopqrstuvwxyz");
    String insertStr("INSERTED");

    // 查找后插入
    auto pos = str.find(String("jkl"), 0);
    str.insert(pos, insertStr);
    EXPECT_EQ(pos, str.find(insertStr, 0));

    // 查找后删除
    pos = str.find("def", 0);
    str.erase(pos, 3);
    EXPECT_EQ(String::npos, str.find(String("def"), 0));

    // 查找字符并替换
    pos = str.find('g', 0);
    str.replace(pos, 1, "XYZ");
    EXPECT_EQ(String::npos, str.find('g', 0));
    EXPECT_EQ(pos, str.find("XYZ", 0));
}

}  // namespace
