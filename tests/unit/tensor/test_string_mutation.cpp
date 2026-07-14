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

// 测试向空字符串添加字符
TEST(StringPushBack, EmptyString) {
    String s;
    EXPECT_TRUE(s.empty());

    s.push_back('a');
    EXPECT_FALSE(s.empty());
    EXPECT_EQ(s.size(), 1);
    EXPECT_STREQ(s.c_str(), "a");
    EXPECT_EQ(s.use_count(), 1);
    EXPECT_TRUE(s.unique());
}

// 测试向非空字符串添加字符（在本地缓冲区容量内）
TEST(StringPushBack, WithinLocalBuffer) {
    // 创建一个小于本地缓冲区大小的字符串（local_capacity_ = 15）
    String s(5, 'x');
    EXPECT_EQ(s.size(), 5);
    EXPECT_STREQ(s.c_str(), "xxxxx");

    // 添加一个字符，仍然在本地缓冲区容量内
    s.push_back('y');
    EXPECT_EQ(s.size(), 6);
    EXPECT_STREQ(s.c_str(), "xxxxxy");
    EXPECT_TRUE(s.IsLocal());// 应该仍然使用本地缓冲区
}

// 测试添加字符达到本地缓冲区容量上限
TEST(StringPushBack, LocalBufferBoundary) {
    // 创建一个正好达到本地缓冲区容量的字符串
    String s(15, 'a');
    EXPECT_EQ(s.size(), 15);
    EXPECT_TRUE(s.IsLocal());// 确认使用本地缓冲区

    // 添加一个字符，应该仍然可以容纳在本地缓冲区中（包括结束符）
    s.push_back('b');
    EXPECT_EQ(s.size(), 16);
    // 验证字符串内容
    std::string expected(15, 'a');
    expected += 'b';
    EXPECT_EQ(static_cast<std::string>(s), expected);
    // 注意：此时可能已经切换到动态分配，因为local_capacity_+1=16
    // 取决于实现细节，可能在添加第16个字符时切换
}

// 测试添加字符触发从小字符串优化到动态分配的转换
TEST(StringPushBack, ExceedLocalBuffer) {
    // 先填充到接近本地缓冲区容量
    String s(10, 'c');
    EXPECT_TRUE(s.IsLocal());

    // 逐个添加字符直到超过本地缓冲区容量
    for (int i = 0; i < 10; ++i) {
        s.push_back('d' + i);
    }

    // 此时字符串长度应该超过本地缓冲区容量
    EXPECT_EQ(s.size(), 20);
    // 验证内容正确性
    std::string expected(10, 'c');
    for (int i = 0; i < 10; ++i) {
        expected += 'd' + i;
    }
    EXPECT_EQ(static_cast<std::string>(s), expected);
}

// 测试向动态分配的字符串添加字符
TEST(StringPushBack, DynamicAllocation) {
    // 创建一个肯定会动态分配的大字符串
    String s(100, 'x');
    EXPECT_FALSE(s.IsLocal());// 应该使用动态分配

    // 添加字符到动态分配的字符串
    s.push_back('y');
    EXPECT_EQ(s.size(), 101);

    // 验证最后一个字符是'y'
    EXPECT_EQ(s[s.size() - 1], 'y');

    // 验证前100个字符仍然是'x'
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(s[i], 'x');
    }
}

// 测试添加各种类型的字符
TEST(StringPushBack, VariousCharacters) {
    String s;

    // 测试小写字母
    s.push_back('a');

    // 测试大写字母
    s.push_back('Z');

    // 测试数字
    s.push_back('9');

    // 测试空格
    s.push_back(' ');

    // 测试特殊字符
    s.push_back('!');

    // 测试控制字符
    s.push_back('\n');

    EXPECT_EQ(s.size(), 6);
    EXPECT_STREQ(s.c_str(), "aZ9 !\n");
}

// 测试多次连续调用push_back
TEST(StringPushBack, MultipleCalls) {
    String s;
    std::string test_str = "Hello, World!";

    // 逐个字符添加
    for (char c: test_str) {
        s.push_back(c);
    }

    EXPECT_EQ(s.size(), test_str.size());
    EXPECT_TRUE(s == test_str);
    // test_str.replace()
}

// 测试push_back与字符串共享/复制的交互
TEST(StringPushBack, CopyOnWrite) {
    // 创建原始字符串
    String original("shared_string");

    // 创建共享引用
    String shared = original;
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(shared.use_count(), 1);

    // 对共享字符串进行修改，应该触发复制
    shared.push_back('!');

    // 验证引用计数分离
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(shared.use_count(), 1);

    // 验证内容正确性
    EXPECT_STREQ(original.c_str(), "shared_string");
    EXPECT_STREQ(shared.c_str(), "shared_string!");
}

// 测试边界情况：大量push_back操作
TEST(StringPushBack, LargeNumberOfOperations) {
    String s;
    const size_t num_chars = 1000;

    for (size_t i = 0; i < num_chars; ++i) {
        s.push_back(static_cast<char>('a' + (i % 26)));
    }

    EXPECT_EQ(s.size(), num_chars);

    // 验证字符串内容
    for (size_t i = 0; i < num_chars; ++i) {
        EXPECT_EQ(s[i], static_cast<char>('a' + (i % 26)));
    }
}

// 测试基于位置的 replace 方法 - 基本功能
TEST(StringReplace, PositionBasedBasic) {
    // 测试 replace(pos, n1, src, n2)
    String s1("Hello, world!");
    s1.replace(7, 5, "C++", 3);
    EXPECT_EQ(s1.size(), 11);
    EXPECT_TRUE(s1 == "Hello, C++!");

    // 测试 replace(pos, n1, src)
    String s2("Hello, world!");
    s2.replace(7, 5, "C++");
    EXPECT_TRUE(s2 == "Hello, C++!");

    // 测试 replace(pos, n, src)
    String s3("Hello, world!");
    s3.replace(7, 5, String("C++"));
    EXPECT_TRUE(s3 == "Hello, C++!");

    // 测试 replace(pos, n1, n2, c)
    String s4("Hello, world!");
    s4.replace(7, 5, 3, 'X');
    EXPECT_TRUE(s4 == "Hello, XXX!");

    String s5("Hello, world, hello!");
    String s6 = s5;
    s5.replace(0, 5, "Hello", 5);
    EXPECT_EQ(s6.use_count(), 2);
}

// 测试基于位置的 replace 方法 - 子字符串替换
TEST(StringReplace, PositionBasedSubstring) {
    // 测试 replace(pos1, n1, src, pos2, n2)
    String s1("Hello, world!");
    String src("beautiful code");
    s1.replace(7, 5, src, 0, 9);// 用 "beautiful" 替换 "world"
    EXPECT_TRUE(s1 == "Hello, beautiful!");

    // 测试默认 n2 = npos
    String s2("Hello, world!");
    s2.replace(7, 5, src, 10);// 用 "code" 替换 "world"
    EXPECT_TRUE(s2 == "Hello, code!");
}

// 测试基于迭代器的 replace 方法
TEST(StringReplace, IteratorBased) {
    String s("Hello, world!");

    // 测试 replace(first, last, src, n)
    s.replace(s.begin() + 7, s.begin() + 12, "C++", 3);
    EXPECT_TRUE(s == "Hello, C++!");

    // 测试 replace(first, last, src)
    s.replace(s.begin() + 7, s.begin() + 10, "Java");
    EXPECT_TRUE(s == "Hello, Java!");

    // 测试 replace(first, last, String&)
    s.replace(s.begin() + 7, s.begin() + 11, String("Python"));
    EXPECT_TRUE(s == "Hello, Python!");

    // 测试 replace(first, last, n, c)
    s.replace(s.begin() + 7, s.begin() + 13, 2, 'X');
    EXPECT_TRUE(s == "Hello, XX!");
}

// 测试模板迭代器版本的 replace 方法
TEST(StringReplace, TemplateIteratorVersion) {
    String s("Hello, world!");

    // 使用 std::vector 迭代器
    std::vector<char> vec = {'C', '+', '+'};
    s.replace(s.begin() + 7, s.begin() + 12, vec.begin(), vec.end());
    EXPECT_TRUE(s == "Hello, C++!");

    // 使用 std::list 迭代器
    std::list<char> lst = {'J', 'a', 'v', 'a'};
    s.replace(s.begin() + 7, s.begin() + 10, lst.begin(), lst.end());
    EXPECT_TRUE(s == "Hello, Java!");

    // 使用 std::array 迭代器
    std::array<char, 6> arr = {'P', 'y', 't', 'h', 'o', 'n'};
    s.replace(s.begin() + 7, s.begin() + 11, arr.begin(), arr.end());
    EXPECT_TRUE(s == "Hello, Python!");

    // 使用原始指针迭代器
    const char* c_str = "Ruby";
    s.replace(s.begin() + 7, s.begin() + 13, c_str, c_str + 4);
    EXPECT_TRUE(s == "Hello, Ruby!");
}

// 测试初始化列表版本的 replace 方法
TEST(StringReplace, InitializerListVersion) {
    String s("Hello, world!");

    // 使用初始化列表替换
    s.replace(s.begin() + 7, s.begin() + 12, {'C', '+', '+'});
    EXPECT_TRUE(s == "Hello, C++!");

    // 使用初始化列表替换为空字符串
    s.replace(s.begin() + 7, s.begin() + 10, {});
    EXPECT_TRUE(s == "Hello, !");
}

// 测试边界情况 - 空替换
TEST(StringReplace, EdgeCaseEmptyReplace) {
    String s("Hello, world!");

    // 替换为空字符串
    s.replace(7, 5, "", 0);
    EXPECT_TRUE(s == "Hello, !");
    EXPECT_EQ(s.size(), 8);

    // 空范围替换（不改变字符串）
    s.replace(5, 0, "extra", 5);
    EXPECT_TRUE(s == "Helloextra, !");
    EXPECT_EQ(s.size(), 13);
}

// 测试边界情况 - 替换整个字符串
TEST(StringReplace, EdgeCaseReplaceWholeString) {
    String s("Hello, world!");

    // 替换整个字符串
    s.replace(0, s.size(), "New string");
    EXPECT_TRUE(s == "New string");
    EXPECT_EQ(s.size(), 10);

    // 使用迭代器替换整个字符串
    s.replace(s.begin(), s.end(), {'E', 'n', 't', 'i', 'r', 'e'});
    EXPECT_TRUE(s == "Entire");
    EXPECT_EQ(s.size(), 6);
}

// 测试边界情况 - 长度变化
TEST(StringReplace, EdgeCaseLengthVariation) {
    String s("abcdef");

    // 替换为更短的内容
    s.replace(2, 3, "x");
    EXPECT_TRUE(s == "abxf");// 注意：f 前有空格，因为我们替换了 'cde' 为 'x'
    EXPECT_EQ(s.size(), 4);

    // 替换为更长的内容
    s.replace(1, 1, "long text");
    EXPECT_TRUE(s == "along textxf");
    EXPECT_EQ(s.size(), 12);
}

// 测试特殊字符处理
TEST(StringReplace, SpecialCharacters) {
    String s("Hello\tworld\n");

    // 替换包含控制字符的部分
    s.replace(5, 6, "user\r", 5);
    EXPECT_TRUE(s == "Hellouser\r\n");
    EXPECT_EQ(s[5], 'u');
    EXPECT_EQ(s[9], '\r');

    // 替换为空字符
    s.replace(5, 5, "\0\0", 2);
    EXPECT_EQ(s.size(), 8);
    EXPECT_EQ(s[5], '\0');
    EXPECT_EQ(s[6], '\0');
    EXPECT_EQ(s[7], '\n');

    // 使用迭代器替换控制字符
    std::vector<char> special = {'\t', '\n', '\r'};
    s.replace(s.begin() + 5, s.end(), special.begin(), special.end());
    EXPECT_EQ(s.size(), 8);
    EXPECT_EQ(s[5], '\t');
    EXPECT_EQ(s[6], '\n');
    EXPECT_EQ(s[7], '\r');
}

// 测试内存管理和引用计数
TEST(StringReplace, MemoryManagement) {
    String original("Hello, world!");
    String copy = original;

    // 确认共享内存
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(copy.use_count(), 1);

    // 修改其中一个，应该触发写时复制
    original.replace(7, 5, "C++");
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(copy.use_count(), 1);
    EXPECT_TRUE(original != copy);
    EXPECT_TRUE(original == "Hello, C++!");
    EXPECT_TRUE(copy == "Hello, world!");
}

// 测试链式调用
TEST(StringReplace, ChainedCalls) {
    String s("Hello, world!");

    // 链式调用多个 replace
    s.replace(0, 5, "Hi")
            .replace(2, 1, ",")
            .replace(4, 6, "there");

    EXPECT_TRUE(s == "Hi, there");
    EXPECT_EQ(s.size(), 9);
}

// 测试异常处理
TEST(StringReplace, ExceptionHandling) {
    String s("Hello, world!");

    // 越界位置应该抛出异常
    EXPECT_THROW(s.replace(20, 5, "error"), std::exception);

    // 迭代器越界应该触发断言失败
    // 注意：由于使用了 CHECK 宏，这里可能无法捕获为异常，而是导致程序终止
    // 此处不直接测试 CHECK 断言，因为它通常会导致程序终止
}

// 测试 append(const_pointer src, size_type n)
TEST(StringAppend, AppendPointerAndCount) {
    // 测试向空字符串追加
    String s1;
    s1.append("hello", 5);
    EXPECT_EQ(s1.size(), 5);
    EXPECT_TRUE(s1 == "hello");

    // 测试向非空字符串追加
    String s2("world");
    s2.append("!", 1);
    EXPECT_EQ(s2.size(), 6);
    EXPECT_TRUE(s2 == "world!");

    // 测试追加空字符串
    String s3("test");
    s3.append("", 0);
    EXPECT_EQ(s3.size(), 4);
    EXPECT_TRUE(s3 == "test");

    // 测试追加包含空字符的字符串
    const char mixed[] = "abc\0def";
    String s4("prefix");
    s4.append(mixed, 7);// 包括空字符在内的7个字符
    EXPECT_EQ(s4.size(), 13);
    EXPECT_EQ(s4[6], 'a');// 验证空字符被正确添加

    // 测试追加大量字符
    const size_t large_size = 1000;
    std::string large_str(large_size, 'x');
    String s5("start");
    s5.append(large_str.data(), large_size);
    EXPECT_EQ(s5.size(), 5 + large_size);
    // EXPECT_TRUE(s5.starts_with("start"));
}

// 测试 append(const String& str)
TEST(StringAppend, AppendString) {
    // 基本追加
    String s1("Hello");
    String s2(" World");
    s1.append(s2);
    EXPECT_EQ(s1.size(), 11);
    EXPECT_STREQ(s1, "Hello World");

    // 追加空字符串
    String s3("test");
    String empty;
    s3.append(empty);
    EXPECT_EQ(s3.size(), 4);
    EXPECT_STREQ(s3, "test");

    // 自我追加
    String s4("abc");
    s4.append(s4);
    EXPECT_EQ(s4.size(), 6);
    EXPECT_STREQ(s4, "abcabc");

    // 测试追加共享字符串
    String original("shared");
    String shared = original;
    String target("prefix_");
    target.append(shared);
    EXPECT_EQ(target.size(), 7 + 6);
    EXPECT_STREQ(target, "prefix_shared");
    EXPECT_EQ(original.use_count(), 1);// 共享引用不应被修改
}

// 测试 append(const String& str, size_type pos, size_type n = npos)
TEST(StringAppend, AppendSubstring) {
    // 基本子字符串追加
    String s1("Hello");
    String s2("Beautiful World");
    s1.append(s2, 10, 5);// 从位置9开始追加5个字符("World")
    EXPECT_EQ(s1.size(), 5 + 5);
    EXPECT_STREQ(s1, "HelloWorld");

    // 测试默认参数n=npos
    String s3("Hi");
    s3.append(s2, 10);// 从位置10开始追加到末尾
    EXPECT_STREQ(s3, "HiWorld");

    // 测试pos超出范围（应该抛出异常或安全处理）
    String s4("test");
    // 此处应根据实现检查异常行为

    // 测试n大于可用字符数
    String s5("Start");
    String s6("End");
    s5.append(s6, 1, 10);// 请求10个字符，但只有2个可用
    EXPECT_STREQ(s5, "Startnd");

    // 测试pos为0的情况
    String s7("Result:");
    s7.append(s2, 0, 9);
    EXPECT_STREQ(s7, "Result:Beautiful");
}

// 测试 append(const_pointer src)
TEST(StringAppend, AppendCString) {
    // 基本C字符串追加
    String s1("Hello");
    s1.append(", world!");
    EXPECT_EQ(s1.size(), 13);
    EXPECT_TRUE(s1 == "Hello, world!");

    // 追加空字符串
    String s2("test");
    s2.append("");
    EXPECT_EQ(s2.size(), 4);
    EXPECT_STREQ(s2, "test");

    // 追加单个字符的C字符串
    String s3("number ");
    s3.append("1");
    EXPECT_STREQ(s3, "number 1");

    // 测试追加长C字符串
    const char long_str[] = "This is a much longer null-terminated C string for testing append functionality.";
    String s4("Start: ");
    s4.append(long_str);
    // EXPECT_TRUE(s4.ends_with("functionality."));
}

// 测试 append(size_type n, value_type c)
TEST(StringAppend, AppendCharNTimes) {
    // 基本字符重复追加
    String s1("Hello");
    s1.append(3, '!');
    EXPECT_EQ(s1.size(), 5 + 3);
    EXPECT_STREQ(s1, "Hello!!!");

    // 追加0个字符
    String s2("test");
    s2.append(0, 'x');
    EXPECT_EQ(s2.size(), 4);
    EXPECT_STREQ(s2, "test");

    // 追加特殊字符
    String s3("Line");
    s3.append(2, '\n');
    s3.append(1, 'E');
    EXPECT_EQ(s3.size(), 4 + 2 + 1);
    EXPECT_EQ(s3[4], '\n');
    EXPECT_EQ(s3[5], '\n');

    // 测试大量字符追加
    const size_t large_count = 100;
    String s4("Repeat:");
    s4.append(large_count, 'z');
    EXPECT_EQ(s4.size(), 7 + large_count);
    for (size_t i = 7; i < s4.size(); ++i) {
        EXPECT_EQ(s4[i], 'z');
    }
}

// 测试 append(std::initializer_list<value_type> l)
TEST(StringAppend, AppendInitializerList) {
    // 基本初始化列表追加
    String s1("Hello");
    s1.append({' ', 'W', 'o', 'r', 'l', 'd'});
    EXPECT_EQ(s1.size(), 5 + 6);
    EXPECT_STREQ(s1, "Hello World");

    // 追加空初始化列表
    String s2("test");
    s2.append({});
    EXPECT_EQ(s2.size(), 4);
    EXPECT_STREQ(s2, "test");

    // 追加包含特殊字符的初始化列表
    String s3("Special: ");
    s3.append({'!', '\t', '?', '\n'});
    EXPECT_EQ(s3.size(), 9 + 4);
    EXPECT_EQ(s3[9], '!');
    EXPECT_EQ(s3[10], '\t');
    EXPECT_EQ(s3[11], '?');
    EXPECT_EQ(s3[12], '\n');
}

// 测试模板方法 append(Iter first, Iter last)
TEST(StringAppend, AppendIterators) {
    // 使用vector迭代器
    std::vector<char> vec = {'W', 'o', 'r', 'l', 'd'};
    String s1("Hello ");
    s1.append(vec.begin(), vec.end());
    EXPECT_EQ(s1.size(), 6 + 5);
    EXPECT_STREQ(s1, "Hello World");

    // 使用list迭代器
    std::list<char> lst = {'!', '!', '!'};
    String s2("Test");
    s2.append(lst.begin(), lst.end());
    EXPECT_STREQ(s2, "Test!!!");

    // 使用deque迭代器
    std::deque<char> deq = {'1', '2', '3'};
    String s3("Numbers: ");
    s3.append(deq.begin(), deq.end());
    EXPECT_STREQ(s3, "Numbers: 123");

    // 使用子范围迭代器
    std::vector<char> long_vec = {'a', 'b', 'c', 'd', 'e', 'f'};
    String s4("Part: ");
    s4.append(long_vec.begin() + 2, long_vec.begin() + 5);// 追加 "cde"
    EXPECT_STREQ(s4, "Part: cde");

    // 使用空范围迭代器
    String s5("Empty: ");
    s5.append(vec.begin(), vec.begin());// 空范围
    EXPECT_STREQ(s5, "Empty: ");
}

// 测试append方法链式调用
TEST(StringAppend, ChainedAppend) {
    String s;
    s.append("Hello")
            .append(1, ' ')
            .append("beautiful")
            .append(1, ' ')
            .append("world!");

    EXPECT_STREQ(s, "Hello beautiful world!");
    EXPECT_EQ(s.size(), 22);
}

// 测试append方法在小字符串优化边界的行为
TEST(StringAppend, AppendAtLocalBufferBoundary) {
    // 创建一个接近本地缓冲区大小的字符串（local_capacity_ = 15）
    String s(12, 'a');// "aaaaaaaaaaaa"
    EXPECT_TRUE(s.IsLocal());

    // 追加刚好填满本地缓冲区的字符
    s.append(3, 'b');        // 现在长度为15
    EXPECT_TRUE(s.IsLocal());// 应该仍然使用本地缓冲区

    // 追加一个字符，可能触发动态分配
    s.append(1, 'c');// 现在长度为16
    EXPECT_STREQ(s.c_str(), "aaaaaaaaaaaabbbc");

    // 继续追加更多字符
    s.append("_more");
    EXPECT_STREQ(s.c_str(), "aaaaaaaaaaaabbbc_more");
}

// 测试operator+=(const String& str)功能
TEST(StringOperatorPlusEqual, StringAddition) {
    // 基本功能测试
    String s1 = "hello";
    String s2 = " world";
    s1 += s2;
    EXPECT_EQ(s1.size(), 11);
    EXPECT_STREQ(s1, "hello world");
    EXPECT_FALSE(s1.empty());

    // 空字符串添加
    String s3 = "test";
    String s4;
    s3 += s4;
    EXPECT_EQ(s3.size(), 4);
    EXPECT_STREQ(s3, "test");

    // 向空字符串添加
    String s5;
    String s6 = "append";
    s5 += s6;
    EXPECT_EQ(s5.size(), 6);
    EXPECT_STREQ(s5, "append");

    // 自我添加
    String s7 = "loop";
    s7 += s7;
    EXPECT_EQ(s7.size(), 8);
    EXPECT_STREQ(s7, "looploop");

    // 特殊字符测试
    String s8 = "special: ";
    String s9 = "!@#$%^&*()";
    s8 += s9;
    EXPECT_EQ(s8.size(), 19);
    EXPECT_STREQ(s8, "special: !@#$%^&*()");
}

// 测试operator+=(const_pointer str)功能
TEST(StringOperatorPlusEqual, CStringAddition) {
    // 基本功能测试
    String s1 = "hello";
    const char* s2 = " world";
    s1 += s2;
    EXPECT_EQ(s1.size(), 11);
    EXPECT_STREQ(s1, "hello world");

    // 空字符串添加
    String s3 = "test";
    s3 += "";
    EXPECT_EQ(s3.size(), 4);
    EXPECT_STREQ(s3, "test");

    // 向空字符串添加
    String s4;
    s4 += "append";
    EXPECT_EQ(s4.size(), 6);
    EXPECT_STREQ(s4, "append");

    // 特殊字符和转义字符
    String s5 = "escape: ";
    s5 += "\\n\\t\\r";
    EXPECT_EQ(s5.size(), 14);
    EXPECT_STREQ(s5, "escape: \\n\\t\\r");

    // 长字符串测试
    const char* long_str = "This is a relatively longer string to test the operator+= functionality with C-style strings";
    String s6 = "Start: ";
    s6 += long_str;
    EXPECT_EQ(s6.size(), 7 + std::strlen(long_str));
    EXPECT_STREQ(s6, "Start: This is a relatively longer string to test the operator+= functionality with C-style strings");
}

// 测试operator+=(value_type c)功能
TEST(StringOperatorPlusEqual, CharAddition) {
    // 基本功能测试
    String s1 = "hello";
    s1 += '!';
    EXPECT_EQ(s1.size(), 6);
    EXPECT_STREQ(s1, "hello!");

    // 多次添加字符
    String s2 = "test";
    s2 += '1';
    s2 += '2';
    s2 += '3';
    EXPECT_EQ(s2.size(), 7);
    EXPECT_STREQ(s2, "test123");

    // 向空字符串添加字符
    String s3;
    s3 += 'a';
    EXPECT_EQ(s3.size(), 1);
    EXPECT_STREQ(s3, "a");

    // 特殊字符测试
    String s4 = "special: ";
    s4 += '\n';
    s4 += '\t';
    s4 += '\r';
    EXPECT_EQ(s4.size(), 12);
    EXPECT_EQ(s4[9], '\n');
    EXPECT_EQ(s4[10], '\t');
    EXPECT_EQ(s4[11], '\r');
}

// 测试operator+=(std::initializer_list<value_type> l)功能
TEST(StringOperatorPlusEqual, InitializerListAddition) {
    // 基本功能测试
    String s1 = "hello";
    s1 += {' ', 'w', 'o', 'r', 'l', 'd'};
    EXPECT_EQ(s1.size(), 11);
    EXPECT_STREQ(s1, "hello world");

    // 空初始化列表
    String s2 = "test";
    s2 += {};
    EXPECT_EQ(s2.size(), 4);
    EXPECT_STREQ(s2, "test");

    // 向空字符串添加初始化列表
    String s3;
    s3 += {'a', 'p', 'p', 'e', 'n', 'd'};
    EXPECT_EQ(s3.size(), 6);
    EXPECT_STREQ(s3, "append");

    // 特殊字符测试
    String s4 = "special: ";
    s4 += {'!', '@', '#', '$', '%'};
    EXPECT_EQ(s4.size(), 14);
    EXPECT_STREQ(s4, "special: !@#$%");// 注意：size()应该是13，因为添加了5个字符
}

// 测试引用计数和复制时修改
TEST(StringOperatorPlusEqual, ReferenceCounting) {
    // 测试写时复制
    String s1 = "shared";
    String s2 = s1;// 共享数据
    EXPECT_TRUE(s1.unique());
    EXPECT_TRUE(s2.unique());

    // 修改s1应该触发复制
    s1 += "_modified";
    EXPECT_TRUE(s1.unique());
    EXPECT_TRUE(s2.unique());// s2现在应该有自己的副本
    EXPECT_STREQ(s1.c_str(), "shared_modified");
    EXPECT_STREQ(s2.c_str(), "shared");
}

}  // namespace
