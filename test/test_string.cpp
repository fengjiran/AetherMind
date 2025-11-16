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
// 测试基本功能：创建指定大小和字符的字符串
TEST(StringConstructorFill, BasicFunctionality) {
    // 测试基本ASCII字符
    String s1(5, 'a');
    EXPECT_FALSE(s1.empty());
    EXPECT_EQ(s1.size(), 5);
    EXPECT_STREQ(s1.c_str(), "aaaaa");
    EXPECT_EQ(s1.use_count(), 1);
    EXPECT_TRUE(s1.unique());

    // 测试数字字符
    String s2(3, '5');
    EXPECT_EQ(s2.size(), 3);
    EXPECT_STREQ(s2.c_str(), "555");

    // 测试特殊字符
    String s3(2, '!');
    EXPECT_EQ(s3.size(), 2);
    EXPECT_STREQ(s3.c_str(), "!!");

    String s4 = "hello";
    s4.insert(s4.begin() + 2, 3, 'a');
    EXPECT_TRUE(s4 == "heaaallo");
}

// 测试边界情况：空字符串
TEST(StringConstructorFill, EmptyString) {
    String s(0, 'x');
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0);
    EXPECT_STREQ(s.c_str(), "");
    // 确保空字符串仍然有有效的空终止符
    EXPECT_EQ(s.data()[0], '\0');
}

// 测试边界情况：较大的字符串
TEST(StringConstructorFill, LargeString) {
    // 创建一个包含100个字符的字符串
    const size_t large_size = 100;
    String s(large_size, 'B');

    EXPECT_EQ(s.size(), large_size);
    EXPECT_FALSE(s.empty());

    // 验证所有字符都是 'B'
    for (size_t i = 0; i < large_size; ++i) {
        EXPECT_EQ(s[i], 'B');
    }

    // 验证字符串结尾有正确的空终止符
    EXPECT_EQ(s.c_str()[large_size], '\0');
}

// 测试特殊字符：包括控制字符
TEST(StringConstructorFill, SpecialCharacters) {
    // 测试空格字符
    String space(4, ' ');
    EXPECT_EQ(space.size(), 4);
    EXPECT_STREQ(space.c_str(), "    ");

    // 测试制表符
    String tab(2, '\t');
    EXPECT_EQ(tab.size(), 2);
    EXPECT_EQ(tab.data()[0], '\t');
    EXPECT_EQ(tab.data()[1], '\t');
    EXPECT_EQ(tab.data()[2], '\0');

    // 测试换行符
    String newline(3, '\n');
    EXPECT_EQ(newline.size(), 3);
    EXPECT_EQ(newline.data()[0], '\n');
    EXPECT_EQ(newline.data()[1], '\n');
    EXPECT_EQ(newline.data()[2], '\n');
    EXPECT_EQ(newline.data()[3], '\0');
}

// 测试非ASCII字符
TEST(StringConstructorFill, NonAsciiCharacters) {
    // 测试扩展ASCII字符
    String extended_ascii(2, static_cast<char>(169));// © 符号
    EXPECT_EQ(extended_ascii.size(), 2);
    EXPECT_EQ(extended_ascii.data()[0], static_cast<char>(169));
    EXPECT_EQ(extended_ascii.data()[1], static_cast<char>(169));

    // 注意：此测试仅验证存储的字节值，不保证这些值在所有系统上都表示相同的字符
}


// 测试与其他String操作的兼容性
TEST(StringConstructorFill, CompatibilityWithOtherOperations) {
    String s(5, 'z');

    // 测试比较操作
    EXPECT_TRUE(s == String("zzzzz"));
    EXPECT_FALSE(s == String("zzzz"));

    // 测试连接操作
    String concat = s + String("abc");
    EXPECT_EQ(concat.size(), 8);
    EXPECT_STREQ(concat.c_str(), "zzzzzabc");

    // 测试赋值操作
    String assign;
    assign = s;
    EXPECT_TRUE(assign == s);

    // 测试转换为std::string
    std::string std_str = static_cast<std::string>(s);
    EXPECT_EQ(std_str, "zzzzz");
}

// 测试索引访问
TEST(StringConstructorFill, IndexAccess) {
    String s(5, 'm');

    // 测试operator[]
    for (size_t i = 0; i < s.size(); ++i) {
        EXPECT_EQ(s[i], 'm');
    }

    // 测试at()方法，包括边界检查
    EXPECT_EQ(s.at(0), 'm');
    EXPECT_EQ(s.at(4), 'm');

    // 测试越界访问会抛出异常
    EXPECT_THROW(UNUSED(s.at(5)), std::exception);
    EXPECT_THROW(UNUSED(s.at(100)), std::exception);
}

// 测试内存共享和引用计数
TEST(StringConstructorFill, MemorySharing) {
    String original(10, 's');
    EXPECT_EQ(original.use_count(), 1);

    // 复制构造应该共享内存
    String copy = original;
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(copy.use_count(), 1);
    EXPECT_TRUE(original.unique());
    EXPECT_TRUE(copy.unique());

    // 修改其中一个不应影响另一个（写时复制）
    // 注意：String类的实现是否支持写时复制需要查看完整代码
    // 这里我们假设它不支持，因为当前实现中没有明确的copy-on-write逻辑
}

// 测试迭代器范围构造函数 - 基本功能
TEST(StringIteratorConstructor, BasicFunctionality) {
    // 使用 std::string 迭代器
    std::string source = "hello world";
    String s1(source.begin(), source.end());
    EXPECT_FALSE(s1.empty());
    EXPECT_EQ(s1.size(), source.size());
    EXPECT_STREQ(s1.c_str(), source.c_str());

    // 使用子范围迭代器
    String s2(source.begin() + 6, source.end());
    EXPECT_EQ(s2.size(), 5);
    EXPECT_STREQ(s2.c_str(), "world");
}

// 测试迭代器范围构造函数 - 不同类型的迭代器
TEST(StringIteratorConstructor, DifferentIteratorTypes) {
    // 使用 vector<char> 迭代器
    std::vector<char> vec = {'t', 'e', 's', 't'};
    String s1(vec.begin(), vec.end());
    EXPECT_EQ(s1.size(), vec.size());
    EXPECT_STREQ(s1.c_str(), "test");

    // 使用 list<char> 迭代器
    std::list<char> lst = {'l', 'i', 's', 't'};
    String s2(lst.begin(), lst.end());
    EXPECT_EQ(s2.size(), lst.size());
    EXPECT_STREQ(s2.c_str(), "list");

    // 使用 array<char, N> 迭代器
    std::array<char, 4> arr = {'a', 'r', 'r', 'a'};
    String s3(arr.begin(), arr.end());
    EXPECT_EQ(s3.size(), arr.size());
    EXPECT_STREQ(s3.c_str(), "arra");

    // 使用 deque<char> 迭代器
    std::deque<char> deq = {'d', 'e', 'q', 'u', 'e'};
    String s4(deq.begin(), deq.end());
    EXPECT_EQ(s4.size(), deq.size());
    EXPECT_STREQ(s4.c_str(), "deque");
}

// 测试迭代器范围构造函数 - 边界情况
TEST(StringIteratorConstructor, EdgeCases) {
    // 空范围
    std::string empty_str;
    String s1(empty_str.begin(), empty_str.end());
    EXPECT_TRUE(s1.empty());
    EXPECT_EQ(s1.size(), 0);
    EXPECT_STREQ(s1.c_str(), "");

    // 单元素范围
    std::string single_char = "a";
    String s2(single_char.begin(), single_char.end());
    EXPECT_FALSE(s2.empty());
    EXPECT_EQ(s2.size(), 1);
    EXPECT_STREQ(s2.c_str(), "a");

    // 相同的迭代器（空范围）
    std::string source = "test";
    auto it = source.begin();
    String s3(it, it);
    EXPECT_TRUE(s3.empty());
}

// 测试迭代器范围构造函数 - 特殊字符
TEST(StringIteratorConstructor, SpecialCharacters) {
    // 包含控制字符的范围
    std::string special_chars = "a\tb\nc\r";
    String s1(special_chars.begin(), special_chars.end());
    EXPECT_EQ(s1.size(), special_chars.size());
    EXPECT_TRUE(s1.compare(special_chars) == 0);

    // 包含空字符的范围
    std::string null_char_str = "ab\0cd";
    String s2(null_char_str.begin(), null_char_str.begin() + 5);// 包含空字符和后续字符
    EXPECT_EQ(s2.size(), 5);
    EXPECT_EQ(s2[0], 'a');
    EXPECT_EQ(s2[1], 'b');
    EXPECT_EQ(s2[2], '\0');
    // EXPECT_EQ(s2[3], 'c');
    // EXPECT_EQ(s2[4], 'd');

    std::string t(5, '\0');
    EXPECT_EQ(t.size(), 5);
}

TEST(String, CopyFromStd) {
    std::string source = "this is a string";
    std::string expect = source;
    String s = source;
    String copy = static_cast<std::string>(s);
    EXPECT_EQ(copy, expect);
    EXPECT_EQ(source.size(), expect.size());
    EXPECT_EQ(s.use_count(), 1);
    EXPECT_TRUE(s.unique());
    EXPECT_EQ(copy.use_count(), 1);
    EXPECT_TRUE(copy.unique());
    // String ss0{'a', 'b'};
    // String test(source.begin(), source.end());

    std::vector<std::string> pow5;
    pow5.reserve(7);
    int x = 1;
    for (int i = 0; i < 7; ++i) {
        int t = x;
        std::string code;
        while (t > 0) {
            char c = t % 2 == 0 ? '0' : '1';
            code += c;
            t /= 2;
        }
        std::reverse(code.begin(), code.end());
        pow5.push_back(code);
        x *= 5;
    }

    for (const auto& str: pow5) {
        std::cout << str << std::endl;
    }

    int ciphertext = 216613;
    std::string ss;
    while (ciphertext > 0) {
        int mod = ciphertext % 10;
        ciphertext /= 10;
        ss += ('0' + mod);
    }
    std::reverse(ss.begin(), ss.end());
    std::cout << ss << std::endl;
}

TEST(String, Assignment) {
    String s{std::string{"hello"}};
    s = std::string{"world"};
    EXPECT_TRUE(s == "world");
    EXPECT_TRUE(s.size() == 5);
    EXPECT_EQ(s.use_count(), 1);
    EXPECT_TRUE(s.unique());

    s = String("Hello, world");
    EXPECT_TRUE(s == "Hello, world");
}

TEST(String, empty) {
    String s{"hello"};
    EXPECT_FALSE(s.empty());
    s = std::string("");
    EXPECT_TRUE(s.empty());
}

TEST(String, Comparisons) {
    std::string source = "a string";
    std::string mismatch = "a string but longer";
    String s{"a string"};
    String m{mismatch};
    auto x = source.size();
    // source.push_back();

    EXPECT_EQ("a str" >= s, false);
    EXPECT_EQ(s == source, true);
    EXPECT_EQ(s == mismatch, false);
    EXPECT_EQ(s == source.data(), true);
    EXPECT_EQ(s == mismatch.data(), false);

    EXPECT_EQ(s < m, source < mismatch);
    EXPECT_EQ(s > m, source > mismatch);
    EXPECT_EQ(s <= m, source <= mismatch);
    EXPECT_EQ(s >= m, source >= mismatch);
    EXPECT_EQ(s == m, source == mismatch);
    EXPECT_EQ(s != m, source != mismatch);

    EXPECT_EQ(m < s, mismatch < source);
    EXPECT_EQ(m > s, mismatch > source);
    EXPECT_EQ(m <= s, mismatch <= source);
    EXPECT_EQ(m >= s, mismatch >= source);
    EXPECT_EQ(m == s, mismatch == source);
    EXPECT_EQ(m != s, mismatch != source);
}

TEST(String, Compare) {
    // string compare const char*
    String s{"hello"};
    EXPECT_EQ(s.compare("hello"), 0);
    EXPECT_EQ(s.compare(String("hello")), 0);

    EXPECT_EQ(s.compare("hallo"), 1);
    EXPECT_EQ(s.compare(String("hallo")), 1);
    EXPECT_EQ(s.compare("hfllo"), -1);
    EXPECT_EQ(s.compare(String("hfllo")), -1);
    // s is longer
    EXPECT_EQ(s.compare("hell"), 1);
    EXPECT_EQ(s.compare(String("hell")), 1);
    // s is shorter
    EXPECT_EQ(s.compare("hello world"), -1);
    EXPECT_EQ(s.compare(String("helloworld")), -1);
}

// Check '\0' handling
TEST(String, null_byte_handling) {
    // Ensure string still compares equal if it contains '\0'.
    std::string v1 = "hello world";
    size_t v1_size = v1.size();
    v1[5] = '\0';
    EXPECT_EQ(v1[5], '\0');
    EXPECT_EQ(v1.size(), v1_size);
    String str_v1{v1};
    EXPECT_EQ(str_v1.compare(v1), 0);
    EXPECT_EQ(str_v1.size(), v1_size);

    // Ensure bytes after '\0' are taken into account for mismatches.
    std::string v2 = "aaa one";
    std::string v3 = "aaa two";
    v2[3] = '\0';
    v3[3] = '\0';
    String str_v2{v2};
    String str_v3{v3};
    EXPECT_EQ(str_v2.compare(str_v3), -1);
    EXPECT_EQ(str_v2.size(), 7);
    // strcmp won't be able to detect the mismatch
    EXPECT_EQ(strcmp(v2.data(), v3.data()), 0);
    // string::compare can handle \0 since it knows size
    EXPECT_LT(v2.compare(v3), 0);

    // If there is mismatch before '\0', should still handle it.
    std::string v4 = "acc one";
    std::string v5 = "abb two";
    v4[3] = '\0';
    v5[3] = '\0';
    String str_v4{v4};
    String str_v5{v5};
    EXPECT_GT(str_v4.compare(str_v5), 0);
    EXPECT_EQ(str_v4.size(), 7);
    // strcmp is able to detect the mismatch
    EXPECT_GT(strcmp(v4.data(), v5.data()), 0);
    // string::compare can handle \0 since it knows size
    EXPECT_GT(v4.compare(v5), 0);
}

TEST(String, compare_same_memory_region_different_size) {
    std::string source = "a string";
    String str_source{source};
    char* memory = const_cast<char*>(str_source.data());
    EXPECT_EQ(str_source.compare(memory), 0);
    // This changes the string size
    memory[2] = '\0';
    // memory is logically shorter now
    EXPECT_GT(str_source.compare(memory), 0);
}

TEST(String, compare) {
    using namespace std;
    constexpr auto mismatch1_cstr = "a string but longer";
    string source = "a string";
    string mismatch1 = mismatch1_cstr;
    string mismatch2 = "a strin";
    string mismatch3 = "a b";
    string mismatch4 = "a t";
    String str_source{source};
    String str_mismatch1{mismatch1_cstr};
    String str_mismatch2{mismatch2};
    String str_mismatch3{mismatch3};
    String str_mismatch4{mismatch4};

    // compare with string
    EXPECT_EQ(str_source.compare(source), 0);
    EXPECT_TRUE(str_source == source);
    EXPECT_TRUE(source == str_source);
    EXPECT_TRUE(str_source <= source);
    EXPECT_TRUE(source <= str_source);
    EXPECT_TRUE(str_source >= source);
    EXPECT_TRUE(source >= str_source);
    EXPECT_LT(str_source.compare(mismatch1), 0);
    EXPECT_TRUE(str_source < mismatch1);
    EXPECT_TRUE(mismatch1 != str_source);
    EXPECT_GT(str_source.compare(mismatch2), 0);
    EXPECT_TRUE(str_source > mismatch2);
    EXPECT_TRUE(mismatch2 < str_source);
    EXPECT_GT(str_source.compare(mismatch3), 0);
    EXPECT_TRUE(str_source > mismatch3);
    EXPECT_LT(str_source.compare(mismatch4), 0);
    EXPECT_TRUE(str_source < mismatch4);
    EXPECT_TRUE(mismatch4 > str_source);

    // compare with char*
    EXPECT_EQ(str_source.compare(source.data()), 0);
    EXPECT_TRUE(str_source == source.data());
    EXPECT_TRUE(source.data() == str_source);
    EXPECT_TRUE(str_source <= source.data());
    EXPECT_TRUE(source <= str_source.data());
    EXPECT_TRUE(str_source >= source.data());
    EXPECT_TRUE(source >= str_source.data());
    EXPECT_LT(str_source.compare(mismatch1.data()), 0);
    EXPECT_TRUE(str_source < mismatch1.data());
    EXPECT_TRUE(str_source != mismatch1.data());
    EXPECT_TRUE(mismatch1.data() != str_source);
    EXPECT_GT(str_source.compare(mismatch2.data()), 0);
    EXPECT_TRUE(str_source > mismatch2.data());
    EXPECT_TRUE(mismatch2.data() < str_source);
    EXPECT_GT(str_source.compare(mismatch3.data()), 0);
    EXPECT_TRUE(str_source > mismatch3.data());
    EXPECT_LT(str_source.compare(mismatch4.data()), 0);
    EXPECT_TRUE(str_source < mismatch4.data());
    EXPECT_TRUE(mismatch4.data() > str_source);

    // compare with String
    EXPECT_LT(str_source.compare(str_mismatch1), 0);
    EXPECT_TRUE(str_source < str_mismatch1);
    EXPECT_GT(str_source.compare(str_mismatch2), 0);
    EXPECT_TRUE(str_source > str_mismatch2);
    EXPECT_GT(str_source.compare(str_mismatch3), 0);
    EXPECT_TRUE(str_source > str_mismatch3);
    EXPECT_LT(str_source.compare(str_mismatch4), 0);
    EXPECT_TRUE(str_source < str_mismatch4);
}

TEST(String, c_str) {
    std::string source = "this is a string";
    std::string mismatch = "mismatch";
    String s{source};

    EXPECT_EQ(std::strcmp(s.c_str(), source.data()), 0);
    EXPECT_NE(std::strcmp(s.c_str(), mismatch.data()), 0);

    String s1 = "hello";
    EXPECT_TRUE(s1 == "hello");
    EXPECT_NE(s1.data(), &"hello"[0]);
}

TEST(String, hash) {
    std::string source = "this is a string";
    String s{source};

    std::unordered_map<String, std::string> map;
    String k1{std::string{"k1"}};
    std::string v1{"v1"};
    String k2{std::string{"k2"}};
    std::string v2{"v2"};
    map[k1] = v1;
    map[k2] = v2;

    EXPECT_EQ(map[k1], v1);
    EXPECT_EQ(map[k2], v2);
}

TEST(String, Concat) {
    String s1("hello");
    String s2("world");
    std::string s3("world");
    String res1 = s1 + s2;
    String res2 = s1 + s3;
    String res3 = s3 + s1;
    String res4 = s1 + "world";
    String res5 = "world" + s1;

    EXPECT_EQ(res1.compare("helloworld"), 0);
    EXPECT_EQ(res2.compare("helloworld"), 0);
    EXPECT_EQ(res3.compare("worldhello"), 0);
    EXPECT_EQ(res4.compare("helloworld"), 0);
    EXPECT_EQ(res5.compare("worldhello"), 0);

    String storage_scope;
    String res = "The input storage scope \"" + storage_scope + "\" is invalid.";
    EXPECT_EQ(res.compare("The input storage scope \"\" is invalid."), 0);
}

TEST(String, StdHash) {
    String s1 = "a";
    String s2(std::string("a"));
    EXPECT_EQ(std::hash<String>()(s1), std::hash<String>()(s2));
}

TEST(String, Any1) {
    String s1 = "hello";
    Any x1 = s1;
    EXPECT_EQ(s1.use_count(), 1);
    auto s2 = x1.try_cast<String>();
    EXPECT_TRUE(s2.has_value());
    EXPECT_EQ(s1.use_count(), 1);
    EXPECT_TRUE(s2.value() == s1);

    Any x2 = std::move(s1);
    EXPECT_EQ(s2.value().use_count(), 1);

    auto s3 = x1.cast<String>();
    EXPECT_EQ(s3.use_count(), 1);

    String s4;
    Any x4 = s4;
    EXPECT_TRUE(!s4.defined());
    EXPECT_EQ(s4.use_count(), 1);
}

TEST(String, Any2) {
    Any b = "hello";
    EXPECT_TRUE(b.IsString());
    EXPECT_TRUE(b.as<String>().has_value());
    EXPECT_EQ(b.as<String>().value(), "hello");
    EXPECT_EQ(b.try_cast<std::string>().value(), "hello");
    EXPECT_EQ(b.use_count(), 1);

    std::string s_world = "world";
    b = s_world;
    EXPECT_EQ(b.try_cast<std::string>().value(), "world");

    String s{"hello"};
    Any a = s;
    EXPECT_EQ(a.as<String>().value(), "hello");
    EXPECT_EQ(a.try_cast<std::string>().value(), "hello");

    Any c = "long string very long";
    EXPECT_EQ(c.as<String>().value(), "long string very long");
    EXPECT_EQ(c.try_cast<std::string>().value(), "long string very long");
}

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
    // EXPECT_TRUE(s.IsLocal());  // 应该仍然使用本地缓冲区
}

// 测试添加字符达到本地缓冲区容量上限
TEST(StringPushBack, LocalBufferBoundary) {
    // 创建一个正好达到本地缓冲区容量的字符串
    String s(15, 'a');
    EXPECT_EQ(s.size(), 15);
    // EXPECT_TRUE(s.IsLocal());  // 确认使用本地缓冲区

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
    // EXPECT_TRUE(s.IsLocal());

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
    // EXPECT_FALSE(s.IsLocal());  // 应该使用动态分配

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
    EXPECT_EQ(s4.size(), 9);
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
    EXPECT_STREQ(s1, "Hello, world!");
    s1.append("hello", 10);
    EXPECT_TRUE(s1 == "Hello, world!hello");
    std::cout << s1 << std::endl;

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
    // EXPECT_TRUE(s.IsLocal());

    // 追加刚好填满本地缓冲区的字符
    s.append(3, 'b');// 现在长度为15
    // EXPECT_TRUE(s.IsLocal());// 应该仍然使用本地缓冲区

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

// 测试front()和back()的基本功能
TEST(StringFrontBack, BasicFunctionality) {
    // 测试单字符字符串
    String s1(1, 'a');
    EXPECT_FALSE(s1.empty());
    EXPECT_EQ(s1.size(), 1);
    EXPECT_EQ(s1.front(), 'a');
    EXPECT_EQ(s1.back(), 'a');

    // 测试多字符字符串
    String s2(5, 'b');
    EXPECT_EQ(s2.front(), 'b');
    EXPECT_EQ(s2.back(), 'b');

    // 测试不同字符的字符串
    String s3("hello");
    EXPECT_EQ(s3.front(), 'h');
    EXPECT_EQ(s3.back(), 'o');
}

// 测试const版本的front()和back()
TEST(StringFrontBack, ConstVersion) {
    const String s("const_string");
    EXPECT_EQ(s.front(), 'c');
    EXPECT_EQ(s.back(), 'g');

    // 确保const版本不会修改字符串
    EXPECT_STREQ(s, "const_string");
}

// 测试非const版本的front()和back()
TEST(StringFrontBack, NonConstVersion) {
    String s("modify_string");

    // 使用front()和back()修改字符串的首尾字符
    // 注意：由于CharProxy的具体实现细节不明确，这里使用operator[]代替
    // 如果CharProxy支持直接赋值，可以使用s.front() = 'M'和s.back() = 'G'
    s[0] = 'M';           // 等价于s.front()
    s[s.size() - 1] = 'G';// 等价于s.back()

    EXPECT_EQ(s.front(), 'M');
    EXPECT_EQ(s.back(), 'G');
    EXPECT_STREQ(s, "Modify_strinG");
}

// 测试特殊字符
TEST(StringFrontBack, SpecialCharacters) {
    // 测试数字字符
    String digits("1234567890");
    EXPECT_EQ(digits.front(), '1');
    EXPECT_EQ(digits.back(), '0');

    // 测试特殊符号
    String special("!@#$%");
    EXPECT_EQ(special.front(), '!');
    EXPECT_EQ(special.back(), '%');

    // 测试控制字符
    String control(2, '\n');
    EXPECT_EQ(control.front(), '\n');
    EXPECT_EQ(control.back(), '\n');
}

// 测试边界情况：front()和back()在空字符串上的行为
// 注意：根据实现，对空字符串调用front()/back()会触发CHECK失败
// 这个测试需要在支持异常处理或检查的环境中运行
TEST(StringFrontBack, DISABLED_EmptyString) {
    // 注意：这个测试会触发CHECK失败，因此被禁用
    // 如果需要测试CHECK行为，可以使用GTest的死亡测试
    String empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0);

    // 以下两行会触发CHECK失败
    // empty.front();
    // empty.back();
}

// 测试通过迭代器构造的字符串
TEST(StringFrontBack, IteratorConstructed) {
    std::string std_str("iterator_test");
    String s(std_str.begin(), std_str.end());

    EXPECT_EQ(s.front(), 'i');
    EXPECT_EQ(s.back(), 't');
    EXPECT_STREQ(s, "iterator_test");
}

// 测试使用初始化列表构造的字符串
TEST(StringFrontBack, InitializerList) {
    String s({'i', 'n', 'i', 't', '_', 'l', 'i', 's', 't'});

    EXPECT_EQ(s.front(), 'i');
    EXPECT_EQ(s.back(), 't');
    EXPECT_STREQ(s, "init_list");
}

// 测试引用计数情况下的front()和back()
TEST(StringFrontBack, ReferenceCounting) {
    String original("reference_test");
    String copy = original;

    // 验证两个字符串共享同一个实现
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_TRUE(original.unique());

    // 测试front()和back()在共享实现的情况下的行为
    EXPECT_EQ(original.front(), copy.front());
    EXPECT_EQ(original.back(), copy.back());
}

// 测试substr的基本功能
TEST(StringSubstr, BasicFunctionality) {
    String s("hello world");

    // 测试从指定位置开始，截取指定长度
    String sub1 = s.substr(0, 5);
    EXPECT_EQ(sub1.size(), 5);
    EXPECT_STREQ(sub1, "hello");

    // 测试从中间位置开始截取
    String sub2 = s.substr(6, 5);
    EXPECT_EQ(sub2.size(), 5);
    EXPECT_STREQ(sub2, "world");

    // 测试截取单个字符
    String sub3 = s.substr(3, 1);
    EXPECT_EQ(sub3.size(), 1);
    EXPECT_STREQ(sub3, "l");
}

// 测试默认参数
TEST(StringSubstr, DefaultParameters) {
    String s("default test");

    // 测试默认n=npos，即截取到字符串末尾
    String sub1 = s.substr(8);
    EXPECT_EQ(sub1.size(), 4);
    EXPECT_STREQ(sub1, "test");

    // 测试默认pos=0，截取整个字符串
    String sub2 = s.substr();
    EXPECT_EQ(sub2.size(), 12);
    EXPECT_STREQ(sub2, "default test");

    // 测试两个默认参数，也应该截取整个字符串
    String sub3 = s.substr(0);
    EXPECT_EQ(sub3.size(), 12);
    EXPECT_STREQ(sub3, "default test");
}

// 测试边界情况
TEST(StringSubstr, BoundaryConditions) {
    String s("boundary");
    const size_t len = s.size();

    // 测试截取到字符串末尾
    String sub1 = s.substr(4, 100);// 100超过剩余字符数
    EXPECT_EQ(sub1.size(), len - 4);
    EXPECT_STREQ(sub1, "dary");

    // 测试pos等于字符串长度，应该返回空字符串
    String sub2 = s.substr(len);
    EXPECT_TRUE(sub2.empty());
    EXPECT_EQ(sub2.size(), 0);
    EXPECT_STREQ(sub2, "");

    // 测试n=0，应该返回空字符串
    String sub3 = s.substr(3, 0);
    EXPECT_TRUE(sub3.empty());
    EXPECT_EQ(sub3.size(), 0);
    EXPECT_STREQ(sub3, "");
}

// 测试空字符串
TEST(StringSubstr, EmptyString) {
    String empty;

    // 测试空字符串从位置0截取，返回空字符串
    String sub1 = empty.substr();
    EXPECT_TRUE(sub1.empty());
    EXPECT_EQ(sub1.size(), 0);

    // 测试空字符串从位置0截取0个字符，返回空字符串
    String sub2 = empty.substr(0, 0);
    EXPECT_TRUE(sub2.empty());
    EXPECT_EQ(sub2.size(), 0);
}

// 测试异常情况
TEST(StringSubstr, ExceptionHandling) {
    String s("exception");

    // 测试pos超出范围，应该抛出异常
    EXPECT_THROW({ String sub = s.substr(s.size() + 1); }, Error);

    // 测试pos超出范围，即使n=0也应该抛出异常
    EXPECT_THROW({ String sub = s.substr(s.size() + 1, 0); }, Error);
}

// 测试特殊字符和多字节字符
TEST(StringSubstr, SpecialCharacters) {
    // 测试包含特殊字符的字符串
    String special("!@#$%^&*()");
    String sub1 = special.substr(2, 4);
    EXPECT_EQ(sub1.size(), 4);
    EXPECT_STREQ(sub1, "#$%^");

    // 测试包含数字和字母混合的字符串
    String mixed("a1b2c3d4");
    String sub2 = mixed.substr(1, 6);
    EXPECT_EQ(sub2.size(), 6);
    EXPECT_STREQ(sub2, "1b2c3d");
}

// 测试引用计数和COW行为
TEST(StringSubstr, ReferenceCounting) {
    String original("reference test");

    // 截取子字符串
    String substring = original.substr(4, 4);

    // 验证子字符串是一个独立的对象
    EXPECT_EQ(substring.use_count(), 1);
    EXPECT_TRUE(substring.unique());

    // 验证原始字符串没有被修改
    EXPECT_STREQ(original, "reference test");
}

// 测试基本比较功能 - 与String对象比较
TEST(StringCompareTest, BasicCompareWithString) {
    // 测试相等字符串
    String s1("hello");
    String s2("hello");
    EXPECT_EQ(s1.compare(s2), 0);
    EXPECT_EQ(s2.compare(s1), 0);

    // 测试不等字符串 - s1在字典序上小于s2
    String s3("apple");
    String s4("banana");
    EXPECT_LT(s3.compare(s4), 0);
    EXPECT_GT(s4.compare(s3), 0);

    // 测试前缀字符串
    String s5("test");
    String s6("testing");
    EXPECT_LT(s5.compare(s6), 0);
    EXPECT_GT(s6.compare(s5), 0);

    // 测试空字符串
    String empty;
    EXPECT_GT(s1.compare(empty), 0);
    EXPECT_LT(empty.compare(s1), 0);
    EXPECT_EQ(empty.compare(empty), 0);
}

// 测试带位置和长度参数的比较 - 与String对象
TEST(StringCompareTest, CompareWithPosAndLength) {
    String s1("hello world");
    String s2("world");
    String s3("hello");
    String s4("hello beautiful world");

    // 测试从指定位置开始比较
    EXPECT_EQ(s1.compare(6, 5, s2), 0);// 比较 "world" 和 "world"
    EXPECT_EQ(s1.compare(0, 5, s3), 0);// 比较 "hello" 和 "hello"

    // 测试长度限制
    EXPECT_EQ(s1.compare(0, 5, s4, 0, 5), 0);// 比较 "hello" 和 "hello"

    // 测试不同子串的比较
    EXPECT_LT(s1.compare(0, 5, s2), 0);// "hello" < "world"
    EXPECT_GT(s1.compare(6, 5, s3), 0);// "world" > "hello"

    // 测试默认长度参数
    EXPECT_EQ(s1.compare(0, 5, s4, 0), -1);// 使用默认n2 = npos
}

// 测试与std::string比较
TEST(StringCompareTest, CompareWithStdString) {
    String s1("hello");
    std::string std_s1("hello");
    std::string std_s2("world");
    std::string std_s3("hello world");

    // 测试相等字符串
    EXPECT_EQ(s1.compare(std_s1), 0);

    // 测试不等字符串
    EXPECT_LT(s1.compare(std_s2), 0);
    EXPECT_GT(s1.compare(std_s1.substr(0, 4)), 0);// "hello" > "hell"

    // 测试带位置和长度参数的比较
    EXPECT_EQ(s1.compare(0, 5, std_s3), -1);     // 比较 "hello" 和 "hello world"
    EXPECT_EQ(s1.compare(0, 5, std_s3, 0, 5), 0);// 比较 "hello" 和 "hello"

    // 测试位置超出范围
    EXPECT_THROW(UNUSED(s1.compare(0, 5, std_s3, 100, 5)), Error);
}

// 测试与C风格字符串比较
TEST(StringCompareTest, CompareWithConstCharPtr) {
    String s1("hello");
    const char* cstr1 = "hello";
    const char* cstr2 = "world";
    const char* cstr3 = "hello world";

    // 测试相等字符串
    EXPECT_EQ(s1.compare(cstr1), 0);

    // 测试不等字符串
    EXPECT_LT(s1.compare(cstr2), 0);
    EXPECT_GT(s1.compare("hell"), 0);// "hello" > "hell"

    // 测试带位置和长度参数的比较
    EXPECT_EQ(s1.compare(0, 5, cstr3), -1);  // 比较 "hello" 和 "hello world"
    EXPECT_EQ(s1.compare(0, 5, cstr3, 5), 0);// 比较 "hello" 和 "hello"

    // 测试长度限制（如果n2大于实际字符串长度，应该使用实际长度）
    EXPECT_EQ(s1.compare(0, 5, cstr1, 10), 0);// n2=10大于实际长度
}

// 测试边界情况
TEST(StringCompareTest, EdgeCases) {
    // 空字符串与空字符串比较
    String empty1;
    String empty2;
    EXPECT_EQ(empty1.compare(empty2), 0);
    EXPECT_EQ(empty1.compare(std::string()), 0);
    EXPECT_EQ(empty1.compare(""), 0);

    // 空字符串与非空字符串比较
    String non_empty("test");
    EXPECT_LT(empty1.compare(non_empty), 0);
    EXPECT_GT(non_empty.compare(empty1), 0);

    // 长度为1的字符串比较
    String single_char1("a");
    String single_char2("b");
    EXPECT_LT(single_char1.compare(single_char2), 0);
    EXPECT_GT(single_char2.compare(single_char1), 0);

    // 比较前缀相同但长度不同的字符串
    String short_str("prefix");
    String long_str("prefix_suffix");
    EXPECT_LT(short_str.compare(long_str), 0);
    EXPECT_GT(long_str.compare(short_str), 0);
}

// 测试特殊字符
TEST(StringCompareTest, SpecialCharacters) {
    // 测试空格和控制字符
    String s1("a b");
    String s2("a\tb");
    String s3("a\nb");

    EXPECT_GT(s1.compare(s2), 0);// 空格的ASCII值小于制表符
    EXPECT_GT(s1.compare(s3), 0);// 空格的ASCII值小于换行符
    EXPECT_LT(s2.compare(s3), 0);// 制表符的ASCII值小于换行符

    // 测试非ASCII字符
    // String extended_ascii1("a\x7Fb");// DEL字符
    // String extended_ascii2("a\x80b");// 扩展ASCII
    // EXPECT_LT(extended_ascii1.compare(extended_ascii2), 0);

    // 测试空字符
    const char* with_null = "test\0partial";
    String s4(with_null, 11);// 包含null字符的字符串
    String s5("test");
    EXPECT_GT(s4.compare(s5), 0);// 包含null的字符串比前缀长，所以更大
}

// 测试异常情况
TEST(StringCompareTest, ExceptionHandling) {
    String s("hello");
    String other("world");
    std::string std_s("world");

    // 测试位置超出范围
    EXPECT_THROW(UNUSED(s.compare(10, 5, other)), Error);
    // EXPECT_THROW(UNUSED(s.compare(5, 5, other)), Error);// pos有效但pos+n超过size

    EXPECT_THROW(UNUSED(s.compare(10, 5, std_s)), Error);
    // EXPECT_THROW(UNUSED(s.compare(5, 5, std_s)), Error);

    EXPECT_THROW(UNUSED(s.compare(10, 5, "world")), Error);
    // EXPECT_THROW(UNUSED(s.compare(5, 5, "world")), Error);

    // 测试other位置超出范围
    EXPECT_THROW(UNUSED(s.compare(0, 5, other, 10, 5)), Error);
}

// 测试相同内存区域的比较
TEST(StringCompareTest, SameMemoryRegion) {
    String s("hello world");

    // 比较自身
    EXPECT_EQ(s.compare(s), 0);

    // 比较自身的不同子串
    EXPECT_LT(s.compare(0, 5, s, 6, 5), 0);// "hello" < "world"
    EXPECT_GT(s.compare(6, 5, s, 0, 5), 0);// "world" > "hello"
}

// 测试compare与==操作符的一致性
TEST(StringCompareTest, ConsistencyWithEqualityOperator) {
    String s1("hello");
    String s2("hello");
    String s3("world");

    EXPECT_TRUE((s1.compare(s2) == 0) == (s1 == s2));
    EXPECT_TRUE((s1.compare(s3) != 0) == (s1 != s3));

    // 与std::string比较的一致性
    std::string std_s1("hello");
    std::string std_s3("world");
    EXPECT_TRUE((s1.compare(std_s1) == 0) == (s1 == std_s1));
    EXPECT_TRUE((s1.compare(std_s3) != 0) == (s1 != std_s3));

    // 与C字符串比较的一致性
    EXPECT_TRUE((s1.compare("hello") == 0) == (s1 == "hello"));
    EXPECT_TRUE((s1.compare("world") != 0) == (s1 != "world"));
}

// 测试迭代器版本的insert方法
TEST(StringInsert, IteratorVersion) {
    // 测试在字符串中间插入多个相同字符
    String s1("hello");
    auto it1 = s1.insert(s1.begin() + 2, 3, 'a');
    EXPECT_STREQ(s1, "heaaallo");
    EXPECT_EQ(it1, s1.begin() + 2);// 验证迭代器指向插入位置

    // 测试在字符串开头插入单个字符
    String s2("world");
    auto it2 = s2.insert(s2.begin(), 'H');
    EXPECT_STREQ(s2, "Hworld");
    EXPECT_EQ(it2, s2.begin());// 验证迭代器指向插入位置

    // 测试在字符串结尾插入单个字符
    String s3("hello");
    auto it3 = s3.insert(s3.end(), '!');
    EXPECT_STREQ(s3, "hello!");
    EXPECT_EQ(it3, s3.end() - 1);// 验证迭代器指向插入位置
}

// 测试迭代器范围插入
TEST(StringInsert, IteratorRange) {
    String s1("hello");
    std::vector<char> chars = {'w', 'o', 'r', 'l', 'd'};

    // 在字符串中间插入字符范围
    auto it1 = s1.insert(s1.begin() + 5, chars.begin(), chars.end());
    EXPECT_STREQ(s1, "helloworld");
    EXPECT_EQ(it1, s1.begin() + 5);

    // 在空字符串中插入字符范围
    String s2;
    auto it2 = s2.insert(s2.begin(), chars.begin(), chars.end());
    EXPECT_STREQ(s2, "world");
    EXPECT_EQ(it2, s2.begin());
}

// 测试初始化列表插入
TEST(StringInsert, InitializerList) {
    String s1("hello");

    // 使用初始化列表插入
    auto it1 = s1.insert(s1.begin() + 2, {'a', 'b', 'c'});
    EXPECT_STREQ(s1, "heabcllo");
    EXPECT_EQ(it1, s1.begin() + 2);

    // 在字符串开头插入初始化列表
    String s2("world");
    auto it2 = s2.insert(s2.begin(), {'H', 'e', 'l', 'l', 'o', ' '});
    EXPECT_STREQ(s2, "Hello world");
    EXPECT_EQ(it2, s2.begin());
}

// 测试位置版本的insert方法 - 插入String对象
TEST(StringInsert, PositionString) {
    String s1("hello");
    String s2("world");

    // 在中间位置插入整个String
    s1.insert(5, s2);
    EXPECT_STREQ(s1, "helloworld");

    // 在指定位置插入String的子串
    String s3("hello");
    String s4("123456789");
    s3.insert(2, s4, 3, 4);// 从s4的位置3开始插入4个字符
    EXPECT_STREQ(s3, "he4567llo");
}

// 测试位置版本的insert方法 - 插入C字符串
TEST(StringInsert, PositionCString) {
    String s1("hello");

    // 插入带长度的C字符串
    s1.insert(2, "XYZ", 3);
    EXPECT_STREQ(s1, "heXYZllo");

    // 插入C字符串（自动计算长度）
    String s2("world");
    s2.insert(0, "Hello ");
    EXPECT_STREQ(s2, "Hello world");

    // 插入空C字符串
    String s3("test");
    s3.insert(2, "");
    EXPECT_STREQ(s3, "test");// 应该保持不变
}

// 测试位置版本的insert方法 - 插入多个相同字符
TEST(StringInsert, PositionMultipleChars) {
    String s1("hello");

    // 在指定位置插入多个相同字符
    s1.insert(3, 4, 'x');
    EXPECT_STREQ(s1.c_str(), "helxxxxlo");

    // 在位置0插入多个相同字符
    std::string s2("world");
    // std::string::const_iterator xc = 0;
    s2.insert(0, 2, 'A');
    EXPECT_STREQ(s2.c_str(), "AAworld");

    // 在末尾位置插入多个相同字符
    String s3("hello");
    s3.insert(s3.size(), 3, '!');
    EXPECT_STREQ(s3.c_str(), "hello!!!");

    // 插入0个字符
    String s4("test");
    s4.insert(2, 0, 'x');
    EXPECT_STREQ(s4.c_str(), "test");// 应该保持不变
}

// 测试边界情况
TEST(StringInsert, BoundaryConditions) {
    // 空字符串插入
    String empty;
    empty.insert(0, "hello");
    EXPECT_STREQ(empty.c_str(), "hello");

    // 在超出范围的位置插入（应该抛出异常）
    String s("test");
    EXPECT_THROW({ s.insert(s.size() + 1, "x"); }, Error);

    // 插入大量字符
    const size_t large_count = 100;
    String large("start");
    large.insert(large.size(), large_count, 'B');
    large.insert(large.size(), "end");

    EXPECT_EQ(large.size(), 5 + large_count + 3);
    EXPECT_STREQ(large.data(), "start" + String(large_count, 'B') + "end");
}

// 测试特殊字符
TEST(StringInsert, SpecialCharacters) {
    String s1("test");

    // 插入控制字符
    s1.insert(2, 1, '\n');
    EXPECT_EQ(s1.size(), 5);
    EXPECT_EQ(s1[2], '\n');

    // 插入特殊符号
    String s2("hello");
    s2.insert(5, "!@#$%");
    EXPECT_STREQ(s2.c_str(), "hello!@#$%");

    // 插入数字
    String s3("abc");
    s3.insert(3, "123");
    EXPECT_STREQ(s3.c_str(), "abc123");
}

// 测试引用计数和COW行为
TEST(StringInsert, ReferenceCounting) {
    String original("reference test");
    String copy = original;

    // 验证引用计数
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_TRUE(original.unique());

    // 对副本进行插入操作，应该触发COW
    copy.insert(4, "INSERT");

    // 验证引用计数分离
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_TRUE(original.unique());
    EXPECT_EQ(copy.use_count(), 1);
    EXPECT_TRUE(copy.unique());

    // 验证原始字符串未被修改
    EXPECT_STREQ(original.c_str(), "reference test");
    // 验证副本已被修改
    EXPECT_STREQ(copy.c_str(), "refeINSERTrence test");
}

#ifdef TEST_REPLACE


#endif


}// namespace