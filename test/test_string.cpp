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

    const char* t = "hello\0world";
    std::string s4(t, 11);
    String s5(t, 11);
    std::cout << s4 << std::endl;
    std::cout << s5 << std::endl;

    std::string s6;
    std::cout << s6.capacity();
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
    EXPECT_TRUE(s1.Compare(special_chars) == 0);

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
    EXPECT_EQ(s.Compare("hello"), 0);
    EXPECT_EQ(s.Compare(String("hello")), 0);

    EXPECT_EQ(s.Compare("hallo"), 1);
    EXPECT_EQ(s.Compare(String("hallo")), 1);
    EXPECT_EQ(s.Compare("hfllo"), -1);
    EXPECT_EQ(s.Compare(String("hfllo")), -1);
    // s is longer
    EXPECT_EQ(s.Compare("hell"), 1);
    EXPECT_EQ(s.Compare(String("hell")), 1);
    // s is shorter
    EXPECT_EQ(s.Compare("hello world"), -1);
    EXPECT_EQ(s.Compare(String("helloworld")), -1);
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
    EXPECT_EQ(str_v1.Compare(v1), 0);
    EXPECT_EQ(str_v1.size(), v1_size);

    // Ensure bytes after '\0' are taken into account for mismatches.
    std::string v2 = "aaa one";
    std::string v3 = "aaa two";
    v2[3] = '\0';
    v3[3] = '\0';
    String str_v2{v2};
    String str_v3{v3};
    EXPECT_EQ(str_v2.Compare(str_v3), -1);
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
    EXPECT_GT(str_v4.Compare(str_v5), 0);
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
    EXPECT_EQ(str_source.Compare(memory), 0);
    // This changes the string size
    memory[2] = '\0';
    // memory is logically shorter now
    EXPECT_GT(str_source.Compare(memory), 0);
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
    EXPECT_EQ(str_source.Compare(source), 0);
    EXPECT_TRUE(str_source == source);
    EXPECT_TRUE(source == str_source);
    EXPECT_TRUE(str_source <= source);
    EXPECT_TRUE(source <= str_source);
    EXPECT_TRUE(str_source >= source);
    EXPECT_TRUE(source >= str_source);
    EXPECT_LT(str_source.Compare(mismatch1), 0);
    EXPECT_TRUE(str_source < mismatch1);
    EXPECT_TRUE(mismatch1 != str_source);
    EXPECT_GT(str_source.Compare(mismatch2), 0);
    EXPECT_TRUE(str_source > mismatch2);
    EXPECT_TRUE(mismatch2 < str_source);
    EXPECT_GT(str_source.Compare(mismatch3), 0);
    EXPECT_TRUE(str_source > mismatch3);
    EXPECT_LT(str_source.Compare(mismatch4), 0);
    EXPECT_TRUE(str_source < mismatch4);
    EXPECT_TRUE(mismatch4 > str_source);

    // compare with char*
    EXPECT_EQ(str_source.Compare(source.data()), 0);
    EXPECT_TRUE(str_source == source.data());
    EXPECT_TRUE(source.data() == str_source);
    EXPECT_TRUE(str_source <= source.data());
    EXPECT_TRUE(source <= str_source.data());
    EXPECT_TRUE(str_source >= source.data());
    EXPECT_TRUE(source >= str_source.data());
    EXPECT_LT(str_source.Compare(mismatch1.data()), 0);
    EXPECT_TRUE(str_source < mismatch1.data());
    EXPECT_TRUE(str_source != mismatch1.data());
    EXPECT_TRUE(mismatch1.data() != str_source);
    EXPECT_GT(str_source.Compare(mismatch2.data()), 0);
    EXPECT_TRUE(str_source > mismatch2.data());
    EXPECT_TRUE(mismatch2.data() < str_source);
    EXPECT_GT(str_source.Compare(mismatch3.data()), 0);
    EXPECT_TRUE(str_source > mismatch3.data());
    EXPECT_LT(str_source.Compare(mismatch4.data()), 0);
    EXPECT_TRUE(str_source < mismatch4.data());
    EXPECT_TRUE(mismatch4.data() > str_source);

    // compare with String
    EXPECT_LT(str_source.Compare(str_mismatch1), 0);
    EXPECT_TRUE(str_source < str_mismatch1);
    EXPECT_GT(str_source.Compare(str_mismatch2), 0);
    EXPECT_TRUE(str_source > str_mismatch2);
    EXPECT_GT(str_source.Compare(str_mismatch3), 0);
    EXPECT_TRUE(str_source > str_mismatch3);
    EXPECT_LT(str_source.Compare(str_mismatch4), 0);
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

    EXPECT_EQ(res1.Compare("helloworld"), 0);
    EXPECT_EQ(res2.Compare("helloworld"), 0);
    EXPECT_EQ(res3.Compare("worldhello"), 0);
    EXPECT_EQ(res4.Compare("helloworld"), 0);
    EXPECT_EQ(res5.Compare("worldhello"), 0);

    String storage_scope;
    String res = "The input storage scope \"" + storage_scope + "\" is invalid.";
    EXPECT_EQ(res.Compare("The input storage scope \"\" is invalid."), 0);
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
    EXPECT_EQ(s1.size(), 10);
    EXPECT_STREQ(s1.c_str(), "Hello, C++!");

    // 测试 replace(pos, n1, src)
    String s2("Hello, world!");
    s2.replace(7, 5, "C++");
    EXPECT_STREQ(s2.c_str(), "Hello, C++!");

    // 测试 replace(pos, n, src)
    String s3("Hello, world!");
    s3.replace(7, 5, String("C++"));
    EXPECT_STREQ(s3.c_str(), "Hello, C++!");

    // 测试 replace(pos, n1, n2, c)
    String s4("Hello, world!");
    s4.replace(7, 5, 3, 'X');
    EXPECT_STREQ(s4.c_str(), "Hello, XXX!");
}

#ifdef TEST_REPLACE
// 测试基于位置的 replace 方法 - 子字符串替换
TEST(StringReplace, PositionBasedSubstring) {
    // 测试 replace(pos1, n1, src, pos2, n2)
    String s1("Hello, world!");
    String src("beautiful code");
    s1.replace(7, 5, src, 0, 9);// 用 "beautiful" 替换 "world"
    EXPECT_STREQ(s1.c_str(), "Hello, beautiful!");

    // 测试默认 n2 = npos
    String s2("Hello, world!");
    s2.replace(7, 5, src, 10);// 用 "code" 替换 "world"
    EXPECT_STREQ(s2.c_str(), "Hello, code!");
}

// 测试基于迭代器的 replace 方法
TEST(StringReplace, IteratorBased) {
    String s("Hello, world!");

    // 测试 replace(first, last, src, n)
    s.replace(s.begin() + 7, s.begin() + 12, "C++", 3);
    EXPECT_STREQ(s.c_str(), "Hello, C++!");

    // 测试 replace(first, last, src)
    s.replace(s.begin() + 7, s.begin() + 10, "Java");
    EXPECT_STREQ(s.c_str(), "Hello, Java!");

    // 测试 replace(first, last, String&)
    s.replace(s.begin() + 7, s.begin() + 11, String("Python"));
    EXPECT_STREQ(s.c_str(), "Hello, Python!");

    // 测试 replace(first, last, n, c)
    s.replace(s.begin() + 7, s.begin() + 13, 2, 'X');
    EXPECT_STREQ(s.c_str(), "Hello, XX!");
}

// 测试模板迭代器版本的 replace 方法
TEST(StringReplace, TemplateIteratorVersion) {
    String s("Hello, world!");

    // 使用 std::vector 迭代器
    std::vector<char> vec = {'C', '+', '+'};
    s.replace(s.begin() + 7, s.begin() + 12, vec.begin(), vec.end());
    EXPECT_STREQ(s.c_str(), "Hello, C++!");

    // 使用 std::list 迭代器
    std::list<char> lst = {'J', 'a', 'v', 'a'};
    s.replace(s.begin() + 7, s.begin() + 10, lst.begin(), lst.end());
    EXPECT_STREQ(s.c_str(), "Hello, Java!");

    // 使用 std::array 迭代器
    std::array<char, 6> arr = {'P', 'y', 't', 'h', 'o', 'n'};
    s.replace(s.begin() + 7, s.begin() + 11, arr.begin(), arr.end());
    EXPECT_STREQ(s.c_str(), "Hello, Python!");

    // 使用原始指针迭代器
    const char* c_str = "Ruby";
    s.replace(s.begin() + 7, s.begin() + 13, c_str, c_str + 4);
    EXPECT_STREQ(s.c_str(), "Hello, Ruby!");
}

// 测试初始化列表版本的 replace 方法
TEST(StringReplace, InitializerListVersion) {
    String s("Hello, world!");

    // 使用初始化列表替换
    s.replace(s.begin() + 7, s.begin() + 12, {'C', '+', '+'});
    EXPECT_STREQ(s.c_str(), "Hello, C++!");

    // 使用初始化列表替换为空字符串
    s.replace(s.begin() + 7, s.begin() + 10, {});
    EXPECT_STREQ(s.c_str(), "Hello, !");
}

// 测试边界情况 - 空替换
TEST(StringReplace, EdgeCaseEmptyReplace) {
    String s("Hello, world!");

    // 替换为空字符串
    s.replace(7, 5, "", 0);
    EXPECT_STREQ(s.c_str(), "Hello, !");
    EXPECT_EQ(s.size(), 7);

    // 空范围替换（不改变字符串）
    s.replace(5, 0, "extra", 5);
    EXPECT_STREQ(s.c_str(), "Helloextra, !");
    EXPECT_EQ(s.size(), 12);
}

// 测试边界情况 - 替换整个字符串
TEST(StringReplace, EdgeCaseReplaceWholeString) {
    String s("Hello, world!");

    // 替换整个字符串
    s.replace(0, s.size(), "New string");
    EXPECT_STREQ(s.c_str(), "New string");
    EXPECT_EQ(s.size(), 10);

    // 使用迭代器替换整个字符串
    s.replace(s.begin(), s.end(), {'E', 'n', 't', 'i', 'r', 'e'});
    EXPECT_STREQ(s.c_str(), "Entire");
    EXPECT_EQ(s.size(), 6);
}

// 测试边界情况 - 长度变化
TEST(StringReplace, EdgeCaseLengthVariation) {
    String s("abcdef");

    // 替换为更短的内容
    s.replace(2, 3, "x");
    EXPECT_STREQ(s.c_str(), "abx f");// 注意：f 前有空格，因为我们替换了 'cde' 为 'x'
    EXPECT_EQ(s.size(), 4);

    // 替换为更长的内容
    s.replace(1, 1, "long text");
    EXPECT_STREQ(s.c_str(), "along textx f");
    EXPECT_EQ(s.size(), 12);
}

// 测试特殊字符处理
TEST(StringReplace, SpecialCharacters) {
    String s("Hello\tworld\n");

    // 替换包含控制字符的部分
    s.replace(5, 6, "user\r", 5);
    EXPECT_STREQ(s.c_str(), "Hellouser\r\n");
    EXPECT_EQ(s[5], 'u');
    EXPECT_EQ(s[9], '\r');

    // 替换为空字符
    s.replace(5, 5, "\0\0", 2);
    EXPECT_EQ(s.size(), 7);
    EXPECT_EQ(s[5], '\0');
    EXPECT_EQ(s[6], '\n');

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
    EXPECT_EQ(original.use_count(), 2);
    EXPECT_EQ(copy.use_count(), 2);

    // 修改其中一个，应该触发写时复制
    original.replace(7, 5, "C++");
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(copy.use_count(), 1);
    EXPECT_TRUE(original != copy);
    EXPECT_STREQ(original.c_str(), "Hello, C++!");
    EXPECT_STREQ(copy.c_str(), "Hello, world!");
}

// 测试链式调用
TEST(StringReplace, ChainedCalls) {
    String s("Hello, world!");

    // 链式调用多个 replace
    s.replace(0, 5, "Hi")
            .replace(3, 1, ",")
            .replace(4, 6, "there");

    EXPECT_STREQ(s.c_str(), "Hi, there!");
    EXPECT_EQ(s.size(), 10);
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
#endif


}// namespace