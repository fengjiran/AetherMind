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

// 测试 starts_with(const String& str) 函数
TEST(StringStartsWithTest, StartsWithString) {
    // 基本功能测试
    String str("Hello, World!");
    String prefix1("Hello");
    String prefix2("Hello,");
    String not_prefix("World");

    EXPECT_TRUE(str.starts_with(prefix1));
    EXPECT_TRUE(str.starts_with(prefix2));
    EXPECT_FALSE(str.starts_with(not_prefix));

    // 相等的字符串
    String same_str("Hello, World!");
    EXPECT_TRUE(str.starts_with(same_str));

    // 空前缀
    String empty_prefix;
    EXPECT_TRUE(str.starts_with(empty_prefix));// 空字符串是任何字符串的前缀

    // 前缀长度大于原字符串
    String long_prefix("Hello, World! Extra");
    EXPECT_FALSE(str.starts_with(long_prefix));

    // 空字符串测试
    String empty_str;
    EXPECT_TRUE(empty_str.starts_with(empty_prefix));// 空字符串以空字符串开头
    EXPECT_FALSE(empty_str.starts_with(prefix1));    // 空字符串不以非空字符串开头
}

// 测试 starts_with(const_pointer str) 函数
TEST(StringStartsWithTest, StartsWithConstPointer) {
    // 基本功能测试
    String str("Hello, World!");

    EXPECT_TRUE(str.starts_with("Hello"));
    EXPECT_TRUE(str.starts_with("Hello,"));
    EXPECT_FALSE(str.starts_with("World"));
    EXPECT_TRUE(str.starts_with("Hello, World!"));// 相等的字符串

    // 空字符串参数
    EXPECT_TRUE(str.starts_with(""));// 空字符串是任何字符串的前缀

    // 前缀长度大于原字符串
    EXPECT_FALSE(str.starts_with("Hello, World! Extra"));

    // 空字符串测试
    String empty_str;
    EXPECT_TRUE(empty_str.starts_with(""));      // 空字符串以空字符串开头
    EXPECT_FALSE(empty_str.starts_with("Hello"));// 空字符串不以非空字符串开头

    // 特殊字符测试
    String special_str("123!@#$");
    EXPECT_TRUE(special_str.starts_with("123"));
    EXPECT_TRUE(special_str.starts_with("123!"));
    EXPECT_FALSE(special_str.starts_with("!@#"));
}

// 测试 starts_with(value_type c) 函数
TEST(StringStartsWithTest, StartsWithChar) {
    // 基本功能测试
    String str("Hello, World!");

    EXPECT_TRUE(str.starts_with('H'));
    EXPECT_FALSE(str.starts_with('h'));// 区分大小写
    EXPECT_FALSE(str.starts_with('W'));

    // 空字符串测试
    String empty_str;
    EXPECT_FALSE(empty_str.starts_with('H'));// 空字符串不以任何字符开头

    // 特殊字符测试
    String special_str("!@#$123");
    EXPECT_TRUE(special_str.starts_with('!'));
    EXPECT_FALSE(special_str.starts_with('1'));

    // 多字节字符边界测试（注意：这里只测试第一个字节）
    // String multi_str("你好，世界！");
    // 警告：直接比较多字节字符的第一个字节可能不准确，取决于编码
    // 这里仅作为示例，实际应用中应根据具体编码处理
}

// 测试所有starts_with重载的一致性
TEST(StringStartsWithTest, ConsistencyBetweenOverloads) {
    String str("Hello, World!");
    char c = 'H';
    const char* cp = "H";
    String sp("H");

    // 测试不同重载函数在相同条件下的一致性
    EXPECT_EQ(str.starts_with(c), str.starts_with(cp));
    EXPECT_EQ(str.starts_with(c), str.starts_with(sp));

    // 更长的前缀
    const char* cpp = "Hello";
    String spp("Hello");
    EXPECT_EQ(str.starts_with(cpp), str.starts_with(spp));
}

// 测试COW机制下的starts_with
TEST(StringStartsWithTest, StartsWithWithCOW) {
    // 创建一个字符串
    String str("Hello, World!");

    // 创建一个共享str内容的新字符串（COW）
    String str_cow = str;

    // 对str_cow调用starts_with不应修改共享数据
    EXPECT_TRUE(str_cow.starts_with("Hello"));

    // 验证str和str_cow仍然相等
    EXPECT_EQ(str, str_cow);

    // 修改原始字符串，触发COW
    str += " More";

    // 验证两个字符串现在不同
    EXPECT_FALSE(str == str_cow);

    // 验证各自的starts_with行为
    EXPECT_TRUE(str.starts_with("Hello"));
    EXPECT_TRUE(str_cow.starts_with("Hello"));
}

// 测试本地缓冲区切换情况下的starts_with
TEST(StringStartsWithTest, StartsWithWithLocalBuffer) {
    // 短字符串（使用本地缓冲区）
    String short_str("abc");
    EXPECT_TRUE(short_str.starts_with("a"));
    EXPECT_TRUE(short_str.starts_with("ab"));
    EXPECT_TRUE(short_str.starts_with("abc"));
    EXPECT_FALSE(short_str.starts_with("abcd"));

    // 长字符串（使用堆分配）
    String long_str("a");
    for (int i = 0; i < 30; ++i) {
        long_str += "a";
    }

    String long_prefix = long_str.substr(0, 10);
    EXPECT_TRUE(long_str.starts_with(long_prefix));
    EXPECT_TRUE(long_str.starts_with("a"));
}

// 测试与其他字符串操作的组合
TEST(StringStartsWithTest, StartsWithWithOtherOperations) {
    String str("Hello, World!");

    // 与append组合
    String appended = str;
    appended.append(" More");
    EXPECT_TRUE(appended.starts_with(str));// 原始字符串是追加后字符串的前缀

    // 与insert组合
    String inserted = str;
    inserted.insert(0, "Prefix: ");
    EXPECT_FALSE(inserted.starts_with(str));      // 原始字符串不再是前缀
    EXPECT_TRUE(inserted.starts_with("Prefix: "));// 新的前缀

    // 与replace组合
    String replaced = str;
    replaced.replace(0, 5, "Hi");
    EXPECT_FALSE(replaced.starts_with("Hello"));// 不再以Hello开头
    EXPECT_TRUE(replaced.starts_with("Hi"));    // 以Hi开头
}

// 测试长字符串中的性能和正确性
TEST(StringStartsWithTest, StartsWithInLongString) {
    // 创建一个长字符串
    String long_str;
    for (int i = 0; i < 1000; ++i) {
        long_str += "abc";
    }

    // 测试短前缀
    EXPECT_TRUE(long_str.starts_with("a"));
    EXPECT_TRUE(long_str.starts_with("ab"));
    EXPECT_TRUE(long_str.starts_with("abc"));

    // 测试较长前缀
    String prefix;
    for (int i = 0; i < 10; ++i) {
        prefix += "abc";
    }
    EXPECT_TRUE(long_str.starts_with(prefix));

    // 测试不是前缀的情况
    prefix += "x";
    EXPECT_FALSE(long_str.starts_with(prefix));
}

// 测试多字节字符
TEST(StringStartsWithTest, StartsWithWithMultiByteChars) {
    // 包含多字节字符的字符串
    String utf8_str("你好世界HelloWorld");

    // 测试以多字节字符开头
    EXPECT_TRUE(utf8_str.starts_with(String("你")));
    EXPECT_TRUE(utf8_str.starts_with(String("你好")));

    // 测试以ASCII字符开头
    String mixed_str("Hello你好世界");
    EXPECT_TRUE(mixed_str.starts_with("Hello"));
    EXPECT_FALSE(mixed_str.starts_with(String("你")));

    // 注意：对于UTF-8字符串，直接使用const char*可能需要特别小心
    // 确保前缀的字节序列与原字符串的开头字节序列完全匹配
}

// 测试边界情况
TEST(StringStartsWithTest, StartsWithEdgeCases) {
    // 只有一个字符的字符串
    String single_char("a");
    EXPECT_TRUE(single_char.starts_with('a'));
    EXPECT_TRUE(single_char.starts_with("a"));
    EXPECT_TRUE(single_char.starts_with(String("a")));
    EXPECT_FALSE(single_char.starts_with('b'));

    // 前缀等于字符串的情况
    String str("exact match");
    EXPECT_TRUE(str.starts_with(str));

    // 包含空字符的字符串
    String with_null("Hello\0World", 11);// 包含null字符的字符串
    EXPECT_TRUE(with_null.starts_with("Hello"));
    // 注意：const char*版本会在null字符处停止比较
    EXPECT_FALSE(with_null.starts_with("HelloWorld"));// 因为const char*会在第一个null处停止
}

// 测试 ends_with(const String& str) 函数
TEST(StringEndsWithTest, EndsWithString) {
    // 基本功能测试
    String str("Hello, World!");
    String suffix1("World!");
    String suffix2("ld!");
    String not_suffix("Hello");

    EXPECT_TRUE(str.ends_with(suffix1));
    EXPECT_TRUE(str.ends_with(suffix2));
    EXPECT_FALSE(str.ends_with(not_suffix));

    // 相等的字符串
    String same_str("Hello, World!");
    EXPECT_TRUE(str.ends_with(same_str));

    // 空后缀
    String empty_suffix;
    EXPECT_TRUE(str.ends_with(empty_suffix));// 空字符串是任何字符串的后缀

    // 后缀长度大于原字符串
    String long_suffix("Extra Hello, World!");
    EXPECT_FALSE(str.ends_with(long_suffix));

    // 空字符串测试
    String empty_str;
    EXPECT_TRUE(empty_str.ends_with(empty_suffix));// 空字符串以空字符串结尾
    EXPECT_FALSE(empty_str.ends_with(suffix1));    // 空字符串不以非空字符串结尾
}

// 测试 ends_with(const_pointer str) 函数
TEST(StringEndsWithTest, EndsWithConstPointer) {
    // 基本功能测试
    String str("Hello, World!");

    EXPECT_TRUE(str.ends_with("World!"));
    EXPECT_TRUE(str.ends_with("ld!"));
    EXPECT_FALSE(str.ends_with("Hello"));
    EXPECT_TRUE(str.ends_with("Hello, World!"));// 相等的字符串

    // 空字符串参数
    EXPECT_TRUE(str.ends_with(""));// 空字符串是任何字符串的后缀

    // 后缀长度大于原字符串
    EXPECT_FALSE(str.ends_with("Extra Hello, World!"));

    // 空字符串测试
    String empty_str;
    EXPECT_TRUE(empty_str.ends_with(""));       // 空字符串以空字符串结尾
    EXPECT_FALSE(empty_str.ends_with("World!"));// 空字符串不以非空字符串结尾

    // 特殊字符测试
    String special_str("123!@#$");
    EXPECT_TRUE(special_str.ends_with("!@#$"));
    EXPECT_TRUE(special_str.ends_with("#$"));
    EXPECT_FALSE(special_str.ends_with("123"));
}

// 测试 ends_with(value_type c) 函数
TEST(StringEndsWithTest, EndsWithChar) {
    // 基本功能测试
    String str("Hello, World!");

    EXPECT_TRUE(str.ends_with('!'));
    EXPECT_FALSE(str.ends_with('?'));
    EXPECT_FALSE(str.ends_with('H'));

    // 空字符串测试
    String empty_str;
    EXPECT_FALSE(empty_str.ends_with('!'));// 空字符串不以任何字符结尾

    // 特殊字符测试
    String special_str("123!@#$");
    EXPECT_TRUE(special_str.ends_with('$'));
    EXPECT_FALSE(special_str.ends_with('1'));

    // 只有一个字符的字符串
    String single_char("a");
    EXPECT_TRUE(single_char.ends_with('a'));
    EXPECT_FALSE(single_char.ends_with('b'));
}

// 测试所有ends_with重载的一致性
TEST(StringEndsWithTest, ConsistencyBetweenOverloads) {
    String str("Hello, World!");
    char c = '!';
    const char* cp = "!";
    String sp("!");

    // 测试不同重载函数在相同条件下的一致性
    EXPECT_EQ(str.ends_with(c), str.ends_with(cp));
    EXPECT_EQ(str.ends_with(c), str.ends_with(sp));

    // 更长的后缀
    const char* cpp = "World!";
    String spp("World!");
    EXPECT_EQ(str.ends_with(cpp), str.ends_with(spp));
}

// 测试COW机制下的ends_with
TEST(StringEndsWithTest, EndsWithWithCOW) {
    // 创建一个字符串
    String str("Hello, World!");

    // 创建一个共享str内容的新字符串（COW）
    String str_cow = str;

    // 对str_cow调用ends_with不应修改共享数据
    EXPECT_TRUE(str_cow.ends_with("World!"));

    // 验证str和str_cow仍然相等
    EXPECT_EQ(str, str_cow);

    // 修改原始字符串，触发COW
    str = "Hello, Universe!";

    // 验证两个字符串现在不同
    EXPECT_FALSE(str == str_cow);

    // 验证各自的ends_with行为
    EXPECT_TRUE(str.ends_with("Universe!"));
    EXPECT_TRUE(str_cow.ends_with("World!"));
}

// 测试本地缓冲区切换情况下的ends_with
TEST(StringEndsWithTest, EndsWithWithLocalBuffer) {
    // 短字符串（使用本地缓冲区）
    String short_str("abc");
    EXPECT_TRUE(short_str.ends_with("c"));
    EXPECT_TRUE(short_str.ends_with("bc"));
    EXPECT_TRUE(short_str.ends_with("abc"));
    EXPECT_FALSE(short_str.ends_with("abcd"));

    // 长字符串（使用堆分配）
    String long_str;
    for (int i = 0; i < 30; ++i) {
        long_str += "a";
    }
    long_str += "bc";

    String long_suffix("bc");
    EXPECT_TRUE(long_str.ends_with(long_suffix));
    EXPECT_TRUE(long_str.ends_with("c"));
}

// 测试与其他字符串操作的组合
TEST(StringEndsWithTest, EndsWithWithOtherOperations) {
    String str("Hello, World!");

    // 与append组合
    String appended = str;
    appended.append(" More");
    EXPECT_FALSE(appended.ends_with(str));   // 原始字符串不再是追加后字符串的后缀
    EXPECT_TRUE(appended.ends_with(" More"));// 新的后缀

    // 与insert组合
    String inserted = str;
    inserted.insert(0, "Prefix: ");
    EXPECT_TRUE(inserted.ends_with(str));// 原始字符串仍然是后缀

    // 与replace组合
    String replaced = str;
    replaced.replace(7, 5, "Universe");
    EXPECT_FALSE(replaced.ends_with("World!"));  // 不再以World!结尾
    EXPECT_TRUE(replaced.ends_with("Universe!"));// 以Universe!结尾
}

// 测试长字符串中的性能和正确性
TEST(StringEndsWithTest, EndsWithInLongString) {
    // 创建一个长字符串
    String long_str;
    for (int i = 0; i < 1000; ++i) {
        long_str += "abc";
    }
    long_str += "xyz";

    // 测试短后缀
    EXPECT_TRUE(long_str.ends_with("z"));
    EXPECT_TRUE(long_str.ends_with("yz"));
    EXPECT_TRUE(long_str.ends_with("xyz"));

    // 测试较长后缀
    String suffix;
    for (int i = 0; i < 10; ++i) {
        suffix += "abc";
    }
    suffix += "xyz";

    // 创建一个具有相同后缀的字符串
    String test_str = long_str;
    EXPECT_TRUE(test_str.ends_with(suffix));

    // 测试不是后缀的情况
    suffix += "x";
    EXPECT_FALSE(long_str.ends_with(suffix));
}

// 测试多字节字符
TEST(StringEndsWithTest, EndsWithWithMultiByteChars) {
    // 包含多字节字符的字符串
    String utf8_str("HelloWorld你好世界");

    // 测试以多字节字符结尾
    EXPECT_TRUE(utf8_str.ends_with(String("界")));
    EXPECT_TRUE(utf8_str.ends_with(String("世界")));

    // 测试以ASCII字符结尾
    String mixed_str("你好世界HelloWorld");
    EXPECT_TRUE(mixed_str.ends_with("World"));
    EXPECT_FALSE(mixed_str.ends_with(String("界")));

    // 注意：对于UTF-8字符串，直接使用const char*可能需要特别小心
    // 确保后缀的字节序列与原字符串的末尾字节序列完全匹配
}

// 测试边界情况
TEST(StringEndsWithTest, EndsWithEdgeCases) {
    // 只有一个字符的字符串
    String single_char("a");
    EXPECT_TRUE(single_char.ends_with('a'));
    EXPECT_TRUE(single_char.ends_with("a"));
    EXPECT_TRUE(single_char.ends_with(String("a")));
    EXPECT_FALSE(single_char.ends_with('b'));

    // 后缀等于字符串的情况
    String str("exact match");
    EXPECT_TRUE(str.ends_with(str));

    // 包含空字符的字符串
    String with_null("Hello\0World", 11);// 包含null字符的字符串
    EXPECT_FALSE(with_null.ends_with("World"));
    // 注意：const char*版本会在null字符处停止比较
    // 但在这里，我们的字符串包含null字符作为有效字符
}

// 测试compare函数的边缘情况影响
TEST(StringEndsWithTest, EndsWithCompareEdgeCases) {
    // 测试compare函数处理的特殊情况
    String str1("abcde");
    String str2("cde");
    String str3("cdf");

    // 确保compare正确处理不同长度和内容的字符串
    EXPECT_TRUE(str1.ends_with(str2));
    EXPECT_FALSE(str1.ends_with(str3));

    // 大小写测试
    String upper("ABCDE");
    String lower("abcde");
    EXPECT_FALSE(upper.ends_with(lower.substr(2)));// 区分大小写
}

#ifdef TEST_REPLACE
#include <container/string.h>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace AetherMind;

#endif

}  // namespace
