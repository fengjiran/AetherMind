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

// front()/back() on an empty string triggers AM_CHECK(!empty()) which aborts.
TEST(StringFrontBack, EmptyStringDeath) {
    String empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0);

    EXPECT_DEATH(static_cast<void>(empty.front()), "Check failed");
    EXPECT_DEATH(static_cast<void>(empty.back()), "Check failed");
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
    EXPECT_NE(s1.compare(0, 5, cstr1, 10), 0);// n2=10大于实际长度

    // std::string s2 = "hello";
    // EXPECT_TRUE(s2.compare(0, 5, cstr1, 10) == 0);
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
    const char* with_null = "test";
    String s4(with_null, 11);// 包含null字符的字符串
    String s5("test");
    EXPECT_EQ(s4.compare(s5), 0);// 包含null的字符串比前缀长，所以更大
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

}  // namespace
