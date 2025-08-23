//
// Created by 赵丹 on 2025/8/22.
//
#include "container/string.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

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
}

TEST(String, Assignment) {
    String s{std::string{"hello"}};
    s = std::string{"world"};
    EXPECT_TRUE(s == "world");
    EXPECT_TRUE(s.size() == 5);
    EXPECT_EQ(s.use_count(), 1);
    EXPECT_TRUE(s.unique());
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

}// namespace