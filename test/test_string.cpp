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
    using namespace std;
    string source = "this is a string";
    string mismatch = "mismatch";
    String s{source};

    EXPECT_EQ(std::strcmp(s.c_str(), source.data()), 0);
    EXPECT_NE(std::strcmp(s.c_str(), mismatch.data()), 0);
}

TEST(String, hash) {
    using namespace std;
    string source = "this is a string";
    String s{source};

    std::unordered_map<String, std::string> map;
    String k1{string{"k1"}};
    string v1{"v1"};
    String k2{string{"k2"}};
    string v2{"v2"};
    map[k1] = v1;
    map[k2] = v2;

    EXPECT_EQ(map[k1], v1);
    EXPECT_EQ(map[k2], v2);
}

}// namespace