//
// Created by richard on 11/8/25.
//
#include "utils/qualified_name.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

// 测试默认构造函数和基本功能
TEST(QualifiedName, DefaultConstructor) {
    QualifiedName qn;
    EXPECT_TRUE(qn.GetAtoms().empty());
    EXPECT_TRUE(qn.GetName().empty());
    EXPECT_TRUE(qn.GetPrefix().empty());
    EXPECT_TRUE(qn.GetQualifiedName().empty());
}

// 测试字符串构造函数 - 简单名称
TEST(QualifiedName, StringConstructorSimple) {
    QualifiedName qn("simple_name");
    EXPECT_EQ(qn.GetAtoms().size(), 1);
    EXPECT_EQ(qn.GetAtoms()[0], "simple_name");
    EXPECT_EQ(qn.GetName(), "simple_name");
    EXPECT_TRUE(qn.GetPrefix().empty());
    EXPECT_EQ(qn.GetQualifiedName(), "simple_name");
}

// 测试字符串构造函数 - 带点的名称
TEST(QualifiedName, StringConstructorQualified) {
    QualifiedName qn("foo.bar.baz");
    EXPECT_EQ(qn.GetAtoms().size(), 3);
    EXPECT_EQ(qn.GetAtoms()[0], "foo");
    EXPECT_EQ(qn.GetAtoms()[1], "bar");
    EXPECT_EQ(qn.GetAtoms()[2], "baz");
    EXPECT_EQ(qn.GetName(), "baz");
    EXPECT_EQ(qn.GetPrefix(), "foo.bar");
    EXPECT_EQ(qn.GetQualifiedName(), "foo.bar.baz");
}

// 测试const char*构造函数
TEST(QualifiedName, CharConstructor) {
    QualifiedName qn("char.name");
    EXPECT_EQ(qn.GetAtoms().size(), 2);
    EXPECT_EQ(qn.GetAtoms()[0], "char");
    EXPECT_EQ(qn.GetAtoms()[1], "name");
    EXPECT_EQ(qn.GetName(), "name");
    EXPECT_EQ(qn.GetPrefix(), "char");
    EXPECT_EQ(qn.GetQualifiedName(), "char.name");
}

// 测试向量构造函数
TEST(QualifiedName, VectorConstructor) {
    std::vector<std::string> atoms = {"vec", "tor", "con", "struct"};
    QualifiedName qn(atoms);
    EXPECT_EQ(qn.GetAtoms().size(), 4);
    EXPECT_EQ(qn.GetAtoms()[0], "vec");
    EXPECT_EQ(qn.GetAtoms()[1], "tor");
    EXPECT_EQ(qn.GetAtoms()[2], "con");
    EXPECT_EQ(qn.GetAtoms()[3], "struct");
    EXPECT_EQ(qn.GetName(), "struct");
    EXPECT_EQ(qn.GetPrefix(), "vec.tor.con");
    EXPECT_EQ(qn.GetQualifiedName(), "vec.tor.con.struct");
}

// 测试前缀+名称构造函数
TEST(QualifiedName, PrefixNameConstructor) {
    QualifiedName prefix("base.prefix");
    QualifiedName qn(prefix, "suffix");
    EXPECT_EQ(qn.GetAtoms().size(), 3);
    EXPECT_EQ(qn.GetAtoms()[0], "base");
    EXPECT_EQ(qn.GetAtoms()[1], "prefix");
    EXPECT_EQ(qn.GetAtoms()[2], "suffix");
    EXPECT_EQ(qn.GetName(), "suffix");
    EXPECT_EQ(qn.GetPrefix(), "base.prefix");
    EXPECT_EQ(qn.GetQualifiedName(), "base.prefix.suffix");
}

// 测试IsPrefixOf方法
TEST(QualifiedName, IsPrefixOf) {
    QualifiedName prefix("foo.bar");
    QualifiedName full("foo.bar.baz");
    QualifiedName different("foo.baz");
    QualifiedName longer("foo.bar.baz.qux");
    QualifiedName shorter("foo");

    EXPECT_TRUE(prefix.IsPrefixOf(full));
    EXPECT_TRUE(prefix.IsPrefixOf(longer));
    EXPECT_TRUE(shorter.IsPrefixOf(prefix));
    EXPECT_FALSE(full.IsPrefixOf(prefix));
    EXPECT_FALSE(prefix.IsPrefixOf(different));
    EXPECT_FALSE(different.IsPrefixOf(prefix));
    EXPECT_FALSE(prefix.IsPrefixOf(shorter));

    // 测试相同名称的情况
    EXPECT_TRUE(prefix.IsPrefixOf(prefix));
}

// 测试相等操作符
TEST(QualifiedName, EqualityOperators) {
    QualifiedName qn1("foo.bar");
    QualifiedName qn2("foo.bar");
    QualifiedName qn3("foo.baz");

    EXPECT_TRUE(qn1 == qn2);
    EXPECT_FALSE(qn1 != qn2);
    EXPECT_FALSE(qn1 == qn3);
    EXPECT_TRUE(qn1 != qn3);

    // 测试通过不同构造函数创建的相同名称
    std::vector<std::string> atoms = {"foo", "bar"};
    QualifiedName qn4(atoms);
    EXPECT_TRUE(qn1 == qn4);
}

// 测试边界情况
TEST(QualifiedName, EdgeCases) {
    // 测试单字符名称
    QualifiedName single_char("a");
    EXPECT_EQ(single_char.GetAtoms().size(), 1);
    EXPECT_EQ(single_char.GetName(), "a");
    EXPECT_TRUE(single_char.GetPrefix().empty());

    // 测试长名称
    std::string long_name;
    for (int i = 0; i < 100; ++i) {
        long_name += "part" + std::to_string(i);
        if (i < 99) long_name += ".";
    }
    QualifiedName long_qn(long_name);
    EXPECT_EQ(long_qn.GetAtoms().size(), 100);
    EXPECT_EQ(long_qn.GetName(), "part99");

    // 测试前缀+名称构造函数的边界情况
    QualifiedName empty_prefix;
    QualifiedName qn_from_empty(empty_prefix, "name");
    EXPECT_EQ(qn_from_empty.GetAtoms().size(), 1);
    EXPECT_EQ(qn_from_empty.GetName(), "name");
    EXPECT_TRUE(qn_from_empty.GetPrefix().empty());
}

}// namespace