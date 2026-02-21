//
// Created by richard on 11/25/25.
//

#include "symbol.h"

#include <gtest/gtest.h>

#ifdef TEST_SYMBOL
namespace {
using namespace aethermind;

// 测试默认构造函数和转换为SymId
TEST(SymbolTest, DefaultConstructor) {
    Symbol sym;
    EXPECT_EQ(static_cast<SymId>(sym), 0);

    // 测试带参数的构造函数
    Symbol sym2(42);
    EXPECT_EQ(static_cast<SymId>(sym2), 42);
}

// 测试FromQualString方法
TEST(SymbolTest, FromQualString) {
    // 测试基本的限定字符串
    Symbol sym1 = Symbol::FromQualString("prim::add");
    EXPECT_NE(static_cast<SymId>(sym1), 0);

    Symbol sym2 = Symbol::FromQualString("cuda::kernel");
    EXPECT_NE(static_cast<SymId>(sym2), 0);

    Symbol sym3 = Symbol::FromQualString("attr::name");
    EXPECT_NE(static_cast<SymId>(sym3), 0);

    // 测试相同字符串返回相同的Symbol
    Symbol sym1_dup = Symbol::FromQualString("prim::add");
    EXPECT_EQ(static_cast<SymId>(sym1), static_cast<SymId>(sym1_dup));
}


// 测试FromDomainAndUnqualifiedName方法
TEST(SymbolTest, FromDomainAndUnqualifiedName) {
    // 测试有效的域名前缀
    Symbol sym1 = Symbol::FromDomainAndUnqualifiedName("aethermind.prim", "add");
    EXPECT_NE(static_cast<SymId>(sym1), 0);

    Symbol sym2 = Symbol::FromDomainAndUnqualifiedName("aethermind.cuda", "kernel");
    EXPECT_NE(static_cast<SymId>(sym2), 0);

    Symbol sym3 = Symbol::FromDomainAndUnqualifiedName("aethermind.attr", "name");
    EXPECT_NE(static_cast<SymId>(sym3), 0);

    // 测试与FromQualString的一致性
    Symbol sym4 = Symbol::FromQualString("prim::add");
    Symbol sym5 = Symbol::FromDomainAndUnqualifiedName("aethermind.prim", "add");
    EXPECT_EQ(static_cast<SymId>(sym4), static_cast<SymId>(sym5));
}

// 测试ToQualString方法
TEST(SymbolTest, ToQualString) {
    // 创建一些Symbol并测试其限定字符串表示
    Symbol sym1 = Symbol::FromQualString("prim::add");
    EXPECT_FALSE(sym1.ToQualString().empty());

    Symbol sym2 = Symbol::FromQualString("cuda::kernel");
    EXPECT_FALSE(sym2.ToQualString().empty());

    // 测试往返转换
    String qual_str = "attr::test_name";
    Symbol sym3 = Symbol::FromQualString(qual_str);
    EXPECT_EQ(sym3.ToQualString(), qual_str);
}

// 测试ToUnQualString方法
TEST(SymbolTest, ToUnQualString) {
    // 创建Symbol并测试其非限定字符串表示
    Symbol sym1 = Symbol::FromQualString("prim::add");
    EXPECT_FALSE(sym1.ToUnQualString().empty());

    Symbol sym2 = Symbol::FromQualString("cuda::kernel");
    EXPECT_FALSE(sym2.ToUnQualString().empty());
}

// 测试GetDomainString方法
TEST(SymbolTest, GetDomainString) {
    // 测试域名字符串获取
    Symbol sym1 = Symbol::FromQualString("prim::add");
    EXPECT_FALSE(sym1.GetDomainString().empty());
    EXPECT_TRUE(sym1.GetDomainString().find(GetDomainPrefix()) == 0);

    Symbol sym2 = Symbol::FromQualString("cuda::kernel");
    EXPECT_FALSE(sym2.GetDomainString().empty());
    EXPECT_TRUE(sym2.GetDomainString().find(GetDomainPrefix()) == 0);
}


// 测试NS方法
TEST(SymbolTest, NS) {
    // 测试命名空间获取
    Symbol sym1 = Symbol::FromQualString("prim::add");
    Symbol ns1 = sym1.NS();
    EXPECT_EQ(static_cast<SymId>(ns1), 0);

    Symbol sym2 = Symbol::FromQualString("prim::multiply");
    Symbol ns2 = sym2.NS();
    // 同一命名空间的不同Symbol应该返回相同的命名空间Symbol
    EXPECT_EQ(static_cast<SymId>(ns1), static_cast<SymId>(ns2));
}

// 测试prim、cuda、attr静态方法
TEST(SymbolTest, NamespaceConstructors) {
    // 测试prim命名空间构造函数
    Symbol sym1 = Symbol::prim("add");
    Symbol sym1_expected = Symbol::FromQualString("prim::add");
    EXPECT_EQ(static_cast<SymId>(sym1), static_cast<SymId>(sym1_expected));

    // 测试cuda命名空间构造函数
    Symbol sym2 = Symbol::cuda("kernel");
    Symbol sym2_expected = Symbol::FromQualString("cuda::kernel");
    EXPECT_EQ(static_cast<SymId>(sym2), static_cast<SymId>(sym2_expected));

    // 测试attr命名空间构造函数
    Symbol sym3 = Symbol::attr("name");
    Symbol sym3_expected = Symbol::FromQualString("attr::name");
    EXPECT_EQ(static_cast<SymId>(sym3), static_cast<SymId>(sym3_expected));
}

// 测试IsPrim、IsCuda、IsAttr方法
TEST(SymbolTest, NamespaceChecks) {
    // 测试prim命名空间检查
    Symbol sym1 = Symbol::prim("add");
    EXPECT_TRUE(sym1.IsPrim());
    EXPECT_FALSE(sym1.IsCuda());
    EXPECT_FALSE(sym1.IsAttr());

    // 测试cuda命名空间检查
    Symbol sym2 = Symbol::cuda("kernel");
    EXPECT_FALSE(sym2.IsPrim());
    EXPECT_TRUE(sym2.IsCuda());
    EXPECT_FALSE(sym2.IsAttr());

    // 测试attr命名空间检查
    Symbol sym3 = Symbol::attr("name");
    EXPECT_FALSE(sym3.IsPrim());
    EXPECT_FALSE(sym3.IsCuda());
    EXPECT_TRUE(sym3.IsAttr());
}

// 测试相等运算符重载
TEST(SymbolTest, EqualityOperator) {
    // 测试相同Symbol的相等性
    Symbol sym1 = Symbol::FromQualString("prim::add");
    Symbol sym1_dup = Symbol::FromQualString("prim::add");
    EXPECT_TRUE(sym1 == sym1_dup);

    // 测试不同Symbol的不等性
    Symbol sym2 = Symbol::FromQualString("prim::multiply");
    EXPECT_FALSE(sym1 == sym2);

    // 测试默认构造的Symbol
    Symbol sym_default;
    Symbol sym_zero(0);
    EXPECT_TRUE(sym_default == sym_zero);
}

// 测试std::hash特化
TEST(SymbolTest, StdHash) {
    // 测试哈希函数
    Symbol sym1 = Symbol::FromQualString("prim::add");
    std::hash<Symbol> hasher;
    size_t hash_val = hasher(sym1);

    // 测试相同Symbol具有相同的哈希值
    Symbol sym1_dup = Symbol::FromQualString("prim::add");
    EXPECT_EQ(hasher(sym1), hasher(sym1_dup));

    // 测试哈希值与SymId的哈希值一致
    EXPECT_EQ(hash_val, std::hash<SymId>()(static_cast<SymId>(sym1)));
}

// 测试特殊字符和边缘情况
TEST(SymbolTest, EdgeCases) {
    // 测试包含特殊字符的字符串
    Symbol sym1 = Symbol::FromQualString("prim::add.Tensor");
    EXPECT_NE(static_cast<SymId>(sym1), 0);

    // 测试包含数字的字符串
    Symbol sym2 = Symbol::FromQualString("prim::op123");
    EXPECT_NE(static_cast<SymId>(sym2), 0);

    // 测试较长的字符串
    String long_str = "prim::";
    long_str.append(100, 'a');
    Symbol sym3 = Symbol::FromQualString(long_str);
    EXPECT_NE(static_cast<SymId>(sym3), 0);
}

// 测试GetDomainPrefix函数
TEST(SymbolTest, GetDomainPrefix) {
    const String& prefix = GetDomainPrefix();
    EXPECT_FALSE(prefix.empty());
    EXPECT_TRUE(prefix.find("aethermind") != String::npos);
}

}// namespace

#endif
