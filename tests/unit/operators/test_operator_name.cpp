//
// Created by 赵丹 on 2025/8/12.
//

#include "operator_name.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>

using namespace aethermind;

namespace {

// === Basic Construction Tests ===

TEST(Operators_OperatorName, DefaultConstructor) {
    OperatorName op;
    EXPECT_EQ(op.name(), "");
    EXPECT_EQ(op.overload_name(), "");
    EXPECT_FALSE(op.GetNamespace().has_value());
    EXPECT_EQ(ToString(op), "");
}

TEST(Operators_OperatorName, BasicConstruction) {
    OperatorName op("aethermind::add", "Tensor");
    EXPECT_EQ(op.name(), "aethermind::add");
    EXPECT_EQ(op.overload_name(), "Tensor");
}

TEST(Operators_OperatorName, EmptyOverloadName) {
    OperatorName op("aethermind::add", "");
    EXPECT_EQ(op.name(), "aethermind::add");
    EXPECT_EQ(op.overload_name(), "");
    EXPECT_EQ(ToString(op), "aethermind::add");
}

TEST(Operators_OperatorName, EmptyNames) {
    OperatorName op("", "");
    EXPECT_EQ(op.name(), "");
    EXPECT_EQ(op.overload_name(), "");
    EXPECT_FALSE(op.GetNamespace().has_value());
}

// === GetNamespace Tests ===

TEST(Operators_OperatorName, GetNamespace_SingleNamespace) {
    OperatorName op("aethermind::add", "Tensor");
    auto ns = op.GetNamespace();
    ASSERT_TRUE(ns.has_value());
    EXPECT_EQ(ns.value(), "aethermind");
}

TEST(Operators_OperatorName, GetNamespace_MultiLevelNamespace) {
    OperatorName op("aethermind::nn::linear", "Tensor");
    auto ns = op.GetNamespace();
    ASSERT_TRUE(ns.has_value());
    EXPECT_EQ(ns.value(), "aethermind");// Only returns first namespace
}

TEST(Operators_OperatorName, GetNamespace_NoNamespace) {
    OperatorName op("add", "Tensor");
    EXPECT_FALSE(op.GetNamespace().has_value());
}

TEST(Operators_OperatorName, GetNamespace_EmptyName) {
    OperatorName op("", "Tensor");
    EXPECT_FALSE(op.GetNamespace().has_value());
}

TEST(Operators_OperatorName, GetNamespace_OnlyDoubleColon) {
    OperatorName op("::", "Tensor");
    auto ns = op.GetNamespace();
    ASSERT_TRUE(ns.has_value());
    EXPECT_EQ(ns.value(), "");// Namespace before :: is empty
}

TEST(Operators_OperatorName, GetNamespace_TrailingDoubleColon) {
    OperatorName op("add::", "Tensor");
    auto ns = op.GetNamespace();
    ASSERT_TRUE(ns.has_value());
    EXPECT_EQ(ns.value(), "add");
}

TEST(Operators_OperatorName, GetNamespace_ViewLifetime) {
    OperatorName op("aethermind::add", "Tensor");
    std::string_view name_copy = op.name();
    auto ns = op.GetNamespace();
    ASSERT_TRUE(ns.has_value());
    // Both views should be valid while op exists
    EXPECT_EQ(ns.value(), "aethermind");
    EXPECT_EQ(name_copy, "aethermind::add");
}

// === ToString Tests ===

TEST(Operators_OperatorName, ToString_WithOverload) {
    OperatorName op("aethermind::add", "Tensor");
    EXPECT_EQ(ToString(op), "aethermind::add.Tensor");
}

TEST(Operators_OperatorName, ToString_WithoutOverload) {
    OperatorName op("aethermind::add", "");
    EXPECT_EQ(ToString(op), "aethermind::add");
}

TEST(Operators_OperatorName, ToString_Empty) {
    OperatorName op("", "");
    EXPECT_EQ(ToString(op), "");
}

TEST(Operators_OperatorName, ToString_EmptyNameWithOverload) {
    OperatorName op("", "Tensor");
    EXPECT_EQ(ToString(op), ".Tensor");
}

TEST(Operators_OperatorName, OstreamOperator) {
    OperatorName op("aethermind::add", "Tensor");
    std::ostringstream oss;
    oss << op;
    EXPECT_EQ(oss.str(), "aethermind::add.Tensor");
}

// === Comparison Tests ===

TEST(Operators_OperatorName, Equality) {
    OperatorName op1("aethermind::add", "Tensor");
    OperatorName op2("aethermind::add", "Tensor");
    OperatorName op3("aethermind::add", "Scalar");
    OperatorName op4("aethermind::mul", "Tensor");

    EXPECT_TRUE(op1 == op2);
    EXPECT_FALSE(op1 == op3);
    EXPECT_FALSE(op1 == op4);
    EXPECT_TRUE(op1 != op3);
    EXPECT_TRUE(op1 != op4);
    EXPECT_FALSE(op1 != op2);
}

TEST(Operators_OperatorName, EqualityEmpty) {
    OperatorName op1("", "");
    OperatorName op2("", "");
    OperatorName op3("", "Tensor");

    EXPECT_TRUE(op1 == op2);
    EXPECT_FALSE(op1 == op3);
}

// === Ordering Tests ===

TEST(Operators_OperatorName, LessThan_NameOnly) {
    OperatorName op1("a", "");
    OperatorName op2("b", "");

    EXPECT_TRUE(op1 < op2);
    EXPECT_FALSE(op2 < op1);
    EXPECT_FALSE(op1 < op1);
}

TEST(Operators_OperatorName, LessThan_OverloadOnly) {
    OperatorName op1("add", "A");
    OperatorName op2("add", "B");

    EXPECT_TRUE(op1 < op2);
    EXPECT_FALSE(op2 < op1);
}

TEST(Operators_OperatorName, LessThan_BothDifferent) {
    OperatorName op1("a", "B");
    OperatorName op2("b", "A");

    // Name comparison takes precedence
    EXPECT_TRUE(op1 < op2);
    EXPECT_FALSE(op2 < op1);
}

TEST(Operators_OperatorName, LessThan_EmptyHandling) {
    OperatorName op1("", "");
    OperatorName op2("a", "");

    EXPECT_TRUE(op1 < op2);// Empty < non-empty
    EXPECT_FALSE(op2 < op1);
}

TEST(Operators_OperatorName, Sorting) {
    std::vector<OperatorName> ops = {
            OperatorName("c", "B"),
            OperatorName("a", "B"),
            OperatorName("b", "A"),
            OperatorName("a", "A"),
            OperatorName("b", "B"),
    };

    std::sort(ops.begin(), ops.end());

    std::vector<OperatorName> expected = {
            OperatorName("a", "A"),
            OperatorName("a", "B"),
            OperatorName("b", "A"),
            OperatorName("b", "B"),
            OperatorName("c", "B"),
    };

    ASSERT_EQ(ops.size(), expected.size());
    for (size_t i = 0; i < ops.size(); ++i) {
        EXPECT_TRUE(ops[i] == expected[i]) << "Mismatch at index " << i;
    }
}

TEST(Operators_OperatorName, SetUsage) {
    std::set<OperatorName> ops;
    ops.emplace("aethermind::add", "Tensor");
    ops.emplace("aethermind::add", "Scalar");
    ops.emplace("aethermind::mul", "Tensor");
    ops.emplace("aethermind::add", "Tensor");// Duplicate

    EXPECT_EQ(ops.size(), 3);// Duplicate not inserted

    // Verify ordering
    auto it = ops.begin();
    EXPECT_EQ(it->name(), "aethermind::add");
    EXPECT_EQ(it->overload_name(), "Scalar");// A < T
    ++it;
    EXPECT_EQ(it->name(), "aethermind::add");
    EXPECT_EQ(it->overload_name(), "Tensor");
    ++it;
    EXPECT_EQ(it->name(), "aethermind::mul");
}

// === Hash Tests ===

TEST(Operators_OperatorName, Hash_DifferentNamesDifferentHash) {
    OperatorName op1("a", "x");
    OperatorName op2("b", "x");

    auto h1 = std::hash<OperatorName>()(op1);
    auto h2 = std::hash<OperatorName>()(op2);

    EXPECT_NE(h1, h2);
}

TEST(Operators_OperatorName, Hash_DifferentOverloadsDifferentHash) {
    OperatorName op1("a", "x");
    OperatorName op2("a", "y");

    auto h1 = std::hash<OperatorName>()(op1);
    auto h2 = std::hash<OperatorName>()(op2);

    EXPECT_NE(h1, h2);
}

TEST(Operators_OperatorName, Hash_SameOperatorSameHash) {
    OperatorName op1("aethermind::add", "Tensor");
    OperatorName op2("aethermind::add", "Tensor");

    EXPECT_EQ(std::hash<OperatorName>()(op1), std::hash<OperatorName>()(op2));
}

TEST(Operators_OperatorName, Hash_SwappedNamesLikelyDifferent) {
    // This test verifies that hash is not simply XOR which would cause collision
    // for swapped values: hash(a,x) XOR ~hash(x,a) could equal hash(x,a) XOR ~hash(a,x)
    OperatorName op1("alpha", "beta");
    OperatorName op2("beta", "alpha");

    auto h1 = std::hash<OperatorName>()(op1);
    auto h2 = std::hash<OperatorName>()(op2);

    // With get_hash (hash_combine), these should be different
    EXPECT_NE(h1, h2) << "Hash collision for swapped name/overload_name";
}

TEST(Operators_OperatorName, Hash_Consistency) {
    // Hash should be consistent across multiple calls
    OperatorName op("test::op", "Tensor");
    auto h1 = std::hash<OperatorName>()(op);
    auto h2 = std::hash<OperatorName>()(op);
    auto h3 = std::hash<OperatorName>()(op);

    EXPECT_EQ(h1, h2);
    EXPECT_EQ(h2, h3);
}

TEST(Operators_OperatorName, UnorderedMapUsage) {
    std::unordered_map<OperatorName, int> map;

    map[OperatorName("aethermind::add", "Tensor")] = 1;
    map[OperatorName("aethermind::mul", "Tensor")] = 2;
    map[OperatorName("aethermind::add", "Scalar")] = 3;

    EXPECT_EQ(map.size(), 3);

    // Lookup should work
    auto it = map.find(OperatorName("aethermind::add", "Tensor"));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 1);

    // Non-existent key should not be found
    it = map.find(OperatorName("aethermind::div", "Tensor"));
    EXPECT_EQ(it, map.end());
}

TEST(Operators_OperatorName, UnorderedSetUsage) {
    std::unordered_set<OperatorName> set;

    set.emplace("aethermind::add", "Tensor");
    set.emplace("aethermind::mul", "Tensor");
    set.emplace("aethermind::add", "Tensor");// Duplicate

    EXPECT_EQ(set.size(), 2);

    EXPECT_TRUE(set.contains(OperatorName("aethermind::add", "Tensor")));
    EXPECT_TRUE(set.contains(OperatorName("aethermind::mul", "Tensor")));
    EXPECT_FALSE(set.contains(OperatorName("aethermind::div", "Tensor")));
}

// === String Conversion Tests ===

TEST(Operators_OperatorName, StringViewConversion) {
    OperatorName op("test", "Tensor");

    std::string_view name_view = op.name();
    std::string_view overload_view = op.overload_name();

    EXPECT_EQ(name_view, "test");
    EXPECT_EQ(overload_view, "Tensor");

    // Can create std::string from view
    std::string name_copy(name_view);
    EXPECT_EQ(name_copy, "test");
}

TEST(Operators_OperatorName, ImplicitStringViewConversion) {
    OperatorName op("test::op", "Tensor");

    // Should work with functions expecting string_view
    auto check_prefix = [](std::string_view sv, std::string_view prefix) {
        return sv.starts_with(prefix);
    };

    EXPECT_TRUE(check_prefix(op.name(), "test"));
    EXPECT_FALSE(check_prefix(op.name(), "other"));
}

// === Edge Cases ===

TEST(Operators_OperatorName, SpecialCharactersInName) {
    OperatorName op("namespace::operator_with_underscore", "Type<int>");
    EXPECT_EQ(op.name(), "namespace::operator_with_underscore");
    EXPECT_EQ(op.overload_name(), "Type<int>");
}

TEST(Operators_OperatorName, UnicodeInName) {
    // UTF-8 encoded Chinese characters
    OperatorName op("命名空间::操作", "类型");
    EXPECT_EQ(op.name(), "命名空间::操作");
    EXPECT_EQ(op.overload_name(), "类型");
    auto ns = op.GetNamespace();
    ASSERT_TRUE(ns.has_value());
    EXPECT_EQ(ns.value(), "命名空间");
}

TEST(Operators_OperatorName, LongNames) {
    std::string long_name(1000, 'a');
    long_name.insert(500, "::");
    OperatorName op(long_name, "Tensor");

    EXPECT_EQ(op.name().size(), 1002);
    auto ns = op.GetNamespace();
    ASSERT_TRUE(ns.has_value());
    EXPECT_EQ(ns.value().size(), 500);
}

// === Copy/Move Tests ===

TEST(Operators_OperatorName, CopyConstruction) {
    OperatorName op1("test::op", "Tensor");
    OperatorName op2(op1);

    EXPECT_EQ(op1.name(), op2.name());
    EXPECT_EQ(op1.overload_name(), op2.overload_name());
    EXPECT_TRUE(op1 == op2);
}

TEST(Operators_OperatorName, MoveConstruction) {
    OperatorName op1("test::op", "Tensor");
    OperatorName op2(std::move(op1));

    EXPECT_EQ(op2.name(), "test::op");
    EXPECT_EQ(op2.overload_name(), "Tensor");
}

TEST(Operators_OperatorName, Assignment) {
    OperatorName op1("test::op", "Tensor");
    OperatorName op2;

    op2 = op1;
    EXPECT_TRUE(op2 == op1);
}

}// namespace