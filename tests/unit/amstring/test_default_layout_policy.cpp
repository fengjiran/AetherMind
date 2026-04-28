#include <gtest/gtest.h>

#include "amstring/basic_string.hpp"
#include "amstring/char_layout_policy.hpp"
#include "amstring/core.hpp"
#include "amstring/generic_layout_policy.hpp"
#include "amstring/layout_policy.hpp"

#include <type_traits>

namespace {

using namespace aethermind;

template<typename T>
concept HasPublicLayoutPolicyType = requires {
    typename T::LayoutPolicyType;
};

template<typename CharT>
class DefaultLayoutPolicyTest : public ::testing::Test {};

template<typename CharT>
struct ExpectedDefaultLayoutPolicy {
    using type = GenericLayoutPolicy<CharT>;
};

template<>
struct ExpectedDefaultLayoutPolicy<char> {
    using type = CharLayoutPolicy;
};

using AmstringCharTypes = ::testing::Types<char, char8_t, char16_t, char32_t, wchar_t>;
TYPED_TEST_SUITE(DefaultLayoutPolicyTest, AmstringCharTypes);

TYPED_TEST(DefaultLayoutPolicyTest, SelectsExpectedLayoutPolicy) {
    using CharT = TypeParam;
    using Selected = DefaultLayoutPolicy<CharT>::type;
    using Expected = ExpectedDefaultLayoutPolicy<CharT>::type;

    static_assert(std::is_same_v<Selected, Expected>);
}

TYPED_TEST(DefaultLayoutPolicyTest, SelectedPolicySatisfiesLayoutConcept) {
    using CharT = TypeParam;
    using Selected = DefaultLayoutPolicy<CharT>::type;

    static_assert(AmStringLayoutPolicy<Selected, CharT>);
}

TYPED_TEST(DefaultLayoutPolicyTest, CoreUsesSelectorByDefault) {
    using CharT = TypeParam;
    using Core = BasicStringCore<CharT>;
    using Expected = DefaultLayoutPolicy<CharT>::type;

    static_assert(std::is_same_v<typename Core::LayoutPolicyType, Expected>);
}

TYPED_TEST(DefaultLayoutPolicyTest, CoreDefaultConstructsWithSelectedLayout) {
    using CharT = TypeParam;
    using Core = BasicStringCore<CharT>;

    Core core;

    EXPECT_TRUE(core.empty());
    EXPECT_EQ(core.size(), 0U);
    EXPECT_EQ(core.data()[0], CharT{});
}

TYPED_TEST(DefaultLayoutPolicyTest, PublicStringDoesNotExposeLayoutPolicyType) {
    using CharT = TypeParam;
    using String = BasicString<CharT>;

    static_assert(!HasPublicLayoutPolicyType<String>);
}

TYPED_TEST(DefaultLayoutPolicyTest, PublicStringDefaultConstructsWithHiddenLayout) {
    using CharT = TypeParam;
    using String = BasicString<CharT>;

    String string;

    EXPECT_TRUE(string.empty());
    EXPECT_EQ(string.size(), 0U);
    EXPECT_EQ(string.data()[0], CharT{});
}

}// namespace
