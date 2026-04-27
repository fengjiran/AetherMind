#include <gtest/gtest.h>

#include "amstring/basic_string.hpp"

#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

template<typename CharT>
class BasicStringSkeletonTest : public ::testing::Test {};

using AmstringPublicCharTypes = ::testing::Types<char, char8_t, char16_t, char32_t, wchar_t>;
TYPED_TEST_SUITE(BasicStringSkeletonTest, AmstringPublicCharTypes);

TYPED_TEST(BasicStringSkeletonTest, CharTMatrixIsAvailable) {
    using CharT = TypeParam;

    static_assert(std::is_trivial_v<CharT>);
    static_assert(std::is_standard_layout_v<CharT>);

    SUCCEED() << "M0 skeleton: BasicString tests are parameterized by CharT.";
}

TYPED_TEST(BasicStringSkeletonTest, GetAllocatorIsAvailable) {
    using CharT = TypeParam;
    using String = aethermind::BasicString<CharT>;

    String string;
    const auto allocator = string.get_allocator();

    static_assert(std::is_same_v<decltype(allocator), const std::allocator<CharT>>);
}

TYPED_TEST(BasicStringSkeletonTest, SwapNoexceptFollowsCore) {
    using CharT = TypeParam;
    using String = aethermind::BasicString<CharT>;

    static_assert(noexcept(std::declval<String&>().swap(std::declval<String&>())));
}

TYPED_TEST(BasicStringSkeletonTest, AssignOverloadsAreAvailable) {
    using CharT = TypeParam;
    using String = aethermind::BasicString<CharT>;

    const CharT data[] = {static_cast<CharT>('a'), static_cast<CharT>('b'), CharT{}};
    String string;

    string.assign(data, 2);
    EXPECT_EQ(string.size(), 2U);

    string.assign(std::basic_string_view<CharT>(data, 1));
    EXPECT_EQ(string.size(), 1U);

    string.assign(3, static_cast<CharT>('z'));
    EXPECT_EQ(string.size(), 3U);
    EXPECT_EQ(string.data()[3], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, PushBackAndPopBackAreAvailable) {
    using CharT = TypeParam;
    using String = aethermind::BasicString<CharT>;

    String string;
    const CharT ch = static_cast<CharT>('x');

    string.push_back(ch);
    EXPECT_EQ(string.size(), 1U);
    EXPECT_EQ(string.data()[0], ch);
    EXPECT_EQ(string.data()[1], CharT{});

    string.pop_back();
    EXPECT_TRUE(string.empty());
    EXPECT_EQ(string.data()[0], CharT{});
}

}// namespace
