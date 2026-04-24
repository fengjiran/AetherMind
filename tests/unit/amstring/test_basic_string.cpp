#include <gtest/gtest.h>

#include <type_traits>

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

}// namespace
