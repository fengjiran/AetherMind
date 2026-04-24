#include <gtest/gtest.h>

#include <type_traits>

namespace {

template<typename CharT>
class BasicStringCoreSkeletonTest : public ::testing::Test {};

using AmstringCoreCharTypes = ::testing::Types<char, char8_t, char16_t, char32_t, wchar_t>;
TYPED_TEST_SUITE(BasicStringCoreSkeletonTest, AmstringCoreCharTypes);

TYPED_TEST(BasicStringCoreSkeletonTest, CharTMatrixIsAvailable) {
    using CharT = TypeParam;

    static_assert(std::is_trivial_v<CharT>);
    static_assert(std::is_standard_layout_v<CharT>);

    SUCCEED() << "M0 skeleton: BasicStringCore tests are parameterized by CharT.";
}

}// namespace
