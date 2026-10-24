#include "utils/parse_number.h"
#include "utils/parse_number_internal.h"

#include "inference/test_malloc_interposer.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

template<typename T>
using Parser = bool (*)(std::string_view, T&) noexcept;

template<typename T>
auto Parsers() {
    constexpr Parser<T> public_parser = [] {
        if constexpr (std::is_same_v<T, float>) {
            return aethermind::utils::ParseFloat;
        } else {
            return aethermind::utils::ParseDouble;
        }
    }();
#if defined(__APPLE__) || defined(__linux__)
    constexpr Parser<T> fallback_parser = [] {
        if constexpr (std::is_same_v<T, float>) {
            return aethermind::utils::detail::ParseFloatFallback;
        } else {
            return aethermind::utils::detail::ParseDoubleFallback;
        }
    }();
    return std::array{public_parser, fallback_parser};
#else
    return std::array{public_parser};
#endif
}

template<typename T>
class ParseNumber : public ::testing::Test {};

using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ParseNumber, FloatingTypes);

TYPED_TEST(ParseNumber, AcceptsDecimalAndExponentForms) {
    struct Case {
        std::string_view input;
        TypeParam expected;
    };
    constexpr std::array cases{
            Case{"0", 0}, Case{"42", 42}, Case{"-42", -42},
            Case{"1.", 1}, Case{".5", TypeParam{0.5}},
            Case{"1.25", TypeParam{1.25}}, Case{"-1.25", TypeParam{-1.25}},
            Case{"1e2", 100}, Case{"1.e+2", 100}, Case{"1E-2", TypeParam{0.01}}};
    for (const auto parser: Parsers<TypeParam>()) {
        for (const auto& [input, expected]: cases) {
            SCOPED_TRACE(input);
            TypeParam result = 17;
            EXPECT_TRUE(parser(input, result));
            EXPECT_EQ(result, expected);
        }
    }
}

TYPED_TEST(ParseNumber, RejectsMalformedLiteralsWithoutPublishing) {
    constexpr std::array<std::string_view, 23> cases{
            "", "+1", " 1", "\t1", "1 ", "1\n", "-", "+", ".", "-.",
            "1e", "1e+", "1e-", "1..0", "--1", "0x1p2", "-0X1P2",
            "1junk", "infinite", "nan(", "nan(a-b)", "nan(a b)", "nan(a))"};
    for (const auto parser: Parsers<TypeParam>()) {
        for (const std::string_view input: cases) {
            SCOPED_TRACE(input);
            TypeParam result = 17;
            EXPECT_FALSE(parser(input, result));
            EXPECT_EQ(result, 17);
        }
    }
}

TYPED_TEST(ParseNumber, RejectsDefaultEmptyViewWithoutPublishing) {
    for (const auto parser: Parsers<TypeParam>()) {
        TypeParam result = 17;
        EXPECT_FALSE(parser(std::string_view{}, result));
        EXPECT_EQ(result, 17);
    }
}

TYPED_TEST(ParseNumber, RejectsEmbeddedNullWithoutPublishing) {
    constexpr char kInput[] = {'1', '\0', '2'};
    for (const auto parser: Parsers<TypeParam>()) {
        TypeParam result = 17;
        EXPECT_FALSE(parser(std::string_view(kInput, sizeof(kInput)), result));
        EXPECT_EQ(result, 17);
    }
}

TYPED_TEST(ParseNumber, PreservesNegativeZero) {
    for (const auto parser: Parsers<TypeParam>()) {
        TypeParam result = 17;
        EXPECT_TRUE(parser("-0.0e10", result));
        EXPECT_EQ(result, 0);
        EXPECT_TRUE(std::signbit(result));
    }
}

TYPED_TEST(ParseNumber, AcceptsInfinityAndNanLiterals) {
    for (const auto parser: Parsers<TypeParam>()) {
        for (const std::string_view input: {"inf", "Infinity", "INF", "-infinity"}) {
            SCOPED_TRACE(input);
            TypeParam result = 17;
            EXPECT_TRUE(parser(input, result));
            EXPECT_TRUE(std::isinf(result));
            EXPECT_EQ(std::signbit(result), input.front() == '-');
        }
        for (const std::string_view input: {"nan", "NaN", "-NAN", "nan()", "nan(a_b0)"}) {
            SCOPED_TRACE(input);
            TypeParam result = 17;
            EXPECT_TRUE(parser(input, result));
            EXPECT_TRUE(std::isnan(result));
        }
    }
}

TYPED_TEST(ParseNumber, RejectsNonfiniteTrailingGarbageWithoutPublishing) {
    for (const auto parser: Parsers<TypeParam>()) {
        for (const std::string_view input: {"inf!", "nan(foo)tail", "+inf", "+nan"}) {
            SCOPED_TRACE(input);
            TypeParam result = 17;
            EXPECT_FALSE(parser(input, result));
            EXPECT_EQ(result, 17);
        }
    }
}

TYPED_TEST(ParseNumber, AcceptsMaximumFiniteValue) {
    constexpr std::string_view kInput = std::is_same_v<TypeParam, float>
                                                ? "3.4028234663852885981170418348451692544e38"
                                                : "1.797693134862315708145274237317043567981e308";
    for (const auto parser: Parsers<TypeParam>()) {
        TypeParam result = 17;
        EXPECT_TRUE(parser(kInput, result));
        EXPECT_EQ(result, std::numeric_limits<TypeParam>::max());
    }
}

TYPED_TEST(ParseNumber, RejectsOverflowWithoutPublishing) {
    constexpr std::string_view kInput = std::is_same_v<TypeParam, float> ? "1e39" : "1e309";
    for (const auto parser: Parsers<TypeParam>()) {
        TypeParam result = 17;
        EXPECT_FALSE(parser(kInput, result));
        EXPECT_EQ(result, 17);
    }
}

TYPED_TEST(ParseNumber, AcceptsRepresentableSubnormal) {
    constexpr std::string_view kInput = std::is_same_v<TypeParam, float>
                                                ? "1.401298464324817070923729583289916131280e-45"
                                                : "4.940656458412465441765687928682213723651e-324";
    for (const auto parser: Parsers<TypeParam>()) {
        TypeParam result = 17;
        EXPECT_TRUE(parser(kInput, result));
        EXPECT_EQ(result, std::numeric_limits<TypeParam>::denorm_min());
    }
}

TYPED_TEST(ParseNumber, RejectsUnderflowToZeroWithoutPublishing) {
    for (const auto parser: Parsers<TypeParam>()) {
        TypeParam result = 17;
        EXPECT_FALSE(parser("1e-99999", result));
        EXPECT_EQ(result, 17);
    }
}

TYPED_TEST(ParseNumber, AcceptsZeroWithExtremeExponent) {
    for (const auto parser: Parsers<TypeParam>()) {
        for (const std::string_view input: {"0e-99999", "0e99999"}) {
            SCOPED_TRACE(input);
            TypeParam result = 17;
            EXPECT_TRUE(parser(input, result));
            EXPECT_EQ(result, 0);
        }
    }
}

#if defined(__APPLE__) || defined(__linux__)
class SavedFloatingEnvironment {
public:
    SavedFloatingEnvironment() : captured_(std::fegetenv(&environment_) == 0) {}
    ~SavedFloatingEnvironment() {
        if (captured_) {
            std::fesetenv(&environment_);
        }
    }

    bool captured() const { return captured_; }

private:
    std::fenv_t environment_;
    bool captured_;
};

TYPED_TEST(ParseNumber, FallbackRestoresRoundingModeAfterSuccessAndFailure) {
    const SavedFloatingEnvironment environment;
    ASSERT_TRUE(environment.captured());
    const auto parser = Parsers<TypeParam>().back();
    constexpr std::string_view kOverflow = std::is_same_v<TypeParam, float> ? "1e39" : "1e309";
    for (const int mode: {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        ASSERT_EQ(std::feraiseexcept(FE_INVALID), 0);
        TypeParam result = 17;
        EXPECT_TRUE(parser("0.1", result));
        EXPECT_EQ(result, static_cast<TypeParam>(0.1));
        EXPECT_EQ(std::fegetround(), mode);
        EXPECT_NE(std::fetestexcept(FE_INVALID), 0);

        for (const std::string_view input: {kOverflow, std::string_view{"1e-99999"}}) {
            result = 17;
            EXPECT_FALSE(parser(input, result));
            EXPECT_EQ(result, 17);
            EXPECT_EQ(std::fegetround(), mode);
            EXPECT_NE(std::fetestexcept(FE_INVALID), 0);
        }
    }
}

TYPED_TEST(ParseNumber, FallbackAllocationFailurePreservesOutputAndCanRetry) {
    namespace injection = aethermind::test;
    if (!injection::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    const SavedFloatingEnvironment environment;
    ASSERT_TRUE(environment.captured());
    const std::string input = "1." + std::string(4096, '0');
    const auto parser = Parsers<TypeParam>().back();
    for (const int mode: {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        TypeParam result = 17;
        injection::ScopedMallocFailure failure(input.size() + 1);
        const bool parsed = parser(input, result);
        const auto injected = failure.Stop();

        EXPECT_TRUE(injected.allocation_failed);
        EXPECT_FALSE(parsed);
        EXPECT_EQ(result, 17);
        EXPECT_EQ(std::fegetround(), mode);
        EXPECT_TRUE(parser(input, result));
        EXPECT_EQ(result, 1);
        EXPECT_EQ(std::fegetround(), mode);
    }
}
#endif

} // namespace
