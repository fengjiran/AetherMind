#include "aethermind/backend/cpu/bf16_conversion.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/dtypes/bf16_conversion.h"
#include "aethermind/dtypes/bfloat16.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using namespace aethermind;

Bf16ConversionFuncs SelectedConversions() {
    static const auto capabilities = cpu::DetectCpuCapabilities();
    EXPECT_TRUE(capabilities.ok());
    return capabilities.ok() ? cpu::SelectBf16Conversions(capabilities->effective_features)
                             : Bf16ConversionFuncs{};
}

double PositiveBf16Value(uint16_t bits) {
    const unsigned exponent = bits >> 7;
    const unsigned fraction = bits & 0x007FU;
    return exponent == 0
                   ? std::ldexp(static_cast<double>(fraction), -133)
                   : std::ldexp(static_cast<double>(128 + fraction), static_cast<int>(exponent) - 134);
}

uint32_t WideningOracle(uint16_t bits) {
    const uint32_t magnitude = bits & 0x7FFFU;
    const uint32_t sign = (bits & 0x8000U) << 16;
    if (magnitude >= 0x7F80U) {
        return sign | (magnitude << 16) | (magnitude > 0x7F80U ? 0x00400000U : 0U);
    }

    const auto value = static_cast<float>(PositiveBf16Value(static_cast<uint16_t>(magnitude)));
    return sign | std::bit_cast<uint32_t>(value);
}

uint16_t NarrowingOracle(uint32_t bits) {
    const auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t magnitude = bits & 0x7FFFFFFFU;
    if (magnitude >= 0x7F800000U) {
        return sign | 0x7F80U | (magnitude > 0x7F800000U ? static_cast<uint16_t>((magnitude >> 16) & 0x007FU) | 0x0040U : 0U);
    }

    // Search representable values and compare exact double distances rather
    // than reproducing the implementation's integer rounding-bias formula.
    static const std::vector<double> values = [] {
        std::vector<double> result;
        result.reserve(0x7F81U);
        for (uint32_t code = 0; code < 0x7F80U; ++code) {
            result.push_back(PositiveBf16Value(static_cast<uint16_t>(code)));
        }
        // The next precision-8 value defines the overflow midpoint to infinity.
        result.push_back(std::ldexp(1.0, 128));
        return result;
    }();

    const auto value = static_cast<double>(std::bit_cast<float>(magnitude));
    const auto upper = std::ranges::lower_bound(values, value);
    if (upper == values.begin()) {
        return sign;
    }

    const auto upper_code = static_cast<uint16_t>(upper - values.begin());
    const double lower_distance = value - *(upper - 1);
    const double upper_distance = *upper - value;
    const uint16_t code = lower_distance < upper_distance || (lower_distance == upper_distance && (upper_code & 1U) != 0)
                                  ? upper_code - 1
                                  : upper_code;
    return sign | code;
}

TEST(Bf16BatchConversion, EveryBfloatPatternWidensExactlyAndQuietsNaNs) {
    std::vector<uint16_t> source(0x10000U);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint16_t>(index);
    }

    std::vector<uint32_t> actual(source.size());
    EXPECT_TRUE(ConvertBf16ToFp32(std::as_bytes(std::span(source)),
                                  std::as_writable_bytes(std::span(actual)))
                        .ok());
    for (size_t index = 0; index < source.size(); ++index) {
        EXPECT_EQ(actual[index], WideningOracle(source[index])) << "BF16 bits=" << index;
    }
}

TEST(Bf16BatchConversion, EveryPatternMatchesOracleInEveryLane) {
    std::vector<uint16_t> source(0x10000U + 15);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint16_t>(index);
    }

    std::vector<uint32_t> actual(0x10000U);
    const auto functions = SelectedConversions();
    for (size_t rotation = 0; rotation < 16; ++rotation) {
        const auto input = std::as_bytes(
                std::span(source).subspan(rotation, actual.size()));
        ASSERT_TRUE(ConvertBf16ToFp32(
                            input,
                            std::as_writable_bytes(std::span(actual)), functions)
                            .ok());
        for (size_t index = 0; index < actual.size(); ++index) {
            ASSERT_EQ(actual[index], WideningOracle(source[index + rotation]))
                    << "index=" << index << " rotation=" << rotation;
        }
    }
}

TEST(Bf16BatchConversion, SelectedConversionsMatchScalarForUnalignedRangesAndAllTails) {
    const auto functions = SelectedConversions();
    for (size_t count = 0; count <= 65; ++count) {
        for (size_t offset = 0; offset < 32; ++offset) {
            for (const bool widening: {false, true}) {
                const size_t input_width = widening ? 2 : 4;
                const size_t output_width = widening ? 4 : 2;
                const size_t destination_offset = (offset * 7 + 3) % 32;
                std::vector source(offset + count * input_width + 32, std::byte{0x5A});
                for (size_t index = 0; index < count * input_width; ++index) {
                    source[offset + index] = static_cast<std::byte>((index * 139 + offset * 17) & 0xFF);
                }
                const auto original = source;
                std::vector actual(destination_offset + count * output_width + 32, std::byte{0xCD});
                std::vector<std::byte> expected(count * output_width);
                const auto input = std::span<const std::byte>(source).subspan(offset, count * input_width);
                auto output = std::span(actual).subspan(destination_offset, count * output_width);
                if (widening) {
                    ASSERT_TRUE(ConvertBf16ToFp32(input, expected).ok());
                    ASSERT_TRUE(ConvertBf16ToFp32(input, output, functions).ok());
                } else {
                    ASSERT_TRUE(ConvertFp32ToBf16(input, expected).ok());
                    ASSERT_TRUE(ConvertFp32ToBf16(input, output, functions).ok());
                }

                EXPECT_TRUE(std::equal(output.begin(), output.end(), expected.begin()))
                        << "count=" << count << " offset=" << offset << " widening=" << widening;
                EXPECT_TRUE(std::all_of(actual.begin(), actual.begin() + destination_offset,
                                        [](std::byte value) { return value == std::byte{0xCD}; }));
                EXPECT_TRUE(std::all_of(actual.begin() + destination_offset + output.size(), actual.end(),
                                        [](std::byte value) { return value == std::byte{0xCD}; }));
                EXPECT_EQ(source, original);
            }
        }
    }
}

TEST(Bf16BatchConversion, SelectedNarrowingMatchesScalarForEveryFiniteMidpoint) {
    std::vector<uint32_t> source;
    for (uint32_t lower = 0; lower < 0x7F80U; ++lower) {
        const uint32_t midpoint = (lower << 16) + 0x8000U;
        for (const uint32_t sign: {0U, 0x80000000U}) {
            source.push_back(sign | (midpoint - 1));
            source.push_back(sign | midpoint);
            source.push_back(sign | (midpoint + 1));
        }
    }

    std::vector<uint16_t> actual(source.size()), expected(source.size());
    const auto input = std::as_bytes(std::span(source));
    ASSERT_TRUE(ConvertFp32ToBf16(input, std::as_writable_bytes(std::span(expected))).ok());
    ASSERT_TRUE(ConvertFp32ToBf16(input, std::as_writable_bytes(std::span(actual)), SelectedConversions()).ok());
    for (size_t index = 0; index < actual.size(); ++index) {
        ASSERT_EQ(actual[index], expected[index]) << "source bits=" << source[index];
    }
}

TEST(Bf16BatchConversion, SelectedNarrowingMatchesScalarForRandomBitPatterns) {
    std::mt19937 random(20261110);
    std::vector<uint32_t> source(200003);
    for (auto& bits: source) {
        bits = random();
    }
    std::vector<uint16_t> actual(source.size()), expected(source.size());
    const auto input = std::as_bytes(std::span(source));
    ASSERT_TRUE(ConvertFp32ToBf16(input, std::as_writable_bytes(std::span(expected))).ok());
    ASSERT_TRUE(ConvertFp32ToBf16(input, std::as_writable_bytes(std::span(actual)), SelectedConversions()).ok());
    for (size_t index = 0; index < actual.size(); ++index) {
        ASSERT_EQ(actual[index], expected[index]) << "source bits=" << source[index];
    }
}

TEST(Bf16BatchConversion, EveryFinitePatternRoundTripsWithoutRounding) {
    std::vector<uint16_t> source;
    for (uint32_t bits = 0; bits <= 0xFFFFU; ++bits) {
        if ((bits & 0x7F80U) != 0x7F80U) {
            source.push_back(static_cast<uint16_t>(bits));
        }
    }
    std::vector<uint32_t> widened(source.size());
    std::vector<uint16_t> round_trip(source.size());
    EXPECT_TRUE(ConvertBf16ToFp32(std::as_bytes(std::span(source)),
                                  std::as_writable_bytes(std::span(widened)))
                        .ok());
    EXPECT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(widened)),
                                  std::as_writable_bytes(std::span(round_trip)))
                        .ok());
    EXPECT_EQ(round_trip, source);
}

TEST(Bf16BatchConversion, EveryFiniteMidpointAndItsNeighborsUseNearestEven) {
    std::vector<uint32_t> source;
    std::vector<uint16_t> expected;
    for (uint32_t lower = 0; lower < 0x7F80U; ++lower) {
        const uint32_t midpoint = (lower << 16) + 0x8000U;
        for (const uint32_t sign: {0U, 0x80000000U}) {
            source.push_back(sign | (midpoint - 1));
            source.push_back(sign | midpoint);
            source.push_back(sign | (midpoint + 1));
            const auto result_sign = static_cast<uint16_t>(sign >> 16);
            expected.push_back(result_sign | static_cast<uint16_t>(lower));
            expected.push_back(result_sign | static_cast<uint16_t>(lower + (lower & 1U)));
            expected.push_back(result_sign | static_cast<uint16_t>(lower + 1));
        }
    }
    std::vector<uint16_t> actual(source.size());
    EXPECT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(source)),
                                  std::as_writable_bytes(std::span(actual)))
                        .ok());
    for (size_t index = 0; index < source.size(); ++index) {
        EXPECT_EQ(actual[index], expected[index]) << "FP32 bits=" << source[index];
    }
}

TEST(Bf16BatchConversion, RandomFloatPatternsMatchIndependentNearestValueOracle) {
    std::mt19937 random(20261110);
    std::vector<uint32_t> source(200003);
    for (auto& bits: source) {
        bits = random();
    }
    std::vector<uint16_t> actual(source.size());
    EXPECT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(source)),
                                  std::as_writable_bytes(std::span(actual)))
                        .ok());
    for (size_t index = 0; index < source.size(); ++index) {
        EXPECT_EQ(actual[index], NarrowingOracle(source[index]))
                << "FP32 bits=" << source[index];
    }
}

TEST(Bf16BatchConversion, QuietNaNsPreserveSignAndRepresentablePayload) {
    constexpr std::array<uint32_t, 10> source{
            0x7F800001U, 0xFF800001U, 0x7F810000U, 0xFF810001U, 0x7FA50001U,
            0xFFA5FFFFU, 0x7FC00000U, 0xFFC50001U, 0x7FFFFFFFU, 0xFFFFFFFFU};
    constexpr std::array<uint16_t, source.size()> expected{
            0x7FC0, 0xFFC0, 0x7FC1, 0xFFC1, 0x7FE5, 0xFFE5, 0x7FC0, 0xFFC5, 0x7FFF, 0xFFFF};
    std::array<uint16_t, source.size()> actual{};
    EXPECT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(source)),
                                  std::as_writable_bytes(std::span(actual)))
                        .ok());
    EXPECT_EQ(actual, expected);
}

TEST(Bf16BatchConversion, BatchAndWrapperWideningQuietRawSignalingNaNsIdentically) {
    constexpr std::array<uint16_t, 2> source{0x7F85U, 0xFF85U};
    std::array<uint32_t, source.size()> actual{};
    EXPECT_TRUE(ConvertBf16ToFp32(std::as_bytes(std::span(source)),
                                  std::as_writable_bytes(std::span(actual)))
                        .ok());

    for (size_t index = 0; index < source.size(); ++index) {
        const BFloat16 wrapper{source[index], BFloat16::from_bits()};
        const auto wrapper_bits = std::bit_cast<uint32_t>(static_cast<float>(wrapper));
        // Batch and single-value widening agree: payload preserved, quiet bit forced.
        EXPECT_EQ(actual[index], (static_cast<uint32_t>(source[index]) << 16) | 0x00400000U);
        EXPECT_EQ(wrapper_bits, actual[index]);
        EXPECT_EQ(wrapper.x, source[index]);
    }
}

TEST(Bf16BatchConversion, UnalignedRangesAndAllTailsPreserveSurroundingBytes) {
    for (size_t count = 0; count <= 65; ++count) {
        for (size_t offset = 0; offset < 32; ++offset) {
            for (const bool widening: {false, true}) {
                const size_t input_width = widening ? 2 : 4;
                const size_t output_width = widening ? 4 : 2;
                const size_t destination_offset = (offset * 7 + 3) % 32;
                std::vector source(offset + count * input_width + 32, std::byte{0x5A});
                std::vector actual(destination_offset + count * output_width + 32, std::byte{0xCD});
                std::vector<std::byte> expected(count * output_width);
                for (size_t index = 0; index < count; ++index) {
                    if (widening) {
                        const auto bits = static_cast<uint16_t>((index * 259 + offset * 179) & 0xFFFFU);
                        const uint32_t result = WideningOracle(bits);
                        std::memcpy(source.data() + offset + index * input_width, &bits, sizeof(bits));
                        std::memcpy(expected.data() + index * output_width, &result, sizeof(result));
                    } else {
                        const auto bits = static_cast<uint32_t>(index * 1442695041U + offset * 1664525U);
                        const uint16_t result = NarrowingOracle(bits);
                        std::memcpy(source.data() + offset + index * input_width, &bits, sizeof(bits));
                        std::memcpy(expected.data() + index * output_width, &result, sizeof(result));
                    }
                }

                const auto original = source;
                const auto input = std::span<const std::byte>(source).subspan(offset, count * input_width);
                auto output = std::span(actual).subspan(destination_offset, count * output_width);
                EXPECT_TRUE((widening ? ConvertBf16ToFp32(input, output) : ConvertFp32ToBf16(input, output)).ok());
                EXPECT_TRUE(std::equal(output.begin(), output.end(), expected.begin()))
                        << "count=" << count << " offset=" << offset;
                EXPECT_TRUE(std::all_of(actual.begin(), actual.begin() + destination_offset,
                                        [](std::byte byte) {
                                            return byte == std::byte{0xCD};
                                        }));
                EXPECT_TRUE(std::all_of(actual.begin() + destination_offset + output.size(), actual.end(),
                                        [](std::byte byte) {
                                            return byte == std::byte{0xCD};
                                        }));
                EXPECT_EQ(source, original);
            }
        }
    }
}

TEST(Bf16BatchConversion, EmptyRangesPermitNullPointersAndFunctions) {
    constexpr Bf16ConversionFuncs missing{nullptr, nullptr};
    EXPECT_TRUE(ConvertBf16ToFp32({}, {}, missing).ok());
    EXPECT_TRUE(ConvertFp32ToBf16({}, {}, missing).ok());
    ConvertBf16ToFp32Scalar(nullptr, nullptr, 0);
    ConvertFp32ToBf16Scalar(nullptr, nullptr, 0);
}

TEST(Bf16BatchConversion, InvalidRangesAreRejectedBeforeWriting) {
    std::array<std::byte, 64> bytes{};
    bytes.fill(std::byte{0x5A});
    const auto original = bytes;
    const auto src = std::span<const std::byte>(bytes);
    auto dst = std::span(bytes);
    EXPECT_EQ(ConvertBf16ToFp32(src.first(3), dst.subspan(32, 4)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToBf16(src.first(3), dst.subspan(32, 2)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertBf16ToFp32(src.first(4), dst.subspan(32, 4)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToBf16(src.first(8), dst.subspan(32, 8)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertBf16ToFp32({}, dst.first(2)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToBf16({}, dst.first(2)), StatusCode::kInvalidArgument);
    const Bf16ConversionFuncs missing{nullptr, nullptr};
    EXPECT_EQ(ConvertBf16ToFp32(src.first(4), dst.subspan(32, 8), missing), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToBf16(src.first(8), dst.subspan(32, 4), missing), StatusCode::kInvalidArgument);
    EXPECT_EQ(bytes, original);
}

TEST(Bf16BatchConversion, OverlapInEitherAddressOrderIsRejectedBeforeWriting) {
    std::array<std::byte, 64> bytes{};
    bytes.fill(std::byte{0x5A});
    const auto original = bytes;
    const auto src = std::span<const std::byte>(bytes);
    auto dst = std::span(bytes);
    EXPECT_EQ(ConvertBf16ToFp32(src.first(8), dst.first(16)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertBf16ToFp32(src.first(8), dst.subspan(1, 16)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertBf16ToFp32(src.subspan(8, 8), dst.first(16)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToBf16(src.first(16), dst.first(8)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToBf16(src.first(16), dst.subspan(1, 8)), StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToBf16(src.subspan(4, 16), dst.first(8)), StatusCode::kInvalidArgument);
    EXPECT_EQ(bytes, original);
}

TEST(Bf16BatchConversion, AdjacentRangesAreAcceptedInEitherAddressOrder) {
    std::array<uint16_t, 12> storage{0x0001, 0x8000, 0x3F80, 0x7F81};
    auto bytes = std::as_writable_bytes(std::span(storage));
    EXPECT_TRUE(ConvertBf16ToFp32(bytes.first(8), bytes.subspan(8, 16)).ok());
    EXPECT_TRUE(ConvertFp32ToBf16(bytes.subspan(8, 16), bytes.first(8)).ok());
    EXPECT_EQ(storage[0], 0x0001U);
    EXPECT_EQ(storage[1], 0x8000U);
    EXPECT_EQ(storage[2], 0x3F80U);
    EXPECT_EQ(storage[3], 0x7FC1U);
}

class Bf16BatchEnvironment : public ::testing::Test {
protected:
    void SetUp() override {
        saved_ = std::feholdexcept(&environment_) == 0;
        ASSERT_TRUE(saved_);
    }
    void TearDown() override {
        if (saved_) {
            EXPECT_EQ(std::fesetenv(&environment_), 0);
        }
    }

private:
    std::fenv_t environment_{};
    bool saved_ = false;
};

TEST_F(Bf16BatchEnvironment, RoundingModesAndStickyExceptionFlagsAreUnchanged) {
    constexpr std::array<uint16_t, 10> bfloat{
            0x7F81, 0xFF81, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F80, 0xFF80, 0x7FE5, 0x8000};
    constexpr std::array<uint32_t, bfloat.size()> expected_wide{
            0x7FC10000, 0xFFC10000, 0x00010000, 0x80010000, 0x007F0000, 0x00800000, 0x7F800000, 0xFF800000, 0x7FE50000, 0x80000000};
    constexpr std::array<uint32_t, 14> floats{
            0x7F800001, 0xFF800001, 0x00000001, 0x80000001, 0x00008000, 0x00008001, 0x00018000,
            0x007F8000, 0x3F808000, 0x3F818000, 0x7F7F8000, 0xFF7F8000, 0x7F800000, 0x80000000};
    constexpr std::array<uint16_t, floats.size()> expected_narrow{
            0x7FC0, 0xFFC0, 0x0000, 0x8000, 0x0000, 0x0001, 0x0002,
            0x0080, 0x3F80, 0x3F82, 0x7F80, 0xFF80, 0x7F80, 0x8000};
    for (const int rounding: {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(rounding), 0);
        for (const int flags: {0, FE_DIVBYZERO, FE_ALL_EXCEPT}) {
            ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            ASSERT_EQ(std::feraiseexcept(flags), 0);
            const int original_flags = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__) || defined(_M_X64)
            const unsigned control = _mm_getcsr();
#endif
            std::array<uint32_t, bfloat.size()> widened{};
            std::array<uint16_t, floats.size()> narrowed{};
            EXPECT_TRUE(ConvertBf16ToFp32(std::as_bytes(std::span(bfloat)), std::as_writable_bytes(std::span(widened))).ok());
            EXPECT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(floats)), std::as_writable_bytes(std::span(narrowed))).ok());
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), original_flags);
            EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
            EXPECT_EQ(_mm_getcsr(), control);
#endif
            EXPECT_EQ(widened, expected_wide);
            EXPECT_EQ(narrowed, expected_narrow);
        }
    }
}

TEST_F(Bf16BatchEnvironment, SelectedConversionsPreserveRoundingAndExceptionFlags) {
    constexpr std::array<uint16_t, 10> bfloat{
            0x7F81, 0xFF81, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F80, 0xFF80, 0x7FE5, 0x8000};
    constexpr std::array<uint32_t, bfloat.size()> expected_wide{
            0x7FC10000, 0xFFC10000, 0x00010000, 0x80010000, 0x007F0000,
            0x00800000, 0x7F800000, 0xFF800000, 0x7FE50000, 0x80000000};
    constexpr std::array<uint32_t, 14> floats{
            0x7F800001, 0xFF800001, 0x00000001, 0x80000001, 0x00008000, 0x00008001, 0x00018000,
            0x007F8000, 0x3F808000, 0x3F818000, 0x7F7F8000, 0xFF7F8000, 0x7F800000, 0x80000000};
    constexpr std::array<uint16_t, floats.size()> expected_narrow{
            0x7FC0, 0xFFC0, 0x0000, 0x8000, 0x0000, 0x0001, 0x0002,
            0x0080, 0x3F80, 0x3F82, 0x7F80, 0xFF80, 0x7F80, 0x8000};
    const auto functions = SelectedConversions();
    for (const int rounding: {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(rounding), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
#if defined(__x86_64__) || defined(_M_X64)
        const unsigned control = _mm_getcsr();
#endif
        std::array<uint32_t, bfloat.size()> widened{};
        std::array<uint16_t, floats.size()> narrowed{};
        ASSERT_TRUE(ConvertBf16ToFp32(std::as_bytes(std::span(bfloat)),
                                      std::as_writable_bytes(std::span(widened)), functions)
                            .ok());
        ASSERT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(floats)),
                                      std::as_writable_bytes(std::span(narrowed)), functions)
                            .ok());
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
        EXPECT_EQ(_mm_getcsr(), control);
#endif
        EXPECT_EQ(widened, expected_wide);
        EXPECT_EQ(narrowed, expected_narrow);
    }
}

#if defined(__x86_64__) || defined(_M_X64)
TEST_F(Bf16BatchEnvironment, DenormalControlsDoNotChangeValuesOrFlags) {
    alignas(16) std::array<std::byte, 512> state{};
    _fxsave(reinterpret_cast<unsigned char*>(state.data()));
    uint32_t writable_control;
    std::memcpy(&writable_control, state.data() + 28, sizeof(writable_control));
    if (writable_control == 0) writable_control = 0xFFBFU;
    const unsigned supported = writable_control & 0x8040U;
    const unsigned original = _mm_getcsr();
    constexpr std::array<uint16_t, 7> source_wide{0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x7F81, 0xFF81};
    constexpr std::array<uint32_t, source_wide.size()> expected_wide{
            0x00010000, 0x80010000, 0x007F0000, 0x807F0000, 0x00800000, 0x7FC10000, 0xFFC10000};
    constexpr std::array<uint32_t, 8> source_narrow{
            0x00000001, 0x80000001, 0x00008000, 0x00008001, 0x80008001, 0x007F8000, 0x7F800001, 0xFF800001};
    constexpr std::array<uint16_t, source_narrow.size()> expected_narrow{0, 0x8000, 0, 1, 0x8001, 0x0080, 0x7FC0, 0xFFC0};
    for (const unsigned flags: {0U, 0x8000U, 0x0040U, 0x8040U}) {
        const unsigned control = (original & ~supported) | (flags & supported);
        _mm_setcsr(control);
        std::array<uint32_t, source_wide.size()> actual_wide{};
        std::array<uint16_t, source_narrow.size()> actual_narrow{};
        EXPECT_TRUE(ConvertBf16ToFp32(std::as_bytes(std::span(source_wide)), std::as_writable_bytes(std::span(actual_wide))).ok());
        EXPECT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(source_narrow)), std::as_writable_bytes(std::span(actual_narrow))).ok());
        EXPECT_EQ(_mm_getcsr(), control);
        EXPECT_EQ(actual_wide, expected_wide);
        EXPECT_EQ(actual_narrow, expected_narrow);
    }
}
#endif

TEST(CpuBf16Conversion, SelectionRequiresAvxAndSelectsAvx2UnderSupport) {
    const auto none = cpu::SelectBf16Conversions({});
    EXPECT_EQ(none.to_fp32, ConvertBf16ToFp32Scalar);
    EXPECT_EQ(none.from_fp32, ConvertFp32ToBf16Scalar);
    const auto no_avx = cpu::SelectBf16Conversions(CpuFeatureSet::From({CpuFeature::kAvx2}));
    EXPECT_EQ(no_avx.to_fp32, ConvertBf16ToFp32Scalar);
    EXPECT_EQ(no_avx.from_fp32, ConvertFp32ToBf16Scalar);
    const auto avx2 = cpu::SelectBf16Conversions(CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kAvx2}));
#if defined(BF16_HAS_AVX2_CONVERSION)
    EXPECT_NE(avx2.to_fp32, ConvertBf16ToFp32Scalar);
    EXPECT_NE(avx2.from_fp32, ConvertFp32ToBf16Scalar);
#else
    EXPECT_EQ(avx2.to_fp32, ConvertBf16ToFp32Scalar);
    EXPECT_EQ(avx2.from_fp32, ConvertFp32ToBf16Scalar);
#endif
}

TEST(CpuBf16Conversion, RestrictivePolicyDoesNotGetReplacedByFreshDetection) {
    auto detected = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok());
    CpuFeaturePolicy policy;
    policy.disabled_features.Enable(CpuFeature::kAvx);
    auto restricted = cpu::ApplyCpuFeaturePolicy(*detected, policy);
    ASSERT_TRUE(restricted.ok());
    const auto functions = cpu::SelectBf16Conversions(restricted->effective_features);
    EXPECT_EQ(functions.to_fp32, ConvertBf16ToFp32Scalar);
    EXPECT_EQ(functions.from_fp32, ConvertFp32ToBf16Scalar);
}

#if defined(__linux__)
class GuardedPages {
public:
    GuardedPages() = default;
    GuardedPages(const GuardedPages&) = delete;
    GuardedPages& operator=(const GuardedPages&) = delete;
    ~GuardedPages() {
        if (storage_ != MAP_FAILED) {
            munmap(storage_, page_size_ * 2);
        }
    }

    bool Init() {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) return false;
        page_size_ = static_cast<size_t>(page_size);
        storage_ = mmap(nullptr, page_size_ * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return storage_ != MAP_FAILED && mprotect(static_cast<std::byte*>(storage_) + page_size_, page_size_, PROT_NONE) == 0;
    }

    AM_NODISCARD std::span<std::byte> Tail(size_t bytes) const {
        return {static_cast<std::byte*>(storage_) + page_size_ - bytes, bytes};
    }

private:
    void* storage_ = MAP_FAILED;
    size_t page_size_ = 0;
};

TEST(Bf16BatchConversion, TailsDoNotReadOrWritePastProtectedPages) {
    GuardedPages source, destination;
    ASSERT_TRUE(source.Init());
    ASSERT_TRUE(destination.Init());
    for (size_t count = 1; count <= 65; ++count) {
        auto input = source.Tail(count * 2);
        std::fill(input.begin(), input.end(), std::byte{0x3C});
        EXPECT_TRUE(ConvertBf16ToFp32(input, destination.Tail(count * 4)).ok());
        auto floats = source.Tail(count * 4);
        std::fill(floats.begin(), floats.end(), std::byte{0x3F});
        EXPECT_TRUE(ConvertFp32ToBf16(floats, destination.Tail(count * 2)).ok());
    }
}

TEST(Bf16BatchConversion, SelectedConversionsTailsDoNotReadOrWritePastProtectedPages) {
    GuardedPages source, destination;
    ASSERT_TRUE(source.Init());
    ASSERT_TRUE(destination.Init());
    const auto functions = SelectedConversions();
    for (size_t count = 1; count <= 65; ++count) {
        auto input = source.Tail(count * 2);
        std::ranges::fill(input, std::byte{0x3C});
        EXPECT_TRUE(ConvertBf16ToFp32(input, destination.Tail(count * 4), functions).ok());
        auto floats = source.Tail(count * 4);
        std::ranges::fill(floats, std::byte{0x3F});
        EXPECT_TRUE(ConvertFp32ToBf16(floats, destination.Tail(count * 2), functions).ok());
    }
}
#endif

} // namespace
