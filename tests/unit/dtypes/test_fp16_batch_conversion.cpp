#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/fp16_conversion.h"
#include "aethermind/dtypes/fp16_conversion.h"
#include "aethermind/dtypes/half.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
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

Fp16ConversionFuncs SelectedConversions() {
    static const auto capabilities = cpu::DetectCpuCapabilities();
    EXPECT_TRUE(capabilities.ok());
    return capabilities.ok() ? cpu::SelectFp16Conversions(capabilities->effective_features)
                             : Fp16ConversionFuncs{};
}

TEST(Fp16BatchConversion, MixedLanesMatchScalar) {
    constexpr std::array<uint16_t, 17> input{
            0x0001, 0x8000, 0x3C00, 0x7C01, 0xFC01, 0x7C00, 0xFC00, 0x03FF,
            0x0400, 0x7BFF, 0xFBFF, 0x7E55, 0xFE55, 0x8001, 0xBC00, 0x0000, 0x3555};
    std::array<uint32_t, input.size()> expected{}, actual{};
    const auto source = std::as_bytes(std::span(input));
    ASSERT_TRUE(ConvertFp16ToFp32(source, std::as_writable_bytes(std::span(expected))).ok());
    ASSERT_TRUE(ConvertFp16ToFp32(source, std::as_writable_bytes(std::span(actual)), SelectedConversions()).ok());
    EXPECT_EQ(actual, expected);
}

TEST(Fp16BatchConversion, EveryHalfPatternMatchesScalarInEveryLane) {
    std::vector<uint16_t> source(0x10000U + 7);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint16_t>(index);
    }

    std::vector<uint32_t> actual(0x10000U);
    const auto functions = SelectedConversions();
    for (size_t rotation = 0; rotation < 8; ++rotation) {
        const auto input = std::as_bytes(
                std::span(source).subspan(rotation, actual.size()));
        ASSERT_TRUE(ConvertFp16ToFp32(
                            input,
                            std::as_writable_bytes(std::span(actual)), functions)
                            .ok());
        for (size_t index = 0; index < actual.size(); ++index) {
            const Half reference{source[index + rotation], Half::from_bits()};
            ASSERT_EQ(actual[index], std::bit_cast<uint32_t>(static_cast<float>(reference)))
                    << "index=" << index << " rotation=" << rotation;
        }
    }
}

TEST(Fp16BatchConversion, NarrowingMatchesScalarAtEveryFiniteMidpoint) {
    std::vector<uint32_t> source;
    for (uint32_t lower = 0; lower < 0x7BFFU; ++lower) {
        const double a = static_cast<float>(Half{static_cast<uint16_t>(lower), Half::from_bits()});
        const double b = static_cast<float>(Half{static_cast<uint16_t>(lower + 1U), Half::from_bits()});
        const auto midpoint = static_cast<float>((a + b) * 0.5);
        for (const float value: {std::nextafter(midpoint, 0.0f), midpoint,
                                 std::nextafter(midpoint, std::numeric_limits<float>::infinity())}) {
            const auto bits = std::bit_cast<uint32_t>(value);
            source.push_back(bits);
            source.push_back(bits | 0x80000000U);
        }
    }

    std::vector<uint16_t> actual(source.size()), expected(source.size());
    const auto input = std::as_bytes(std::span(source));
    ASSERT_TRUE(ConvertFp32ToFp16(input,
                                  std::as_writable_bytes(std::span(expected)))
                        .ok());
    ASSERT_TRUE(ConvertFp32ToFp16(input,
                                  std::as_writable_bytes(std::span(actual)),
                                  SelectedConversions())
                        .ok());
    for (size_t index = 0; index < actual.size(); ++index) {
        ASSERT_EQ(actual[index], expected[index]) << "source bits=" << source[index];
    }
}

TEST(Fp16BatchConversion, NarrowingMatchesScalarForRandomBitPatterns) {
    std::mt19937 random(20261110);
    std::vector<uint32_t> source(200003);
    for (auto& bits: source) {
        bits = random();
    }

    std::vector<uint16_t> actual(source.size()), expected(source.size());
    const auto input = std::as_bytes(std::span(source));
    ASSERT_TRUE(ConvertFp32ToFp16(input,
                                  std::as_writable_bytes(std::span(expected)))
                        .ok());
    ASSERT_TRUE(ConvertFp32ToFp16(input,
                                  std::as_writable_bytes(std::span(actual)),
                                  SelectedConversions())
                        .ok());
    for (size_t index = 0; index < actual.size(); ++index) {
        ASSERT_EQ(actual[index], expected[index]) << "source bits=" << source[index];
    }
}

TEST(Fp16BatchConversion, UnalignedRangesAndAllTailsMatchScalarWithoutOverwrites) {
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
                    ASSERT_TRUE(ConvertFp16ToFp32(input, expected).ok());
                    ASSERT_TRUE(ConvertFp16ToFp32(input, output, functions).ok());
                } else {
                    ASSERT_TRUE(ConvertFp32ToFp16(input, expected).ok());
                    ASSERT_TRUE(ConvertFp32ToFp16(input, output, functions).ok());
                }
                EXPECT_TRUE(std::equal(output.begin(), output.end(), expected.begin()));
                EXPECT_TRUE(std::all_of(actual.begin(), actual.begin() + destination_offset,
                                        [](std::byte value) { return value == std::byte{0xCD}; }));
                EXPECT_TRUE(std::all_of(actual.begin() + destination_offset + output.size(), actual.end(),
                                        [](std::byte value) { return value == std::byte{0xCD}; }));
                EXPECT_EQ(source, original);
            }
        }
    }
}

TEST(Fp16BatchConversion, EmptyRangesPermitNullPointersAndFunctions) {
    constexpr Fp16ConversionFuncs functions{nullptr, nullptr};
    EXPECT_TRUE(ConvertFp16ToFp32({}, {}, functions).ok());
    EXPECT_TRUE(ConvertFp32ToFp16({}, {}, functions).ok());
}

TEST(Fp16BatchConversion, RejectsInvalidRangesBeforeWriting) {
    std::array<std::byte, 64> bytes{};
    bytes.fill(std::byte{0x5A});
    const auto original = bytes;
    const auto src = std::span<const std::byte>(bytes);
    auto dst = std::span(bytes);
    EXPECT_EQ(ConvertFp16ToFp32(src.first(3),
                                dst.subspan(32, 4)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToFp16(src.first(3),
                                dst.subspan(32, 2)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp16ToFp32(src.first(4),
                                dst.subspan(32, 4)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToFp16(src.first(8),
                                dst.subspan(32, 8)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp16ToFp32(src.first(8),
                                dst.subspan(1, 16)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp16ToFp32(src.subspan(8, 8),
                                dst.first(16)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToFp16(src.first(16),
                                dst.subspan(1, 8)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToFp16(src.subspan(4, 16),
                                dst.first(8)),
              StatusCode::kInvalidArgument);
    const Fp16ConversionFuncs missing{nullptr, nullptr};
    EXPECT_EQ(ConvertFp16ToFp32(src.first(4),
                                dst.subspan(32, 8), missing),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertFp32ToFp16(src.first(8),
                                dst.subspan(32, 4), missing),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(bytes, original);
}

TEST(CpuFp16Conversion, SelectionRequiresAvxAndIsIndependentPerDirection) {
    const auto none = cpu::SelectFp16Conversions({});
    EXPECT_EQ(none.to_fp32, ConvertFp16ToFp32Scalar);
    EXPECT_EQ(none.from_fp32, ConvertFp32ToFp16Scalar);
    const auto no_avx = cpu::SelectFp16Conversions(
            CpuFeatureSet::From({CpuFeature::kF16c, CpuFeature::kAvx2}));
    EXPECT_EQ(no_avx.to_fp32, ConvertFp16ToFp32Scalar);
    EXPECT_EQ(no_avx.from_fp32, ConvertFp32ToFp16Scalar);
    const auto f16c = cpu::SelectFp16Conversions(
            CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kF16c}));
    const auto avx2 = cpu::SelectFp16Conversions(
            CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kAvx2}));
#if defined(FP16_HAS_F16C_CONVERSION)
    EXPECT_NE(f16c.to_fp32, ConvertFp16ToFp32Scalar);
#else
    EXPECT_EQ(f16c.to_fp32, ConvertFp16ToFp32Scalar);
#endif
#if defined(FP16_HAS_AVX2_CONVERSION)
    EXPECT_NE(avx2.from_fp32, ConvertFp32ToFp16Scalar);
#else
    EXPECT_EQ(avx2.from_fp32, ConvertFp32ToFp16Scalar);
#endif
    EXPECT_EQ(f16c.from_fp32, ConvertFp32ToFp16Scalar);
    EXPECT_EQ(avx2.to_fp32, ConvertFp16ToFp32Scalar);
}

TEST(CpuFp16Conversion, RestrictivePolicyDoesNotGetReplacedByFreshDetection) {
    auto detected = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok());
    CpuFeaturePolicy policy;
    policy.disabled_features.Enable(CpuFeature::kAvx);
    auto restricted = cpu::ApplyCpuFeaturePolicy(*detected, policy);
    ASSERT_TRUE(restricted.ok());
    const auto functions = cpu::SelectFp16Conversions(restricted->effective_features);
    EXPECT_EQ(functions.to_fp32, ConvertFp16ToFp32Scalar);
    EXPECT_EQ(functions.from_fp32, ConvertFp32ToFp16Scalar);
}

class Fp16BatchEnvironment : public ::testing::Test {
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

TEST_F(Fp16BatchEnvironment, PreservesRoundingAndExceptionFlags) {
    constexpr std::array<uint16_t, 9> halves{
            0x7C01, 0xFC01, 0x0001, 0x8001, 0x03FF, 0x7C00, 0xFC00, 0x7E55, 0x8000};
    constexpr std::array<uint32_t, 17> floats{
            0x7F800001, 0xFF800001, 0x7F7FFFFF, 0xFF7FFFFF, 0x00000001, 0x80000001,
            0x33000000, 0x33000001, 0x33800000, 0x33C00000, 0x387FE000, 0x3F801000,
            0x3F803000, 0x477FF000, 0x7F800000, 0xFF800000, 0x80000000};
    std::array<uint32_t, halves.size()> widened{};
    std::array<uint16_t, floats.size()> narrowed{};
    const auto functions = SelectedConversions();
    for (const int rounding: {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(rounding), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
#if defined(__x86_64__) || defined(_M_X64)
        const unsigned control = _mm_getcsr();
#endif
        ASSERT_TRUE(ConvertFp16ToFp32(
                            std::as_bytes(std::span(halves)),
                            std::as_writable_bytes(std::span(widened)),
                            functions)
                            .ok());
        ASSERT_TRUE(ConvertFp32ToFp16(
                            std::as_bytes(std::span(floats)),
                            std::as_writable_bytes(std::span(narrowed)),
                            functions)
                            .ok());
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
        EXPECT_EQ(_mm_getcsr(), control);
#endif
        EXPECT_EQ(widened[0], 0x7FC02000U);
        EXPECT_EQ(narrowed[7], 0x0001U);
        EXPECT_EQ(narrowed[10], 0x0400U);
        EXPECT_EQ(narrowed[13], 0x7C00U);
    }
}

#if defined(__x86_64__) || defined(_M_X64)
TEST_F(Fp16BatchEnvironment, DenormalControlsDoNotChangeValuesOrFlags) {
    alignas(16) std::array<std::byte, 512> state{};
    _fxsave(reinterpret_cast<unsigned char*>(state.data()));
    uint32_t writable_control;
    std::memcpy(&writable_control, state.data() + 28, sizeof(writable_control));
    // A zero MXCSR_MASK has the architectural default, excluding optional DAZ.
    if (writable_control == 0) writable_control = 0xFFBFU;
    const unsigned supported = writable_control & 0x8040U;
    const unsigned original = _mm_getcsr();
    const auto functions = SelectedConversions();
    constexpr std::array<uint16_t, 9> halves{
            1, 0x8001, 0x03FF, 0x83FF, 0x7C01, 0xFC01, 0, 0x8000, 0x0400};
    constexpr std::array<uint32_t, 9> floats{
            1, 0x80000001, 0x33800000, 0xB3800000, 0x33000001,
            0x387FE000, 0x7F800001, 0xFF800001, 0x477FF000};
    std::array<uint32_t, halves.size()> expected_wide{}, actual_wide{};
    std::array<uint16_t, floats.size()> expected_narrow{}, actual_narrow{};
    ASSERT_TRUE(ConvertFp16ToFp32(
                        std::as_bytes(std::span(halves)),
                        std::as_writable_bytes(std::span(expected_wide)))
                        .ok());
    ASSERT_TRUE(ConvertFp32ToFp16(
                        std::as_bytes(std::span(floats)),
                        std::as_writable_bytes(std::span(expected_narrow)))
                        .ok());
    for (const unsigned flags: {0U, 0x8000U, 0x0040U, 0x8040U}) {
        const unsigned control = (original & ~supported) | (flags & supported);
        _mm_setcsr(control);
        ASSERT_TRUE(ConvertFp16ToFp32(
                            std::as_bytes(std::span(halves)),
                            std::as_writable_bytes(std::span(actual_wide)),
                            functions)
                            .ok());
        ASSERT_TRUE(ConvertFp32ToFp16(
                            std::as_bytes(std::span(floats)),
                            std::as_writable_bytes(std::span(actual_narrow)),
                            functions)
                            .ok());
        EXPECT_EQ(_mm_getcsr(), control);
        EXPECT_EQ(actual_wide, expected_wide);
        EXPECT_EQ(actual_narrow, expected_narrow);
    }
}
#endif

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
        storage_ = mmap(nullptr, page_size_ * 2,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return storage_ != MAP_FAILED &&
               mprotect(static_cast<std::byte*>(storage_) + page_size_,
                        page_size_, PROT_NONE) == 0;
    }

    AM_NODISCARD std::span<std::byte> Tail(size_t bytes) const {
        return {static_cast<std::byte*>(storage_) + page_size_ - bytes, bytes};
    }

private:
    void* storage_ = MAP_FAILED;
    size_t page_size_ = 0;
};

TEST(Fp16BatchConversion, TailsDoNotReadOrWritePastProtectedPages) {
    GuardedPages source, destination;
    ASSERT_TRUE(source.Init());
    ASSERT_TRUE(destination.Init());
    const auto functions = SelectedConversions();
    for (size_t count = 1; count <= 33; ++count) {
        auto input = source.Tail(count * 2);
        std::ranges::fill(input, std::byte{0x3C});
        ASSERT_TRUE(ConvertFp16ToFp32(input,
                                      destination.Tail(count * 4),
                                      functions)
                            .ok());
        auto floats = source.Tail(count * 4);
        std::ranges::fill(floats, std::byte{0x3F});
        ASSERT_TRUE(ConvertFp32ToFp16(floats,
                                      destination.Tail(count * 2),
                                      functions)
                            .ok());
    }
}
#endif

} // namespace
