#include "aethermind/backend/cpu/cpu_info.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"
#include "gemm_avx2_candidate_entries.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)

// Exercise the production AVX2 GEMM and the DupA candidate around tile and
// panel boundaries against the double-accumulation reference.

float TestValue(size_t index) {
    return static_cast<float>(static_cast<int64_t>((index * 17U + 5U) % 29U) - 14) * 0.125F;
}

void FillValues(std::vector<float>& values) {
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = TestValue(index);
    }
}

bool CanExecuteAvx2Fma() {
    const auto capabilities = cpu::DetectCpuCapabilities();
    return capabilities.ok() &&
           capabilities->effective_features.Contains(CpuFeature::kAvx2) &&
           capabilities->effective_features.Contains(CpuFeature::kFma);
}

enum class Tile6Entry {
    kProduction,
    kDupA,
};

const char* Tile6EntryName(Tile6Entry entry) {
    switch (entry) {
        case Tile6Entry::kProduction:
            return "production";
        case Tile6Entry::kDupA:
            return "dupA";
    }
    return "unknown";
}

Status RunTile6(Tile6Entry entry, const cpu::detail::GemmArgs& args) noexcept {
    switch (entry) {
        case Tile6Entry::kProduction:
            return cpu::detail::RunGemmA32W32Avx2Fma(args);
        case Tile6Entry::kDupA:
            return cpu::detail::RunGemmA32W32Avx2FmaTile6DupA(args);
    }
    return Status::Internal("unknown Tile6 entry");
}

void ExpectTile6NearReference(Tile6Entry entry,
                              const cpu::detail::GemmArgs& candidate_args,
                              const cpu::detail::GemmArgs& reference_args) {
    const Status candidate = RunTile6(entry, candidate_args);
    ASSERT_TRUE(candidate.ok()) << candidate.ToString();
    const Status reference = cpu::detail::RunGemmA32W32Reference(reference_args);
    ASSERT_TRUE(reference.ok()) << reference.ToString();

    const float scale_adjusted_absolute_tolerance =
            2.0e-4F + 2.0e-5F * std::sqrt(static_cast<float>(candidate_args.k));
    for (int64_t row = 0; row < candidate_args.m; ++row) {
        for (int64_t col = 0; col < candidate_args.n; ++col) {
            const float actual = reinterpret_cast<const float*>(candidate_args.output)[row * candidate_args.output_m_stride +
                                                                                       col * candidate_args.output_n_stride];
            const float expected = reinterpret_cast<const float*>(reference_args.output)[row * reference_args.output_m_stride +
                                                                                         col * reference_args.output_n_stride];
            const float absolute_error = std::fabs(actual - expected);
            const float relative_error = absolute_error / std::max(std::fabs(expected), 1.0e-6F);
            EXPECT_TRUE(absolute_error <= scale_adjusted_absolute_tolerance ||
                        relative_error <= 2.0e-4F)
                    << "row=" << row << " col=" << col << " actual=" << actual
                    << " expected=" << expected << " absolute_error=" << absolute_error
                    << " relative_error=" << relative_error;
        }
    }
}

struct Tile6Shape {
    Tile6Entry entry;
    int64_t m;
    int64_t k;
    int64_t n;
};

std::string Tile6ShapeName(const testing::TestParamInfo<Tile6Shape>& info) {
    return std::string{Tile6EntryName(info.param.entry)} + "M" +
           std::to_string(info.param.m) + "K" + std::to_string(info.param.k) +
           "N" + std::to_string(info.param.n);
}

class CPUKernelGemmTile6BoundaryTest : public testing::TestWithParam<Tile6Shape> {};

TEST_P(CPUKernelGemmTile6BoundaryTest, MatchesDoubleReference) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    const Tile6Shape test_case = GetParam();
    // Padded strides exercise the packed-panel pointer arithmetic as well as
    // the N-tail and M-tail edge handling of the blocked path.
    const int64_t lhs_m_stride = test_case.k + 3;
    const int64_t output_m_stride = test_case.n + 2;
    const int64_t rhs_n_stride = test_case.k + 5;
    std::vector<float> lhs(static_cast<size_t>(test_case.m * lhs_m_stride));
    std::vector<float> rhs(static_cast<size_t>(test_case.n * rhs_n_stride));
    std::vector<float> candidate_output(static_cast<size_t>(test_case.m * output_m_stride), 11.0F);
    std::vector<float> reference_output(static_cast<size_t>(test_case.m * output_m_stride), -13.0F);
    FillValues(lhs);
    FillValues(rhs);

    const cpu::detail::GemmArgs candidate_args{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
            .output = reinterpret_cast<std::byte*>(candidate_output.data()),
            .m = test_case.m,
            .n = test_case.n,
            .k = test_case.k,
            .lhs_m_stride = lhs_m_stride,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = rhs_n_stride,
            .output_m_stride = output_m_stride,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reinterpret_cast<std::byte*>(reference_output.data());
    ExpectTile6NearReference(test_case.entry, candidate_args, reference_args);

    for (int64_t row = 0; row < test_case.m; ++row) {
        EXPECT_EQ(candidate_output[static_cast<size_t>(row * output_m_stride + test_case.n)], 11.0F);
        EXPECT_EQ(reference_output[static_cast<size_t>(row * output_m_stride + test_case.n)], -13.0F);
    }
}

INSTANTIATE_TEST_SUITE_P(
        Tile6Boundaries,
        CPUKernelGemmTile6BoundaryTest,
        testing::Values(
                // Production entry: narrow-N row pairs, 6/4/2-row blocked
                // decomposition and the odd-row tail.
                Tile6Shape{Tile6Entry::kProduction, 9, 33, 9},
                Tile6Shape{Tile6Entry::kProduction, 10, 17, 17},
                Tile6Shape{Tile6Entry::kProduction, 11, 31, 31},
                Tile6Shape{Tile6Entry::kProduction, 15, 513, 15},
                Tile6Shape{Tile6Entry::kProduction, 16, 1024, 33},
                Tile6Shape{Tile6Entry::kProduction, 17, 511, 65},
                Tile6Shape{Tile6Entry::kProduction, 33, 4097, 17},
                Tile6Shape{Tile6Entry::kProduction, 64, 512, 129},
                Tile6Shape{Tile6Entry::kProduction, 129, 1025, 264},
                // Dup-A entry: KC=256 / MC=48 / NC=128 panel boundaries and
                // their off-by-one neighbours.
                Tile6Shape{Tile6Entry::kDupA, 9, 33, 9},
                Tile6Shape{Tile6Entry::kDupA, 47, 257, 127},
                Tile6Shape{Tile6Entry::kDupA, 48, 256, 128},
                Tile6Shape{Tile6Entry::kDupA, 49, 255, 129},
                Tile6Shape{Tile6Entry::kDupA, 16, 1025, 33},
                Tile6Shape{Tile6Entry::kDupA, 65, 513, 264},
                Tile6Shape{Tile6Entry::kDupA, 129, 4096, 55}),
        Tile6ShapeName);

TEST(CPUKernelGemmTile6, OverwritesNanPrefilledOutput) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    constexpr int64_t m = 10;
    constexpr int64_t k = 512;
    constexpr int64_t n = 17;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs(static_cast<size_t>(n * k));
    std::vector<float> candidate_output(static_cast<size_t>(m * n),
                                        std::numeric_limits<float>::quiet_NaN());
    std::vector<float> reference_output(static_cast<size_t>(m * n),
                                        std::numeric_limits<float>::quiet_NaN());
    FillValues(lhs);
    FillValues(rhs);

    const cpu::detail::GemmArgs candidate_args{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
            .output = reinterpret_cast<std::byte*>(candidate_output.data()),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = k,
            .output_m_stride = n,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reinterpret_cast<std::byte*>(reference_output.data());
    for (const Tile6Entry entry: {Tile6Entry::kProduction, Tile6Entry::kDupA}) {
        // Re-poison the candidate buffer so each entry proves it overwrites
        // every logical output element on its own.
        std::fill(candidate_output.begin(), candidate_output.end(),
                  std::numeric_limits<float>::quiet_NaN());
        ExpectTile6NearReference(entry, candidate_args, reference_args);
        for (const float value: candidate_output) {
            EXPECT_TRUE(std::isfinite(value));
        }
    }
}

TEST(CPUKernelGemmTile6, CancelsAcrossKPanelsInAscendingOrder) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    // Alternating +1/-1 products over an even K: ascending accumulation keeps
    // every partial sum in {-1, 0, +1} exactly, so the result must be exactly
    // +0.0 across K panels for both drivers. M >= 9 and N = NR keep both
    // entries fully on the blocked path (12 rows decompose to two 6-row tiles).
    constexpr int64_t m = 12;
    constexpr int64_t k = 1024;
    constexpr int64_t n = 16;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs(static_cast<size_t>(n * k), 1.0F);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t index = 0; index < k; ++index) {
            lhs[static_cast<size_t>(row * k + index)] = (index % 2 == 0) ? 1.0F : -1.0F;
        }
    }

    for (const Tile6Entry entry: {Tile6Entry::kProduction, Tile6Entry::kDupA}) {
        std::vector<float> output(static_cast<size_t>(m * n), -7.0F);
        const cpu::detail::GemmArgs candidate_args{
                .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
                .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
                .output = reinterpret_cast<std::byte*>(output.data()),
                .m = m,
                .n = n,
                .k = k,
                .lhs_m_stride = k,
                .lhs_k_stride = 1,
                .rhs_k_stride = 1,
                .rhs_n_stride = k,
                .output_m_stride = n,
                .output_n_stride = 1,
        };
        const Status status = RunTile6(entry, candidate_args);
        ASSERT_TRUE(status.ok()) << status.ToString();
        for (const float value: output) {
            EXPECT_EQ(value, 0.0F) << Tile6EntryName(entry);
            EXPECT_FALSE(std::signbit(value)) << Tile6EntryName(entry);
        }
    }
}

TEST(CPUKernelGemmTile6, HandlesSmallMAndNContiguousWeights) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    // The production entry uses row pairs here. DupA falls back to scalar.
    constexpr int64_t small_m = 3;
    constexpr int64_t k = 257;
    constexpr int64_t n = 129;
    std::vector<float> lhs(static_cast<size_t>(small_m * k));
    std::vector<float> rhs(static_cast<size_t>(n * k));
    std::vector<float> candidate_output(static_cast<size_t>(small_m * n), 7.0F);
    std::vector<float> reference_output(static_cast<size_t>(small_m * n), -11.0F);
    FillValues(lhs);
    FillValues(rhs);

    const cpu::detail::GemmArgs small_m_args{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
            .output = reinterpret_cast<std::byte*>(candidate_output.data()),
            .m = small_m,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = k,
            .output_m_stride = n,
            .output_n_stride = 1,
    };
    auto small_m_reference_args = small_m_args;
    small_m_reference_args.output = reinterpret_cast<std::byte*>(reference_output.data());
    for (const Tile6Entry entry: {Tile6Entry::kProduction, Tile6Entry::kDupA}) {
        ExpectTile6NearReference(entry, small_m_args, small_m_reference_args);
    }

    // The production entry selects the N-contiguous SIMD path here. DupA
    // delegates to the scalar driver.
    constexpr int64_t m = 16;
    constexpr int64_t rhs_k_stride = n + 4;
    std::vector<float> strided_lhs(static_cast<size_t>(m * (k + 3)));
    std::vector<float> strided_rhs(static_cast<size_t>(k * rhs_k_stride));
    std::vector<float> strided_candidate(static_cast<size_t>(m * (n + 4)), 5.0F);
    std::vector<float> strided_reference(static_cast<size_t>(m * (n + 4)), -9.0F);
    FillValues(strided_lhs);
    FillValues(strided_rhs);

    const cpu::detail::GemmArgs strided_args{
            .lhs = reinterpret_cast<const std::byte*>(strided_lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(strided_rhs.data()),
            .output = reinterpret_cast<std::byte*>(strided_candidate.data()),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = k + 3,
            .lhs_k_stride = 1,
            .rhs_k_stride = rhs_k_stride,
            .rhs_n_stride = 1,
            .output_m_stride = n + 4,
            .output_n_stride = 1,
    };
    auto strided_reference_args = strided_args;
    strided_reference_args.output = reinterpret_cast<std::byte*>(strided_reference.data());
    for (const Tile6Entry entry: {Tile6Entry::kProduction, Tile6Entry::kDupA}) {
        ExpectTile6NearReference(entry, strided_args, strided_reference_args);
    }
}

TEST(CPUKernelGemmTile6, PreservesZeroDimensionAndOverwriteSemantics) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    for (const Tile6Entry entry: {Tile6Entry::kProduction, Tile6Entry::kDupA}) {
        std::array<float, 8> zero_output{};
        zero_output.fill(-7.0F);
        const Status zero_k = RunTile6(entry, cpu::detail::GemmArgs{
                                                      .output = reinterpret_cast<std::byte*>(zero_output.data()),
                                                      .m = 1,
                                                      .n = 3,
                                                      .k = 0,
                                                      .output_m_stride = 5,
                                                      .output_n_stride = 1,
                                              });
        ASSERT_TRUE(zero_k.ok()) << zero_k.ToString();
        for (int64_t col = 0; col < 3; ++col) {
            EXPECT_EQ(zero_output[static_cast<size_t>(col)], 0.0F) << Tile6EntryName(entry);
            EXPECT_FALSE(std::signbit(zero_output[static_cast<size_t>(col)])) << Tile6EntryName(entry);
        }

        EXPECT_TRUE(RunTile6(entry, cpu::detail::GemmArgs{.m = 0, .n = 3, .k = 7}).ok());
        constexpr int64_t k = 16;
        const std::array<float, 9 * k> lhs{};
        const cpu::detail::GemmArgs empty_output{
                .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
                .m = 9,
                .n = 0,
                .k = k,
                .lhs_m_stride = k,
                .lhs_k_stride = 1,
                .rhs_k_stride = 1,
                .rhs_n_stride = k,
                .output_m_stride = 3,
                .output_n_stride = 1,
        };
        EXPECT_TRUE(RunTile6(entry, empty_output).ok()) << Tile6EntryName(entry);
    }
}

#endif

} // namespace
