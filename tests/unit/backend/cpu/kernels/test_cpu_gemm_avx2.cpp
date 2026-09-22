#include "aethermind/backend/cpu/cpu_info.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)

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

void ExpectAvx2NearReference(const cpu::detail::GemmF32Args& candidate_args,
                             const cpu::detail::GemmF32Args& reference_args) {
    const Status candidate = cpu::detail::RunGemmF32Avx2Fma(candidate_args);
    ASSERT_TRUE(candidate.ok()) << candidate.ToString();
    const Status reference = cpu::detail::RunGemmF32Reference(reference_args);
    ASSERT_TRUE(reference.ok()) << reference.ToString();

    const float scale_adjusted_absolute_tolerance =
            2.0e-4F + 2.0e-5F * std::sqrt(static_cast<float>(candidate_args.k));
    for (int64_t row = 0; row < candidate_args.m; ++row) {
        for (int64_t col = 0; col < candidate_args.n; ++col) {
            const float actual = candidate_args.output[row * candidate_args.output_m_stride +
                                                       col * candidate_args.output_n_stride];
            const float expected = reference_args.output[row * reference_args.output_m_stride +
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

struct Avx2FastPathCase {
    int64_t k;
    int64_t n;
};

std::string Avx2FastPathCaseName(const testing::TestParamInfo<Avx2FastPathCase>& info) {
    return "K" + std::to_string(info.param.k) + "N" + std::to_string(info.param.n);
}

class CPUKernelGemmAvx2FastPathTest : public testing::TestWithParam<Avx2FastPathCase> {};

TEST_P(CPUKernelGemmAvx2FastPathTest, MatchesDoubleReferenceForKAndNTails) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    const Avx2FastPathCase test_case = GetParam();
    const int64_t weight_row_stride = test_case.k + 3;
    std::vector<float> lhs(static_cast<size_t>(test_case.k));
    std::vector<float> rhs(static_cast<size_t>(test_case.n * weight_row_stride));
    std::vector<float> candidate_output(static_cast<size_t>(test_case.n), 19.0F);
    std::vector<float> reference_output(static_cast<size_t>(test_case.n), -23.0F);
    FillValues(lhs);
    FillValues(rhs);

    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = 1,
            .n = test_case.n,
            .k = test_case.k,
            .lhs_m_stride = test_case.k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = weight_row_stride,
            .output_m_stride = test_case.n,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_output.data();
    ExpectAvx2NearReference(candidate_args, reference_args);
}

INSTANTIATE_TEST_SUITE_P(
        FastPathBoundaries,
        CPUKernelGemmAvx2FastPathTest,
        testing::Values(
                Avx2FastPathCase{1, 1},
                Avx2FastPathCase{7, 2},
                Avx2FastPathCase{8, 3},
                Avx2FastPathCase{9, 4},
                Avx2FastPathCase{15, 5},
                Avx2FastPathCase{16, 7},
                Avx2FastPathCase{17, 8},
                Avx2FastPathCase{31, 31},
                Avx2FastPathCase{32, 32},
                Avx2FastPathCase{33, 33},
                Avx2FastPathCase{127, 5},
                Avx2FastPathCase{128, 7},
                Avx2FastPathCase{129, 8},
                Avx2FastPathCase{4095, 31},
                Avx2FastPathCase{4096, 32},
                Avx2FastPathCase{4097, 33}),
        Avx2FastPathCaseName);

TEST(CPUKernelGemmAvx2, SupportsPaddedUnalignedInputsAndOutputGuards) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    constexpr int64_t k = 17;
    constexpr int64_t n = 5;
    constexpr int64_t weight_row_stride = 23;
    std::vector<float> lhs_storage(static_cast<size_t>(k + 1));
    std::vector<float> rhs_storage(static_cast<size_t>(n * weight_row_stride + 1));
    std::vector<float> candidate_storage(static_cast<size_t>(n + 2), 31.0F);
    std::vector<float> reference_storage(static_cast<size_t>(n + 2), -37.0F);
    FillValues(lhs_storage);
    FillValues(rhs_storage);

    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs_storage.data() + 1,
            .rhs = rhs_storage.data() + 1,
            .output = candidate_storage.data() + 1,
            .m = 1,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = weight_row_stride,
            .output_m_stride = n,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_storage.data() + 1;
    ExpectAvx2NearReference(candidate_args, reference_args);
    EXPECT_EQ(candidate_storage.front(), 31.0F);
    EXPECT_EQ(candidate_storage.back(), 31.0F);
    EXPECT_EQ(reference_storage.front(), -37.0F);
    EXPECT_EQ(reference_storage.back(), -37.0F);
}

TEST(CPUKernelGemmAvx2, FallsBackForNContiguousLayouts) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    constexpr int64_t m = 2;
    constexpr int64_t k = 33;
    constexpr int64_t n = 5;
    constexpr int64_t lhs_m_stride = k + 3;
    constexpr int64_t output_m_stride = n + 2;
    constexpr int64_t rhs_k_stride = n + 4;
    std::vector<float> lhs(static_cast<size_t>(m * lhs_m_stride));
    std::vector<float> rhs(static_cast<size_t>(k * rhs_k_stride));
    std::vector<float> candidate_output(static_cast<size_t>(m * output_m_stride), 13.0F);
    std::vector<float> reference_output(static_cast<size_t>(m * output_m_stride), -17.0F);
    FillValues(lhs);
    FillValues(rhs);

    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = lhs_m_stride,
            .lhs_k_stride = 1,
            .rhs_k_stride = rhs_k_stride,
            .rhs_n_stride = 1,
            .output_m_stride = output_m_stride,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_output.data();
    ExpectAvx2NearReference(candidate_args, reference_args);
}

struct SmallMShape {
    int64_t m;
    int64_t k;
    int64_t n;
};

std::string SmallMShapeName(const testing::TestParamInfo<SmallMShape>& info) {
    return "M" + std::to_string(info.param.m) + "K" + std::to_string(info.param.k) +
           "N" + std::to_string(info.param.n);
}

class CPUKernelGemmAvx2SmallMTest : public testing::TestWithParam<SmallMShape> {};

TEST_P(CPUKernelGemmAvx2SmallMTest, MatchesDoubleReference) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    const SmallMShape test_case = GetParam();
    // Padded row strides exercise the per-row pointer arithmetic of the
    // row-pair fast path and the leftover-row single-row path.
    const int64_t lhs_m_stride = test_case.k + 3;
    const int64_t output_m_stride = test_case.n + 2;
    const int64_t rhs_n_stride = test_case.k + 5;
    std::vector<float> lhs(static_cast<size_t>(test_case.m * lhs_m_stride));
    std::vector<float> rhs(static_cast<size_t>(test_case.n * rhs_n_stride));
    std::vector<float> candidate_output(static_cast<size_t>(test_case.m * output_m_stride), 17.0F);
    std::vector<float> reference_output(static_cast<size_t>(test_case.m * output_m_stride), -19.0F);
    FillValues(lhs);
    FillValues(rhs);

    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
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
    reference_args.output = reference_output.data();
    ExpectAvx2NearReference(candidate_args, reference_args);

    // Output guards stay untouched in the padding past each logical row end.
    for (int64_t row = 0; row < test_case.m; ++row) {
        EXPECT_EQ(candidate_output[static_cast<size_t>(row * output_m_stride + test_case.n)], 17.0F);
        EXPECT_EQ(candidate_output[static_cast<size_t>(row * output_m_stride + test_case.n + 1)], 17.0F);
        EXPECT_EQ(reference_output[static_cast<size_t>(row * output_m_stride + test_case.n)], -19.0F);
        EXPECT_EQ(reference_output[static_cast<size_t>(row * output_m_stride + test_case.n + 1)], -19.0F);
    }
}

INSTANTIATE_TEST_SUITE_P(
        SmallMShapes,
        CPUKernelGemmAvx2SmallMTest,
        testing::Values(
                SmallMShape{2, 8, 5},
                SmallMShape{2, 31, 31},
                SmallMShape{3, 33, 7},
                SmallMShape{4, 32, 33},
                SmallMShape{7, 33, 31},
                SmallMShape{8, 64, 9},
                SmallMShape{8, 1024, 33}),
        SmallMShapeName);

TEST(CPUKernelGemmAvx2, FallsBackForGenericMAndNonUnitStrides) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    constexpr int64_t m = 16;
    constexpr int64_t k = 17;
    constexpr int64_t n = 4;
    std::vector<float> lhs(static_cast<size_t>(m * (k + 3)));
    std::vector<float> rhs(static_cast<size_t>(n * (k + 5)));
    std::vector<float> candidate_output(static_cast<size_t>(m * (n + 4)), 7.0F);
    std::vector<float> reference_output(static_cast<size_t>(m * (n + 4)), -11.0F);
    FillValues(lhs);
    FillValues(rhs);

    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = k + 3,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = k + 5,
            .output_m_stride = n + 4,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_output.data();
    ExpectAvx2NearReference(candidate_args, reference_args);

    const cpu::detail::GemmF32Args strided_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = 1,
            .n = 2,
            .k = 3,
            .lhs_m_stride = 6,
            .lhs_k_stride = 2,
            .rhs_k_stride = 7,
            .rhs_n_stride = 2,
            .output_m_stride = 6,
            .output_n_stride = 2,
    };
    auto strided_reference_args = strided_args;
    strided_reference_args.output = reference_output.data();
    ExpectAvx2NearReference(strided_args, strided_reference_args);
}

TEST(CPUKernelGemmAvx2, PreservesZeroDimensionAndOverwriteSemantics) {
    if (!CanExecuteAvx2Fma()) {
        GTEST_SKIP() << "AVX2+FMA GEMM kernel is unavailable on this host";
    }

    std::array<float, 8> zero_output{};
    zero_output.fill(-7.0F);
    const Status zero_k = cpu::detail::RunGemmF32Avx2Fma(cpu::detail::GemmF32Args{
            .output = zero_output.data(),
            .m = 1,
            .n = 3,
            .k = 0,
            .output_m_stride = 5,
            .output_n_stride = 1,
    });
    ASSERT_TRUE(zero_k.ok()) << zero_k.ToString();
    for (int64_t col = 0; col < 3; ++col) {
        EXPECT_EQ(zero_output[static_cast<size_t>(col)], 0.0F);
        EXPECT_FALSE(std::signbit(zero_output[static_cast<size_t>(col)]));
    }

    EXPECT_TRUE(cpu::detail::RunGemmF32Avx2Fma(cpu::detail::GemmF32Args{
                                                       .m = 0,
                                                       .n = 3,
                                                       .k = 7,
                                               })
                        .ok());
    EXPECT_TRUE(cpu::detail::RunGemmF32Avx2Fma(cpu::detail::GemmF32Args{
                                                       .m = 1,
                                                       .n = 0,
                                                       .k = 7,
                                               })
                        .ok());

    constexpr std::array<float, 3> lhs = {1.0F, -2.0F, 0.5F};
    constexpr std::array<float, 6> rhs = {
            2.0F,
            1.0F,
            -1.0F,
            -0.5F,
            3.0F,
            4.0F,
    };
    std::array<float, 2> output = {101.0F, -101.0F};
    const Status overwrite = cpu::detail::RunGemmF32Avx2Fma(cpu::detail::GemmF32Args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = output.data(),
            .m = 1,
            .n = 2,
            .k = 3,
            .lhs_m_stride = 3,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = 3,
            .output_m_stride = 2,
            .output_n_stride = 1,
    });
    ASSERT_TRUE(overwrite.ok()) << overwrite.ToString();
    EXPECT_FLOAT_EQ(output[0], -0.5F);
    EXPECT_FLOAT_EQ(output[1], -4.5F);
}

#endif

} // namespace
