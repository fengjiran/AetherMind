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

enum class RhsLayout {
    kNContiguous,
    kKContiguous,
};

float TestValue(size_t index) {
    return static_cast<float>(static_cast<int64_t>((index * 17U + 5U) % 29U) - 14) * 0.125F;
}

void FillValues(std::vector<float>& values) {
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = TestValue(index);
    }
}

void ExpectCandidateNearReference(const cpu::detail::GemmF32Args& candidate_args,
                                  const cpu::detail::GemmF32Args& reference_args) {
    const Status candidate = cpu::detail::RunGemmF32ScalarOptimized(candidate_args);
    ASSERT_TRUE(candidate.ok()) << candidate.ToString();
    const Status reference = cpu::detail::RunGemmF32Reference(reference_args);
    ASSERT_TRUE(reference.ok()) << reference.ToString();

    for (int64_t row = 0; row < candidate_args.m; ++row) {
        for (int64_t col = 0; col < candidate_args.n; ++col) {
            const float actual = candidate_args.output[row * candidate_args.output_m_stride +
                                                       col * candidate_args.output_n_stride];
            const float expected = reference_args.output[row * reference_args.output_m_stride +
                                                         col * reference_args.output_n_stride];
            const float absolute_error = std::fabs(actual - expected);
            const float relative_error = absolute_error / std::max(std::fabs(expected), 1.0e-6F);
            EXPECT_TRUE(absolute_error <= 1.0e-4F || relative_error <= 1.0e-4F)
                    << "row=" << row << " col=" << col << " actual=" << actual
                    << " expected=" << expected << " absolute_error=" << absolute_error
                    << " relative_error=" << relative_error;
        }
    }
}

struct ScalarFastPathCase {
    RhsLayout layout;
    int64_t k;
    int64_t n;
};

std::string ScalarFastPathCaseName(const testing::TestParamInfo<ScalarFastPathCase>& info) {
    return std::string{info.param.layout == RhsLayout::kNContiguous ? "NContiguous" : "KContiguous"} +
           "K" + std::to_string(info.param.k) + "N" + std::to_string(info.param.n);
}

class CPUKernelGemmScalarFastPathTest : public testing::TestWithParam<ScalarFastPathCase> {};

TEST_P(CPUKernelGemmScalarFastPathTest, MatchesDoubleReferenceWithinF32ErrorBudget) {
    const ScalarFastPathCase test_case = GetParam();
    std::vector<float> lhs(static_cast<size_t>(test_case.k));
    std::vector<float> rhs(static_cast<size_t>(test_case.k * test_case.n));
    std::vector<float> candidate_output(static_cast<size_t>(test_case.n), 31.0F);
    std::vector<float> reference_output(static_cast<size_t>(test_case.n), -17.0F);
    FillValues(lhs);
    FillValues(rhs);

    const int64_t rhs_k_stride = test_case.layout == RhsLayout::kNContiguous ? test_case.n : 1;
    const int64_t rhs_n_stride = test_case.layout == RhsLayout::kNContiguous ? 1 : test_case.k;
    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = 1,
            .n = test_case.n,
            .k = test_case.k,
            .lhs_m_stride = test_case.k,
            .lhs_k_stride = 1,
            .rhs_k_stride = rhs_k_stride,
            .rhs_n_stride = rhs_n_stride,
            .output_m_stride = test_case.n,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_output.data();
    ExpectCandidateNearReference(candidate_args, reference_args);
}

INSTANTIATE_TEST_SUITE_P(
        FastPathBoundaries,
        CPUKernelGemmScalarFastPathTest,
        testing::Values(
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 1},
                ScalarFastPathCase{RhsLayout::kNContiguous, 31, 2},
                ScalarFastPathCase{RhsLayout::kNContiguous, 32, 3},
                ScalarFastPathCase{RhsLayout::kNContiguous, 33, 4},
                ScalarFastPathCase{RhsLayout::kNContiguous, 127, 5},
                ScalarFastPathCase{RhsLayout::kNContiguous, 128, 31},
                ScalarFastPathCase{RhsLayout::kNContiguous, 129, 32},
                ScalarFastPathCase{RhsLayout::kNContiguous, 33, 33},
                ScalarFastPathCase{RhsLayout::kNContiguous, 4096, 5},
                ScalarFastPathCase{RhsLayout::kKContiguous, 1, 1},
                ScalarFastPathCase{RhsLayout::kKContiguous, 31, 2},
                ScalarFastPathCase{RhsLayout::kKContiguous, 32, 3},
                ScalarFastPathCase{RhsLayout::kKContiguous, 33, 4},
                ScalarFastPathCase{RhsLayout::kKContiguous, 127, 5},
                ScalarFastPathCase{RhsLayout::kKContiguous, 128, 31},
                ScalarFastPathCase{RhsLayout::kKContiguous, 129, 32},
                ScalarFastPathCase{RhsLayout::kKContiguous, 33, 33},
                ScalarFastPathCase{RhsLayout::kKContiguous, 4096, 5}),
        ScalarFastPathCaseName);

TEST(CPUKernelGemmScalar, KContiguousPaddedAndUnalignedMatchesReference) {
    constexpr int64_t k = 5;
    constexpr int64_t n = 5;
    constexpr int64_t weight_row_stride = 8;
    std::array<float, 8> lhs_storage{};
    std::array<float, n * weight_row_stride + 1> rhs_storage{};
    std::array<float, 8> candidate_storage{};
    std::array<float, 8> reference_storage{};
    for (size_t index = 0; index < lhs_storage.size(); ++index) {
        lhs_storage[index] = TestValue(index);
    }
    for (size_t index = 0; index < rhs_storage.size(); ++index) {
        rhs_storage[index] = TestValue(index + 19U);
    }
    candidate_storage.fill(19.0F);
    reference_storage.fill(-23.0F);

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
    ExpectCandidateNearReference(candidate_args, reference_args);
    EXPECT_EQ(candidate_storage.front(), 19.0F);
    EXPECT_EQ(reference_storage.front(), -23.0F);
}

TEST(CPUKernelGemmScalar, NContiguousPaddedMatchesReference) {
    constexpr int64_t k = 5;
    constexpr int64_t n = 5;
    constexpr int64_t rhs_k_stride = 8;
    std::array<float, k> lhs{};
    std::array<float, k * rhs_k_stride> rhs{};
    std::array<float, n> candidate_output{};
    std::array<float, n> reference_output{};
    for (size_t index = 0; index < lhs.size(); ++index) {
        lhs[index] = TestValue(index);
    }
    for (size_t index = 0; index < rhs.size(); ++index) {
        rhs[index] = TestValue(index + 31U);
    }

    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = 1,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = rhs_k_stride,
            .rhs_n_stride = 1,
            .output_m_stride = n,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_output.data();
    ExpectCandidateNearReference(candidate_args, reference_args);
}

TEST(CPUKernelGemmScalar, FallsBackToReferenceForMNotOneAndNonUnitStrides) {
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    constexpr int64_t k = 3;
    std::array<float, 18> lhs{};
    std::array<float, 28> rhs{};
    std::array<float, 20> candidate_output{};
    std::array<float, 20> reference_output{};
    for (size_t index = 0; index < lhs.size(); ++index) {
        lhs[index] = TestValue(index);
    }
    for (size_t index = 0; index < rhs.size(); ++index) {
        rhs[index] = TestValue(index + 7U);
    }

    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = 9,
            .lhs_k_stride = 2,
            .rhs_k_stride = 7,
            .rhs_n_stride = 2,
            .output_m_stride = 10,
            .output_n_stride = 3,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_output.data();
    ExpectCandidateNearReference(candidate_args, reference_args);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            EXPECT_EQ(candidate_args.output[row * candidate_args.output_m_stride +
                                            col * candidate_args.output_n_stride],
                      reference_args.output[row * reference_args.output_m_stride +
                                            col * reference_args.output_n_stride]);
        }
    }
}

TEST(CPUKernelGemmScalar, M1NonUnitOutputStrideFallsBackToReference) {
    constexpr int64_t n = 3;
    constexpr int64_t k = 3;
    constexpr std::array<float, k> lhs = {1.0F, -2.0F, 0.5F};
    constexpr std::array<float, n * k> rhs = {
            2.0F,
            1.0F,
            -1.0F,
            -0.5F,
            3.0F,
            4.0F,
            1.5F,
            -2.0F,
            0.25F,
    };
    std::array<float, 9> candidate_output{};
    std::array<float, 9> reference_output{};
    const cpu::detail::GemmF32Args candidate_args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = candidate_output.data(),
            .m = 1,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = k,
            .output_m_stride = n * 3,
            .output_n_stride = 3,
    };
    auto reference_args = candidate_args;
    reference_args.output = reference_output.data();
    ExpectCandidateNearReference(candidate_args, reference_args);
    for (int64_t col = 0; col < n; ++col) {
        EXPECT_EQ(candidate_output[static_cast<size_t>(col * candidate_args.output_n_stride)],
                  reference_output[static_cast<size_t>(col * reference_args.output_n_stride)]);
    }
}

TEST(CPUKernelGemmScalar, ZeroInnerDimensionWritesPositiveZeroWithoutInputs) {
    std::array<float, 8> output{};
    output.fill(-7.0F);
    const Status status = cpu::detail::RunGemmF32ScalarOptimized(cpu::detail::GemmF32Args{
            .output = output.data(),
            .m = 1,
            .n = 3,
            .k = 0,
            .output_m_stride = 5,
            .output_n_stride = 1,
    });
    ASSERT_TRUE(status.ok()) << status.ToString();
    for (int64_t col = 0; col < 3; ++col) {
        EXPECT_EQ(output[static_cast<size_t>(col)], 0.0F);
        EXPECT_FALSE(std::signbit(output[static_cast<size_t>(col)]));
    }
}

TEST(CPUKernelGemmScalar, OverwritesExistingOutput) {
    constexpr std::array<float, 3> lhs = {1.0F, -2.0F, 0.5F};
    constexpr std::array<float, 6> weights = {
            2.0F,
            1.0F,
            -1.0F,
            -0.5F,
            3.0F,
            4.0F,
    };
    std::array<float, 2> output = {101.0F, -101.0F};
    const Status status = cpu::detail::RunGemmF32ScalarOptimized(cpu::detail::GemmF32Args{
            .lhs = lhs.data(),
            .rhs = weights.data(),
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
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], -0.5F);
    EXPECT_FLOAT_EQ(output[1], -4.5F);
}

} // namespace
