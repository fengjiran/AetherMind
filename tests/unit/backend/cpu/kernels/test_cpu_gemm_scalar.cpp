#include "backend/cpu/kernels/gemm/gemm_internal.h"
#include "inference/test_malloc_interposer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
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

void ExpectCandidateNearReference(const cpu::detail::GemmArgs& candidate_args,
                                  const cpu::detail::GemmArgs& reference_args) {
    const Status candidate = cpu::detail::RunGemmF32Scalar(candidate_args);
    ASSERT_TRUE(candidate.ok()) << candidate.ToString();
    const Status reference = cpu::detail::RunGemmF32Reference(reference_args);
    ASSERT_TRUE(reference.ok()) << reference.ToString();

    for (int64_t row = 0; row < candidate_args.m; ++row) {
        for (int64_t col = 0; col < candidate_args.n; ++col) {
            const float actual = reinterpret_cast<const float*>(candidate_args.output)[row * candidate_args.output_m_stride +
                                                                                       col * candidate_args.output_n_stride];
            const float expected = reinterpret_cast<const float*>(reference_args.output)[row * reference_args.output_m_stride +
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
    int64_t m;
    int64_t k;
    int64_t n;
};

std::string ScalarFastPathCaseName(const testing::TestParamInfo<ScalarFastPathCase>& info) {
    return std::string{info.param.layout == RhsLayout::kNContiguous ? "NContiguous" : "KContiguous"} +
           "M" + std::to_string(info.param.m) + "K" + std::to_string(info.param.k) + "N" +
           std::to_string(info.param.n);
}

class CPUKernelGemmScalarFastPathTest : public testing::TestWithParam<ScalarFastPathCase> {};

TEST_P(CPUKernelGemmScalarFastPathTest, MatchesDoubleReferenceWithinF32ErrorBudget) {
    const ScalarFastPathCase test_case = GetParam();
    const int64_t lhs_m_stride = test_case.k + 3;
    const int64_t rhs_k_stride =
            test_case.layout == RhsLayout::kNContiguous ? test_case.n + 5 : 1;
    const int64_t rhs_n_stride =
            test_case.layout == RhsLayout::kNContiguous ? 1 : test_case.k + 5;
    const int64_t output_m_stride = test_case.n + 4;
    std::vector<float> lhs(static_cast<size_t>(test_case.m * lhs_m_stride));
    std::vector<float> rhs(static_cast<size_t>(test_case.layout == RhsLayout::kNContiguous
                                                       ? test_case.k * rhs_k_stride
                                                       : test_case.n * rhs_n_stride));
    std::vector<float> candidate_output(static_cast<size_t>(test_case.m * output_m_stride), 31.0F);
    std::vector<float> reference_output(static_cast<size_t>(test_case.m * output_m_stride), -17.0F);
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
            .rhs_k_stride = rhs_k_stride,
            .rhs_n_stride = rhs_n_stride,
            .output_m_stride = output_m_stride,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reinterpret_cast<std::byte*>(reference_output.data());
    ExpectCandidateNearReference(candidate_args, reference_args);
    for (int64_t row = 0; row < test_case.m; ++row) {
        for (int64_t col = test_case.n; col < output_m_stride; ++col) {
            EXPECT_EQ(candidate_output[static_cast<size_t>(row * output_m_stride + col)], 31.0F)
                    << "row=" << row << " padding_col=" << col;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
        FastPathBoundaries,
        CPUKernelGemmScalarFastPathTest,
        testing::Values(
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 1, 1},
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 31, 5},
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 32, 31},
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 33, 33},
                ScalarFastPathCase{RhsLayout::kNContiguous, 2, 31, 2},
                ScalarFastPathCase{RhsLayout::kNContiguous, 3, 32, 3},
                ScalarFastPathCase{RhsLayout::kNContiguous, 4, 33, 4},
                ScalarFastPathCase{RhsLayout::kNContiguous, 7, 127, 5},
                ScalarFastPathCase{RhsLayout::kNContiguous, 8, 128, 31},
                ScalarFastPathCase{RhsLayout::kNContiguous, 10, 33, 5},
                ScalarFastPathCase{RhsLayout::kNContiguous, 11, 33, 5},
                ScalarFastPathCase{RhsLayout::kNContiguous, 16, 129, 32},
                ScalarFastPathCase{RhsLayout::kNContiguous, 17, 33, 33},
                ScalarFastPathCase{RhsLayout::kNContiguous, 17, 4096, 5},
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 129, 63},
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 257, 64},
                ScalarFastPathCase{RhsLayout::kNContiguous, 1, 4097, 65},
                ScalarFastPathCase{RhsLayout::kNContiguous, 2, 129, 127},
                ScalarFastPathCase{RhsLayout::kNContiguous, 3, 257, 128},
                ScalarFastPathCase{RhsLayout::kNContiguous, 4, 513, 129},
                ScalarFastPathCase{RhsLayout::kNContiguous, 7, 127, 65},
                ScalarFastPathCase{RhsLayout::kNContiguous, 8, 128, 64},
                ScalarFastPathCase{RhsLayout::kNContiguous, 9, 129, 65},
                ScalarFastPathCase{RhsLayout::kNContiguous, 10, 257, 127},
                ScalarFastPathCase{RhsLayout::kNContiguous, 11, 255, 128},
                ScalarFastPathCase{RhsLayout::kNContiguous, 16, 257, 129},
                ScalarFastPathCase{RhsLayout::kNContiguous, 17, 129, 65},
                ScalarFastPathCase{RhsLayout::kNContiguous, 2, 32, 33},
                ScalarFastPathCase{RhsLayout::kNContiguous, 3, 128, 59},
                ScalarFastPathCase{RhsLayout::kNContiguous, 3, 129, 59},
                ScalarFastPathCase{RhsLayout::kNContiguous, 3, 128, 60},
                ScalarFastPathCase{RhsLayout::kNContiguous, 9, 128, 59},
                ScalarFastPathCase{RhsLayout::kKContiguous, 1, 1, 1},
                ScalarFastPathCase{RhsLayout::kKContiguous, 1, 31, 5},
                ScalarFastPathCase{RhsLayout::kKContiguous, 1, 32, 31},
                ScalarFastPathCase{RhsLayout::kKContiguous, 1, 33, 33},
                ScalarFastPathCase{RhsLayout::kKContiguous, 2, 31, 2},
                ScalarFastPathCase{RhsLayout::kKContiguous, 3, 32, 3},
                ScalarFastPathCase{RhsLayout::kKContiguous, 4, 33, 4},
                ScalarFastPathCase{RhsLayout::kKContiguous, 7, 127, 5},
                ScalarFastPathCase{RhsLayout::kKContiguous, 8, 128, 31},
                ScalarFastPathCase{RhsLayout::kKContiguous, 10, 33, 5},
                ScalarFastPathCase{RhsLayout::kKContiguous, 11, 33, 5},
                ScalarFastPathCase{RhsLayout::kKContiguous, 16, 129, 32},
                ScalarFastPathCase{RhsLayout::kKContiguous, 17, 33, 33},
                ScalarFastPathCase{RhsLayout::kKContiguous, 17, 4096, 5}),
        ScalarFastPathCaseName);

TEST(CPUKernelGemmScalar, KContiguousMultiRowPaddedAndUnalignedMatchesReference) {
    constexpr int64_t m = 3;
    constexpr int64_t k = 5;
    constexpr int64_t n = 5;
    constexpr int64_t lhs_m_stride = 7;
    constexpr int64_t weight_row_stride = 8;
    constexpr int64_t output_m_stride = 7;
    std::array<float, 21> lhs_storage{};
    std::array<float, n * weight_row_stride + 1> rhs_storage{};
    std::array<float, 22> candidate_storage{};
    std::array<float, 22> reference_storage{};
    for (size_t index = 0; index < lhs_storage.size(); ++index) {
        lhs_storage[index] = TestValue(index);
    }
    for (size_t index = 0; index < rhs_storage.size(); ++index) {
        rhs_storage[index] = TestValue(index + 19U);
    }
    candidate_storage.fill(19.0F);
    reference_storage.fill(-23.0F);

    const cpu::detail::GemmArgs candidate_args{
            .lhs = reinterpret_cast<const std::byte*>(lhs_storage.data() + 1),
            .rhs = reinterpret_cast<const std::byte*>(rhs_storage.data() + 1),
            .output = reinterpret_cast<std::byte*>(candidate_storage.data() + 1),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = lhs_m_stride,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = weight_row_stride,
            .output_m_stride = output_m_stride,
            .output_n_stride = 1,
    };
    auto reference_args = candidate_args;
    reference_args.output = reinterpret_cast<std::byte*>(reference_storage.data() + 1);
    ExpectCandidateNearReference(candidate_args, reference_args);
    EXPECT_EQ(candidate_storage.front(), 19.0F);
    EXPECT_EQ(reference_storage.front(), -23.0F);
}

TEST(CPUKernelGemmScalar, NContiguousMultiRowPaddedAndUnalignedMatchesReference) {
    constexpr int64_t m = 3;
    constexpr int64_t k = 5;
    constexpr int64_t n = 5;
    constexpr int64_t lhs_m_stride = 7;
    constexpr int64_t rhs_k_stride = 8;
    constexpr int64_t output_m_stride = 7;
    std::array<float, 21> lhs_storage{};
    std::array<float, k * rhs_k_stride + 1> rhs_storage{};
    std::array<float, 22> candidate_storage{};
    std::array<float, 22> reference_storage{};
    for (size_t index = 0; index < lhs_storage.size(); ++index) {
        lhs_storage[index] = TestValue(index);
    }
    for (size_t index = 0; index < rhs_storage.size(); ++index) {
        rhs_storage[index] = TestValue(index + 31U);
    }

    const cpu::detail::GemmArgs candidate_args{
            .lhs = reinterpret_cast<const std::byte*>(lhs_storage.data() + 1),
            .rhs = reinterpret_cast<const std::byte*>(rhs_storage.data() + 1),
            .output = reinterpret_cast<std::byte*>(candidate_storage.data() + 1),
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
    reference_args.output = reinterpret_cast<std::byte*>(reference_storage.data() + 1);
    ExpectCandidateNearReference(candidate_args, reference_args);
}

TEST(CPUKernelGemmScalar, NContiguousBlocksOverwriteNanOutputAndPreserveUnalignedGuards) {
    constexpr int64_t m = 11;
    constexpr int64_t k = 513;
    constexpr int64_t n = 129;
    constexpr int64_t lhs_m_stride = k + 3;
    constexpr int64_t rhs_k_stride = n + 13;
    constexpr int64_t output_m_stride = n + 7;
    constexpr float guard = 937.25F;
    std::vector<float> lhs(static_cast<size_t>(m * lhs_m_stride + 2), guard);
    std::vector<float> rhs(static_cast<size_t>(k * rhs_k_stride + 2), guard);
    std::vector<float> output(static_cast<size_t>(m * output_m_stride + 2), guard);
    auto reference_output = output;
    std::mt19937 random(20261117);
    std::uniform_real_distribution<float> distribution(-0.75F, 0.75F);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t inner = 0; inner < k; ++inner) {
            lhs[static_cast<size_t>(1 + row * lhs_m_stride + inner)] = distribution(random);
        }
        for (int64_t col = 0; col < n; ++col) {
            output[static_cast<size_t>(1 + row * output_m_stride + col)] =
                    std::numeric_limits<float>::quiet_NaN();
        }
    }
    for (int64_t inner = 0; inner < k; ++inner) {
        for (int64_t col = 0; col < n; ++col) {
            rhs[static_cast<size_t>(1 + inner * rhs_k_stride + col)] = distribution(random);
        }
    }
    const auto original_lhs = lhs;
    const auto original_rhs = rhs;
    const cpu::detail::GemmArgs args{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data() + 1),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data() + 1),
            .output = reinterpret_cast<std::byte*>(output.data() + 1),
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
    auto reference_args = args;
    reference_args.output = reinterpret_cast<std::byte*>(reference_output.data() + 1);
    ExpectCandidateNearReference(args, reference_args);
    EXPECT_EQ(lhs, original_lhs);
    EXPECT_EQ(rhs, original_rhs);
    EXPECT_EQ(output.front(), guard);
    EXPECT_EQ(output.back(), guard);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = n; col < output_m_stride; ++col) {
            EXPECT_EQ(output[static_cast<size_t>(1 + row * output_m_stride + col)], guard);
        }
    }
    const auto first_output = output;
    ASSERT_TRUE(cpu::detail::RunGemmF32Scalar(args).ok());
    EXPECT_EQ(output, first_output);
}

TEST(CPUKernelGemmScalar, NContiguousPreservesAscendingKAccumulation) {
    constexpr int64_t k = 257;
    constexpr int64_t n = 257;
    std::array<float, k> lhs{};
    lhs.fill(1.0F);
    std::vector<float> rhs(static_cast<size_t>(k * n), 0.0F);
    std::array<float, n> output{};
    output.fill(std::numeric_limits<float>::quiet_NaN());
    for (int64_t col = 0; col < n; ++col) {
        // Independently summing K tiles would lose the +1 at K=129 before
        // cancellation with K=0, whereas carrying the accumulator yields 2.
        rhs[static_cast<size_t>(col)] = 33554432.0F;
        rhs[static_cast<size_t>(128 * n + col)] = -33554432.0F;
        rhs[static_cast<size_t>(129 * n + col)] = 1.0F;
        rhs[static_cast<size_t>(256 * n + col)] = 1.0F;
    }
    const auto status = cpu::detail::RunGemmF32Scalar(cpu::detail::GemmArgs{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
            .output = reinterpret_cast<std::byte*>(output.data()),
            .m = 1,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = n,
            .rhs_n_stride = 1,
            .output_m_stride = n,
            .output_n_stride = 1,
    });
    ASSERT_TRUE(status.ok()) << status.ToString();
    for (float value: output) {
        EXPECT_FLOAT_EQ(value, 2.0F);
    }
}

TEST(CPUKernelGemmScalar, NContiguousBlockedSteadyStateDoesNotAllocate) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "malloc interposition is unavailable in this build";
    }
    constexpr int64_t m = 11;
    constexpr int64_t k = 257;
    constexpr int64_t n = 257;
    std::vector<float> lhs(static_cast<size_t>(m * k), 0.5F);
    std::vector<float> rhs(static_cast<size_t>(k * n), -0.25F);
    std::vector<float> output(static_cast<size_t>(m * n));
    const cpu::detail::GemmArgs args{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
            .output = reinterpret_cast<std::byte*>(output.data()),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = n,
            .rhs_n_stride = 1,
            .output_m_stride = n,
            .output_n_stride = 1,
    };
    ASSERT_TRUE(cpu::detail::RunGemmF32Scalar(args).ok());
    test::BeginMallocCallCounting();
    const auto status = cpu::detail::RunGemmF32Scalar(args);
    const auto counts = test::EndMallocCallCounting();
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(counts.malloc_calls, 0U);
    EXPECT_EQ(counts.calloc_calls, 0U);
    EXPECT_EQ(counts.realloc_calls, 0U);
    EXPECT_EQ(counts.free_calls, 0U);
    EXPECT_EQ(counts.aligned_alloc_calls, 0U);
    EXPECT_EQ(counts.posix_memalign_calls, 0U);
    EXPECT_EQ(counts.memalign_calls, 0U);
}

TEST(CPUKernelGemmScalar, FallsBackToReferenceForNonCanonicalStrides) {
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

    const cpu::detail::GemmArgs candidate_args{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
            .output = reinterpret_cast<std::byte*>(candidate_output.data()),
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
    reference_args.output = reinterpret_cast<std::byte*>(reference_output.data());
    ExpectCandidateNearReference(candidate_args, reference_args);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            EXPECT_EQ(reinterpret_cast<const float*>(candidate_args.output)[row * candidate_args.output_m_stride +
                                                                            col * candidate_args.output_n_stride],
                      reinterpret_cast<const float*>(reference_args.output)[row * reference_args.output_m_stride +
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
    const cpu::detail::GemmArgs candidate_args{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(rhs.data()),
            .output = reinterpret_cast<std::byte*>(candidate_output.data()),
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
    reference_args.output = reinterpret_cast<std::byte*>(reference_output.data());
    ExpectCandidateNearReference(candidate_args, reference_args);
    for (int64_t col = 0; col < n; ++col) {
        EXPECT_EQ(candidate_output[static_cast<size_t>(col * candidate_args.output_n_stride)],
                  reference_output[static_cast<size_t>(col * reference_args.output_n_stride)]);
    }
}

TEST(CPUKernelGemmScalar, ZeroInnerDimensionWritesPositiveZeroWithoutInputs) {
    std::array<float, 13> output{};
    output.fill(-7.0F);
    const Status status = cpu::detail::RunGemmF32Scalar(cpu::detail::GemmArgs{
            .output = reinterpret_cast<std::byte*>(output.data()),
            .m = 3,
            .n = 3,
            .k = 0,
            .output_m_stride = 5,
            .output_n_stride = 1,
    });
    ASSERT_TRUE(status.ok()) << status.ToString();
    for (int64_t row = 0; row < 3; ++row) {
        for (int64_t col = 0; col < 3; ++col) {
            const size_t output_index = static_cast<size_t>(row * 5 + col);
            EXPECT_EQ(output[output_index], 0.0F);
            EXPECT_FALSE(std::signbit(output[output_index]));
        }
    }
}

TEST(CPUKernelGemmScalar, ZeroOutputDimensionsAreNoOps) {
    const Status zero_m = cpu::detail::RunGemmF32Scalar(cpu::detail::GemmArgs{
            .m = 0,
            .n = 3,
            .k = 7,
    });
    ASSERT_TRUE(zero_m.ok()) << zero_m.ToString();

    const Status zero_n = cpu::detail::RunGemmF32Scalar(cpu::detail::GemmArgs{
            .m = 3,
            .n = 0,
            .k = 7,
    });
    ASSERT_TRUE(zero_n.ok()) << zero_n.ToString();
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
    const Status status = cpu::detail::RunGemmF32Scalar(cpu::detail::GemmArgs{
            .lhs = reinterpret_cast<const std::byte*>(lhs.data()),
            .rhs = reinterpret_cast<const std::byte*>(weights.data()),
            .output = reinterpret_cast<std::byte*>(output.data()),
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
