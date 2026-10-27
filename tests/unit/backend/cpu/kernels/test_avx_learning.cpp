// AVX2/FMA examples keep their GTest entry points in this baseline translation
// unit. Only the helper implementation is compiled with optional ISA flags.
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/kernels/common/dot_product.h"
#include "avx_learning/test_avx_learning_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

class CPUKernelAvxLearning : public ::testing::Test {
protected:
    void SetUp() override {
#if defined(AETHERMIND_TEST_HAS_AVX2_FMA) && AETHERMIND_TEST_HAS_AVX2_FMA
        const auto capabilities = aethermind::cpu::DetectCpuCapabilities();
        ASSERT_TRUE(capabilities.ok()) << capabilities.status().ToString();
        constexpr auto required = aethermind::CpuFeatureSet::From({aethermind::CpuFeature::kAvx2, aethermind::CpuFeature::kFma});
        if (!capabilities->usable_features.ContainsAll(required) ||
            !capabilities->effective_features.ContainsAll(required)) {
            GTEST_SKIP() << "Host CPU/OS does not support usable AVX2 and FMA";
        }
#else
        GTEST_SKIP() << "AVX2/FMA learning helpers are not compiled in this build";
#endif
    }
};

#if defined(AETHERMIND_TEST_HAS_AVX2_FMA) && AETHERMIND_TEST_HAS_AVX2_FMA
namespace avx_learning = aethermind::test::avx_learning;

void ExpectClose(float actual, float expected, float rel_eps = 1.0e-3F) {
    const float max_abs = std::max({std::abs(actual), std::abs(expected),
                                    std::numeric_limits<float>::min()});
    if (const float abs_err = std::abs(actual - expected); abs_err > rel_eps * max_abs) {
        ADD_FAILURE() << "Expected: " << expected << " (±" << (rel_eps * 100.0F)
                      << "%), actual: " << actual
                      << ", abs_err: " << abs_err
                      << ", rel_err: " << (abs_err / max_abs);
    }
}

/// 标量 add: 逐元素相加，验证 SIMD Load + Add + Store 的正确性。
void ReferenceAdd(const float* a, const float* b, float* out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void ReferenceFMAdd(const float* a, const float* b, const float* c, float* out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}

float ReferenceDotProduct(const float* a, const float* b, std::size_t n) noexcept {
    float result = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

TEST_F(CPUKernelAvxLearning, LoadStoreAdd) {
    alignas(32) constexpr float a[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    alignas(32) constexpr float b[8] = {8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
    alignas(32) float expected[8] = {};
    alignas(32) float actual[8] = {};

    avx_learning::Add(a, b, actual, 8);
    ReferenceAdd(a, b, expected, 8);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, LoadStoreAddTail) {
    alignas(32) constexpr float a[10] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    alignas(32) constexpr float b[10] = {9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F, 0.0F};
    alignas(32) float expected[10] = {};
    alignas(32) float actual[10] = {};

    avx_learning::Add(a, b, actual, 10);
    ReferenceAdd(a, b, expected, 10);

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, Fma) {
    alignas(32) constexpr float a[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    alignas(32) constexpr float b[8] = {2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
    alignas(32) constexpr float c[8] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    alignas(32) float actual[8] = {};
    alignas(32) float expected[8] = {};

    avx_learning::FusedMultiplyAdd(a, b, c, actual, 8);
    ReferenceFMAdd(a, b, c, expected, 8);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, FmaTail) {
    alignas(32) constexpr float a[10] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    alignas(32) constexpr float b[10] = {2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
    alignas(32) constexpr float c[10] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    alignas(32) float actual[10] = {};
    alignas(32) float expected[10] = {};

    avx_learning::FusedMultiplyAdd(a, b, c, actual, 10);
    ReferenceFMAdd(a, b, c, expected, 10);

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, DotProduct) {
    alignas(32) constexpr float a[10] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
    alignas(32) constexpr float b[10] = {10.0F, 9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};

    const float actual = aethermind::cpu::DotProductF32(a, b, 10);
    const float expected = ReferenceDotProduct(a, b, 10);

    EXPECT_FLOAT_EQ(actual, expected);
}

TEST_F(CPUKernelAvxLearning, DotProductLargeVectorAligned32) {
    constexpr size_t kSize = 1024;
    alignas(32) std::vector<float> a(kSize);
    alignas(32) std::vector<float> b(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        a[i] = i * 0.5F;
        b[i] = i * 0.25F;
    }

    const float actual = aethermind::cpu::DotProductF32(a.data(), b.data(), kSize);
    const float expected = ReferenceDotProduct(a.data(), b.data(), kSize);
    ExpectClose(actual, expected);
}

TEST_F(CPUKernelAvxLearning, DotProductVeryLargeVector) {
    constexpr size_t kSize = 1 << 20; // 1,048,576 elements
    std::vector<float> a(kSize);
    std::vector<float> b(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        a[i] = static_cast<float>(i & 0xFF) * 0.1F;
        b[i] = static_cast<float>(255 - (i & 0xFF)) * 0.1F;
    }

    const float actual = aethermind::cpu::DotProductF32(a.data(), b.data(), kSize);
    const float expected = ReferenceDotProduct(a.data(), b.data(), kSize);
    ExpectClose(actual, expected);
}

TEST_F(CPUKernelAvxLearning, Broadcast) {
    alignas(32) float simd_out[8] = {};

    avx_learning::Broadcast8(3.14F, simd_out);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], 3.14F);
    }
}

TEST_F(CPUKernelAvxLearning, HorizontalSum) {
    alignas(32) const float data[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};

    const float simd_sum = avx_learning::HorizontalSum8(data);

    // 标量 reference
    float ref_sum = 0.0F;
    for (int i = 0; i < 8; ++i) {
        ref_sum += data[i];
    }

    EXPECT_FLOAT_EQ(simd_sum, ref_sum);
}

TEST_F(CPUKernelAvxLearning, HorizontalMax) {
    alignas(32) const float data[8] = {1.0F, 2.0F, 15.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};

    const float simd_max = avx_learning::HorizontalMax8(data);

    float ref_max = data[0];
    for (int i = 1; i < 8; ++i) {
        if (data[i] > ref_max) ref_max = data[i];
    }

    EXPECT_FLOAT_EQ(simd_max, ref_max);
}

TEST_F(CPUKernelAvxLearning, ShuffleWithinLane) {
    alignas(32) const float data[8] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    alignas(32) float simd_out[8] = {};

    avx_learning::ShuffleWithinLane8(data, simd_out);

    EXPECT_FLOAT_EQ(simd_out[0], 1.0F); // dst[0] = v[1]
    EXPECT_FLOAT_EQ(simd_out[1], 0.0F); // dst[1] = v[0]
    EXPECT_FLOAT_EQ(simd_out[2], 3.0F); // dst[2] = v[3]
    EXPECT_FLOAT_EQ(simd_out[3], 2.0F); // dst[3] = v[2]
    EXPECT_FLOAT_EQ(simd_out[4], 5.0F); // lane 1: dst[0] = v[5]
    EXPECT_FLOAT_EQ(simd_out[5], 4.0F); // dst[1] = v[4]
    EXPECT_FLOAT_EQ(simd_out[6], 7.0F); // dst[2] = v[7]
    EXPECT_FLOAT_EQ(simd_out[7], 6.0F); // dst[3] = v[6]
}

TEST_F(CPUKernelAvxLearning, CrossLanePermute) {
    alignas(32) const float data[8] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    alignas(32) float simd_out[8] = {};

    // 将 [0..7] 按索引 [7,6,5,4,3,2,1,0] 逆序
    alignas(32) const int32_t indices[8] = {7, 6, 5, 4, 3, 2, 1, 0};
    avx_learning::Permute8(data, indices, simd_out);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], 7.0F - i) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, CompareBlend) {
    alignas(32) const float data[8] = {1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F, 7.0F, -8.0F};
    alignas(32) float simd_out[8] = {};
    alignas(32) float ref_out[8] = {};

    avx_learning::ClampNegativeToZero8(data, simd_out);

    // 标量 reference
    for (int i = 0; i < 8; ++i) {
        ref_out[i] = (data[i] < 0.0F) ? 0.0F : data[i];
    }

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], ref_out[i]) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, RmsNormSumSq) {
    alignas(32) const float row[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};

    const float simd_sum_sq = avx_learning::SumSquares8(row);

    // 标量 reference
    float ref_sum_sq = 0.0F;
    for (int i = 0; i < 8; ++i) {
        ref_sum_sq += row[i] * row[i];
    }

    EXPECT_FLOAT_EQ(simd_sum_sq, ref_sum_sq);
}

TEST_F(CPUKernelAvxLearning, RmsNormScale) {
    alignas(32) const float row[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    alignas(32) const float weight[8] = {1.0F, 0.5F, 1.5F, 2.0F, 1.0F, 0.5F, 1.5F, 2.0F};
    alignas(32) float simd_out[8] = {};
    alignas(32) float ref_out[8] = {};
    constexpr float inv_rms = 0.730297F; // 假设值，来自 row {1,2,3,4} 的 rms_norm

    avx_learning::ScaleRmsNorm8(row, weight, inv_rms, simd_out);

    // 标量 reference
    for (int i = 0; i < 8; ++i) {
        ref_out[i] = row[i] * inv_rms * weight[i];
    }

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], ref_out[i]) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, Gather) {
    alignas(32) const float table[16] = {
            0.0F,
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
            12.0F,
            13.0F,
            14.0F,
            15.0F,
    };
    alignas(16) const int32_t idx[8] = {3, 7, 1, 14, 0, 9, 5, 11};
    alignas(32) float simd_out[8] = {};
    alignas(32) float ref_out[8] = {};

    avx_learning::Gather8(table, idx, simd_out);

    // 标量 reference
    for (int i = 0; i < 8; ++i) {
        ref_out[i] = table[idx[i]];
    }

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], ref_out[i]) << "index " << i;
    }
}

TEST_F(CPUKernelAvxLearning, TailHandling) {
    // 假设 hidden_size = 10, 需要 8 + 2 两段处理
    alignas(32) const float data[10] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F,
                                        5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    float sum_simd[10] = {}; // 初始化为 0
    float sum_ref[10] = {};

    avx_learning::SquareWithScalarTail(data, sum_simd, 10);
    float tail_buf[4] = {};
    avx_learning::MaskedLoadTwo(data + 8, tail_buf);
    EXPECT_FLOAT_EQ(tail_buf[0], data[8]);
    EXPECT_FLOAT_EQ(tail_buf[1], data[9]);
    EXPECT_FLOAT_EQ(tail_buf[2], 0.0F);
    EXPECT_FLOAT_EQ(tail_buf[3], 0.0F);

    // 标量 reference
    for (int i = 0; i < 10; ++i) {
        sum_ref[i] = data[i] * data[i];
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(sum_simd[i], sum_ref[i]) << "index " << i;
    }
}

#else
// Keep discovery stable when optional ISA implementation is absent.
TEST_F(CPUKernelAvxLearning, LoadStoreAdd) {}
TEST_F(CPUKernelAvxLearning, LoadStoreAddTail) {}
TEST_F(CPUKernelAvxLearning, Fma) {}
TEST_F(CPUKernelAvxLearning, FmaTail) {}
TEST_F(CPUKernelAvxLearning, DotProduct) {}
TEST_F(CPUKernelAvxLearning, DotProductLargeVectorAligned32) {}
TEST_F(CPUKernelAvxLearning, DotProductVeryLargeVector) {}
TEST_F(CPUKernelAvxLearning, Broadcast) {}
TEST_F(CPUKernelAvxLearning, HorizontalSum) {}
TEST_F(CPUKernelAvxLearning, HorizontalMax) {}
TEST_F(CPUKernelAvxLearning, ShuffleWithinLane) {}
TEST_F(CPUKernelAvxLearning, CrossLanePermute) {}
TEST_F(CPUKernelAvxLearning, CompareBlend) {}
TEST_F(CPUKernelAvxLearning, RmsNormSumSq) {}
TEST_F(CPUKernelAvxLearning, RmsNormScale) {}
TEST_F(CPUKernelAvxLearning, Gather) {}
TEST_F(CPUKernelAvxLearning, TailHandling) {}
#endif

} // namespace
