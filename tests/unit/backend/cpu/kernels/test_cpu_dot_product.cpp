#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/kernels/common/dot_product.h"
#include "backend/cpu/kernels/common/dot_product_internal.h"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

using aethermind::cpu::DetectCpuCapabilities;
using aethermind::cpu::DotProductF32;
using aethermind::cpu::DotProductF32Scalar;

std::vector<float> MakeValues(std::size_t n) {
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<float>(static_cast<int>(i * 17U + 5U) % 101 - 50) * 0.125F;
    }
    return values;
}

double ReferenceDotProductF64(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

void ExpectNearGolden(const std::vector<float>& a, const std::vector<float>& b, float actual) {
    const double golden = ReferenceDotProductF64(a, b);
    const double tolerance = 1.0e-4 * std::abs(golden) + 1.0e-4;
    EXPECT_NEAR(static_cast<double>(actual), golden, tolerance);
}

TEST(CPUKernelDotProduct, MatchesDoubleGoldenAcrossBlockBoundaries) {
    const std::size_t sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1024, 65536};
    for (const std::size_t n: sizes) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const std::vector<float> a = MakeValues(n);
        const std::vector<float> b = MakeValues(n);
        const float actual = DotProductF32(a.data(), b.data(), n);
        ExpectNearGolden(a, b, actual);
    }
}

TEST(CPUKernelDotProduct, ZeroLengthReturnsPlusZero) {
    const float actual = DotProductF32(static_cast<const float*>(nullptr),
                                       static_cast<const float*>(nullptr), 0);
    EXPECT_EQ(actual, +0.0F);
}

TEST(CPUKernelDotProduct, HandlesUnalignedPointers) {
    constexpr std::size_t kSize = 1000;
    std::vector<float> a_storage(kSize + 8);
    std::vector<float> b_storage(kSize + 8);
    for (std::size_t i = 0; i < kSize + 8; ++i) {
        a_storage[i] = static_cast<float>(static_cast<int>(i * 3U + 1U) % 97 - 48) * 0.25F;
        b_storage[i] = static_cast<float>(static_cast<int>(i * 11U + 7U) % 89 - 44) * 0.125F;
    }
    for (std::size_t offset = 0; offset <= 3; ++offset) {
        SCOPED_TRACE("offset=" + std::to_string(offset));
        std::vector<float> a(a_storage.begin() + static_cast<std::ptrdiff_t>(offset),
                             a_storage.begin() + static_cast<std::ptrdiff_t>(offset + kSize));
        std::vector<float> b(b_storage.begin() + static_cast<std::ptrdiff_t>(offset),
                             b_storage.begin() + static_cast<std::ptrdiff_t>(offset + kSize));
        const float actual = DotProductF32(a.data(), b.data(), kSize);
        ExpectNearGolden(a, b, actual);
    }
}

TEST(CPUKernelDotProduct, ScalarCoreMatchesGolden) {
    constexpr std::size_t kSize = 1000;
    const std::vector<float> a = MakeValues(kSize);
    const std::vector<float> b = MakeValues(kSize);
    const float actual = DotProductF32Scalar(a.data(), b.data(), kSize);
    ExpectNearGolden(a, b, actual);
}

#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
TEST(CPUKernelDotProduct, Avx2CoreMatchesGolden) {
    constexpr std::size_t kSize = 1000;
    const std::vector<float> a = MakeValues(kSize);
    const std::vector<float> b = MakeValues(kSize);
    const float actual = aethermind::cpu::DotProductF32Avx2(a.data(), b.data(), kSize);
    ExpectNearGolden(a, b, actual);
}
#endif

TEST(CPUKernelDotProduct, DispatchMatchesSelectedCore) {
    constexpr std::size_t kSize = 100;
    const std::vector<float> a = MakeValues(kSize);
    const std::vector<float> b = MakeValues(kSize);
    const auto capabilities = DetectCpuCapabilities();
    ASSERT_TRUE(capabilities.ok());
#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
    const bool expect_avx2 = capabilities->effective_features.Contains(aethermind::CpuFeature::kAvx2) &&
                             capabilities->effective_features.Contains(aethermind::CpuFeature::kFma);
    if (expect_avx2) {
        EXPECT_EQ(DotProductF32(a.data(), b.data(), kSize),
                  aethermind::cpu::DotProductF32Avx2(a.data(), b.data(), kSize));
    } else {
        EXPECT_EQ(DotProductF32(a.data(), b.data(), kSize),
                  DotProductF32Scalar(a.data(), b.data(), kSize));
    }
#else
    EXPECT_EQ(DotProductF32(a.data(), b.data(), kSize),
              DotProductF32Scalar(a.data(), b.data(), kSize));
#endif
}

// Multi-target: out[t] = sum_k lhs[k] * rhs[t * rhs_stride + k].
namespace {

std::vector<float> MakeMatrix(std::size_t num_targets, std::size_t stride, std::size_t n) {
    std::vector<float> matrix(num_targets * stride + n);
    for (std::size_t t = 0; t < num_targets; ++t) {
        for (std::size_t k = 0; k < n; ++k) {
            matrix[t * stride + k] = static_cast<float>(static_cast<int>((t + 1U) * 19U + k * 7U) % 89 - 44) * 0.25F;
        }
    }
    return matrix;
}

void ExpectMultiNearGolden(const std::vector<float>& lhs, const std::vector<float>& rhs,
                           std::size_t n, std::size_t num_targets, std::size_t rhs_stride,
                           const std::vector<float>& actual) {
    ASSERT_EQ(actual.size(), num_targets);
    for (std::size_t t = 0; t < num_targets; ++t) {
        double golden = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            golden += static_cast<double>(lhs[k]) * static_cast<double>(rhs[t * rhs_stride + k]);
        }
        const double tolerance = 1.0e-4 * std::abs(golden) + 1.0e-4;
        EXPECT_NEAR(static_cast<double>(actual[t]), golden, tolerance) << "t=" << t;
    }
}

void ExpectMultiDispatchMatchesCore(const std::vector<float>& lhs, const std::vector<float>& rhs,
                                    std::size_t n, std::size_t num_targets, std::size_t rhs_stride) {
    const auto capabilities = DetectCpuCapabilities();
    ASSERT_TRUE(capabilities.ok());
    std::vector<float> dispatched(num_targets);
    aethermind::cpu::DotProductF32MultiTarget(lhs.data(), rhs.data(), n, num_targets,
                                              rhs_stride, dispatched.data());
#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
    const bool expect_avx2 = capabilities->effective_features.Contains(aethermind::CpuFeature::kAvx2) &&
                             capabilities->effective_features.Contains(aethermind::CpuFeature::kFma);
    std::vector<float> core(num_targets);
    if (expect_avx2) {
        aethermind::cpu::DotProductF32MultiTargetAvx2(lhs.data(), rhs.data(), n, num_targets,
                                                      rhs_stride, core.data());
    } else {
        aethermind::cpu::DotProductF32MultiTargetScalar(lhs.data(), rhs.data(), n, num_targets,
                                                        rhs_stride, core.data());
    }
#else
    std::vector<float> core(num_targets);
    aethermind::cpu::DotProductF32MultiTargetScalar(lhs.data(), rhs.data(), n, num_targets,
                                                    rhs_stride, core.data());
#endif
    EXPECT_EQ(dispatched, core);
}

} // namespace

TEST(CPUKernelDotProduct, MultiTargetMatchesGoldenAcrossShapes) {
    const std::size_t sizes[] = {0, 1, 8, 15, 16, 31, 33, 64, 65, 1024};
    const std::size_t target_counts[] = {1, 3, 4, 5, 8};
    for (const std::size_t n: sizes) {
        const std::vector<float> lhs = MakeValues(n);
        for (const std::size_t num_targets: target_counts) {
            for (const std::size_t stride: {n, n + 3}) {
                SCOPED_TRACE("n=" + std::to_string(n) + " targets=" + std::to_string(num_targets) +
                             " stride=" + std::to_string(stride));
                const std::vector<float> rhs = MakeMatrix(num_targets, stride, n);
                std::vector<float> actual(num_targets);
                aethermind::cpu::DotProductF32MultiTarget(lhs.data(), rhs.data(), n, num_targets,
                                                          stride, actual.data());
                ExpectMultiNearGolden(lhs, rhs, n, num_targets, stride, actual);
            }
        }
    }
}

TEST(CPUKernelDotProduct, MultiTargetZeroLengthWritesPlusZeroPerTarget) {
    constexpr std::size_t kTargets = 5;
    float dummy = 0.0F;
    std::vector<float> actual(kTargets, -1.0F);
    aethermind::cpu::DotProductF32MultiTarget(&dummy, &dummy, 0, kTargets, 8, actual.data());
    for (const float value: actual) {
        EXPECT_EQ(value, +0.0F);
    }
}

TEST(CPUKernelDotProduct, MultiTargetZeroTargetsIsNoOp) {
    float dummy = 0.0F;
    std::vector<float> out{1.0F, 2.0F, 3.0F};
    aethermind::cpu::DotProductF32MultiTarget(&dummy, &dummy, 16, 0, 16, out.data());
    EXPECT_EQ(out, (std::vector<float>{1.0F, 2.0F, 3.0F}));
}

TEST(CPUKernelDotProduct, MultiTargetScalarCoreMatchesGolden) {
    constexpr std::size_t kSize = 1000;
    constexpr std::size_t kTargets = 6;
    constexpr std::size_t kStride = kSize + 3;
    const std::vector<float> lhs = MakeValues(kSize);
    const std::vector<float> rhs = MakeMatrix(kTargets, kStride, kSize);
    std::vector<float> actual(kTargets);
    aethermind::cpu::DotProductF32MultiTargetScalar(lhs.data(), rhs.data(), kSize, kTargets,
                                                    kStride, actual.data());
    ExpectMultiNearGolden(lhs, rhs, kSize, kTargets, kStride, actual);
}

#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
TEST(CPUKernelDotProduct, MultiTargetAvx2CoreMatchesGolden) {
    constexpr std::size_t kSize = 1000;
    constexpr std::size_t kTargets = 6;
    constexpr std::size_t kStride = kSize + 3;
    const std::vector<float> lhs = MakeValues(kSize);
    const std::vector<float> rhs = MakeMatrix(kTargets, kStride, kSize);
    std::vector<float> actual(kTargets);
    aethermind::cpu::DotProductF32MultiTargetAvx2(lhs.data(), rhs.data(), kSize, kTargets,
                                                  kStride, actual.data());
    ExpectMultiNearGolden(lhs, rhs, kSize, kTargets, kStride, actual);
}
#endif

TEST(CPUKernelDotProduct, MultiTargetDispatchMatchesSelectedCore) {
    constexpr std::size_t kSize = 100;
    constexpr std::size_t kTargets = 7;
    constexpr std::size_t kStride = kSize + 5;
    const std::vector<float> lhs = MakeValues(kSize);
    const std::vector<float> rhs = MakeMatrix(kTargets, kStride, kSize);
    ExpectMultiDispatchMatchesCore(lhs, rhs, kSize, kTargets, kStride);
}

} // namespace