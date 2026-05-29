// ═══════════════════════════════════════════════════════════════════════════
// test_avx_learning.cpp — AVX2 / AVX-512 SIMD 入门学习测试
//
// 设计目标:
//   每个 TEST 只演示一个 SIMD 概念，格式统一为:
//     1. 标量 reference 实现 (语义清晰)
//     2. 等价的 SIMD intrinsic 实现
//     3. EXPECT_NEAR 验证二者数值等价
//
// 编译方式:
//   cmake -DCMAKE_CXX_FLAGS="-mavx2 -mfma" ...
//   或直接在 CMakeLists.txt 中 add_compile_options(-mavx2 -mfma)
//
// 保护方式:
//   编译期 #ifdef 防止在不支持的机器上编译失败；
//   运行时 GTEST_SKIP() 避免在不支持的机器上执行。
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

// ---- SIMD 概念 1: Load / Store ─────────────────────────────────────────
// _mm256_loadu_ps / _mm256_storeu_ps: 从内存加载 / 写回 8 个 float。
// "u" = unaligned; 对齐版本 _mm256_load_ps 要求 32 字节对齐。
#if defined(__AVX2__)
#include <immintrin.h>

namespace aethermind {
namespace {

/// Approximate float equality suitable for reduction results.
///
/// Floating-point addition is not associative, so SIMD and scalar
/// accumulation produce slightly different results for large vectors.
/// This helper checks that the relative error is within a small epsilon
/// (1e-4 by default) rather than requiring exact ULP-level match.
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

void AVX2Add(const float* a, const float* b, float* out, std::size_t n) noexcept {
    constexpr std::size_t kStep = 8;
    std::size_t i = 0;
    while (i + kStep <= n) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vsum = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vsum);
        i += kStep;
    }

    while (i < n) {
        out[i] = a[i] + b[i];
        ++i;
    }
}

void AVX2FMAdd(const float* a, const float* b, const float* c, float* out, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 8;
    while (i + kStep <= n) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vres = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(out + i, vres);
        i += kStep;
    }

    while (i < n) {
        out[i] = a[i] * b[i] + c[i];
        ++i;
    }
}

// Horizontal Sum
float hsum_ps256(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhign = _mm256_extractf128_ps(v, 1);
    __m128 v128 = _mm_add_ps(vhign, vlow);

    v128 = _mm_hadd_ps(v128, v128);
    v128 = _mm_hadd_ps(v128, v128);
    return _mm_cvtss_f32(v128);
}

float dot_product_avx2_unroll(const float* a, const float* b, std::size_t n) noexcept {
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    std::size_t i = 0;
    while (i + 32 <= n) {
        vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), vsum0);
        vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), vsum1);
        vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), vsum2);
        vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), vsum3);
        i += 32;
    }

    __m256 vres = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1), _mm256_add_ps(vsum2, vsum3));

    while (i + 8 <= n) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vres = _mm256_add_ps(vres, _mm256_mul_ps(va, vb));
        i += 8;
    }

    float res = hsum_ps256(vres);
    while (i < n) {
        res += a[i] * b[i];
        ++i;
    }

    return res;
}

}// namespace
}// namespace aethermind
#endif// __AVX2__

// =========================================================================
// 1. Load / Store — 加载 8 个 float，做向量加法，再写回
// =========================================================================
#if defined(__AVX2__)
TEST(AvxLearning, LoadStoreAdd) {
    alignas(32) constexpr float a[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    alignas(32) constexpr float b[8] = {8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
    alignas(32) float expected[8] = {};
    alignas(32) float actual[8] = {};

    aethermind::AVX2Add(a, b, actual, 8);
    aethermind::ReferenceAdd(a, b, expected, 8);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

// AVX2Add 也应处理非 8 倍数的长度，验证尾部标量回退的正确性。
TEST(AvxLearning, LoadStoreAddTail) {
    alignas(32) constexpr float a[10] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    alignas(32) constexpr float b[10] = {9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F, 0.0F};
    alignas(32) float expected[10] = {};
    alignas(32) float actual[10] = {};

    aethermind::AVX2Add(a, b, actual, 10);
    aethermind::ReferenceAdd(a, b, expected, 10);

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}
#else
TEST(AvxLearning, LoadStoreAdd) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 2. Fused Multiply-Add (FMA) — a * b + c
// =========================================================================
// _mm256_fmadd_ps 需要 FMA ISA (通常与 AVX2 一起启用，但 feature flag 独立)。
#if defined(__AVX2__) && defined(__FMA__)

TEST(AvxLearning, Fma) {
    alignas(32) constexpr float a[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    alignas(32) constexpr float b[8] = {2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
    alignas(32) constexpr float c[8] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    alignas(32) float actual[8] = {};
    alignas(32) float expected[8] = {};

    aethermind::AVX2FMAdd(a, b, c, actual, 8);
    aethermind::ReferenceFMAdd(a, b, c, expected, 8);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST(AvxLearning, FmaTail) {
    alignas(32) constexpr float a[10] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    alignas(32) constexpr float b[10] = {2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
    alignas(32) constexpr float c[10] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    alignas(32) float actual[10] = {};
    alignas(32) float expected[10] = {};

    aethermind::AVX2FMAdd(a, b, c, actual, 10);
    aethermind::ReferenceFMAdd(a, b, c, expected, 10);

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST(AvxLearning, DotProduct) {
    alignas(32) constexpr float a[10] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
    alignas(32) constexpr float b[10] = {10.0F, 9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};

    const float actual = aethermind::dot_product_avx2_unroll(a, b, 10);
    const float expected = aethermind::ReferenceDotProduct(a, b, 10);

    EXPECT_FLOAT_EQ(actual, expected);
}

TEST(AvxLearning, DotProductLargeVectorAligned32) {
    constexpr size_t kSize = 1024;
    alignas(32) std::vector<float> a(kSize);
    alignas(32) std::vector<float> b(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        a[i] = i * 0.5F;
        b[i] = i * 0.25F;
    }

    const float actual = aethermind::dot_product_avx2_unroll(a.data(), b.data(), kSize);
    const float expected = aethermind::ReferenceDotProduct(a.data(), b.data(), kSize);
    aethermind::ExpectClose(actual, expected);
}

TEST(AvxLearning, DotProductVeryLargeVector) {
    constexpr size_t kSize = 1 << 20;// 1,048,576 elements
    std::vector<float> a(kSize);
    std::vector<float> b(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        a[i] = static_cast<float>(i & 0xFF) * 0.1F;
        b[i] = static_cast<float>(255 - (i & 0xFF)) * 0.1F;
    }

    const float actual = aethermind::dot_product_avx2_unroll(a.data(), b.data(), kSize);
    const float expected = aethermind::ReferenceDotProduct(a.data(), b.data(), kSize);
    aethermind::ExpectClose(actual, expected);
}

#else
TEST(AvxLearning, Fma) {
    GTEST_SKIP() << "FMA not supported by compiler (add -mfma)";
}
#endif

// =========================================================================
// 3. Broadcast — 把单个值复制到所有 lane
// =========================================================================
// _mm256_set1_ps: 将标量 "广播" 到 8 个 float 位置。
// 在 RMSNorm 中用于将 inv_rms 广播后与 row 相乘。
#if defined(__AVX2__)
TEST(AvxLearning, Broadcast) {
    alignas(32) float simd_out[8] = {};

    // SIMD: 把 3.14F 复制到所有 8 个 lane
    __m256 vbcast = _mm256_set1_ps(3.14F);
    _mm256_storeu_ps(simd_out, vbcast);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], 3.14F);
    }
}
#else
TEST(AvxLearning, Broadcast) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 4. Horizontal Reduction — 向量内求和 (RMSNorm sum_sq 的核心)
// =========================================================================
// _mm256_hadd_ps + extract: 将 8 个 float 加成一个标量。
// 这是 RMSNorm 第一阶段 sum(x²) 的 SIMD 实现基础。
#if defined(__AVX2__)
TEST(AvxLearning, HorizontalSum) {
    alignas(32) const float data[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};

    // SIMD horizontal sum
    __m256 v = _mm256_loadu_ps(data);
    // 高 128 位 + 低 128 位
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);  // [a0+a4, a1+a5, a2+a6, a3+a7]
    sum128 = _mm_hadd_ps(sum128, sum128);// [a0+a4+a1+a5, a2+a6+a3+a7, ...]
    sum128 = _mm_hadd_ps(sum128, sum128);// [total, total, total, total]
    float simd_sum = _mm_cvtss_f32(sum128);

    // 标量 reference
    float ref_sum = 0.0F;
    for (int i = 0; i < 8; ++i) {
        ref_sum += data[i];
    }

    EXPECT_FLOAT_EQ(simd_sum, ref_sum);
}
#else
TEST(AvxLearning, HorizontalSum) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 5. Horizontal Max — 向量内求最大值
// =========================================================================
// 思路同 sum，把加法换成取 max。
#if defined(__AVX2__)
TEST(AvxLearning, HorizontalMax) {
    alignas(32) const float data[8] = {1.0F, 2.0F, 15.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};

    // SIMD horizontal max
    __m256 v = _mm256_loadu_ps(data);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 mx128 = _mm_max_ps(lo, hi);// per-lane max of low/high halves
    mx128 = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, _MM_SHUFFLE(2, 3, 0, 1)));
    mx128 = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, _MM_SHUFFLE(1, 0, 3, 2)));
    float simd_max = _mm_cvtss_f32(mx128);

    float ref_max = data[0];
    for (int i = 1; i < 8; ++i) {
        if (data[i] > ref_max) ref_max = data[i];
    }

    EXPECT_FLOAT_EQ(simd_max, ref_max);
}
#else
TEST(AvxLearning, HorizontalMax) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 6. Shuffle — 在 128-bit lane 内重排元素
// =========================================================================
// _mm256_shuffle_ps: 每个 128-bit lane 内按 mask 重排。
// 跨 lane 重排需用 _mm256_permutevar8x32_ps (AVX2)。
#if defined(__AVX2__)
TEST(AvxLearning, ShuffleWithinLane) {
    alignas(32) const float data[8] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    alignas(32) float simd_out[8] = {};

    // 在每个 128-bit lane 内做 shuffle: 交换 lane 内的第 0 和第 3 个元素。
    // _MM_SHUFFLE(z, y, x, w) 各参数含义:
    //   w (bit[1:0]) → dst[0] = a[w]
    //   x (bit[3:2]) → dst[1] = a[x]
    //   y (bit[5:4]) → dst[2] = b[y]
    //   z (bit[7:6]) → dst[3] = b[z]
    // 这里 a == b == v, 所以:
    //   dst[0] = v[1], dst[1] = v[0], dst[2] = v[3], dst[3] = v[2]
    __m256 v = _mm256_loadu_ps(data);
    __m256 vshuf = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    _mm256_storeu_ps(simd_out, vshuf);

    EXPECT_FLOAT_EQ(simd_out[0], 1.0F);// dst[0] = v[1]
    EXPECT_FLOAT_EQ(simd_out[1], 0.0F);// dst[1] = v[0]
    EXPECT_FLOAT_EQ(simd_out[2], 3.0F);// dst[2] = v[3]
    EXPECT_FLOAT_EQ(simd_out[3], 2.0F);// dst[3] = v[2]
    EXPECT_FLOAT_EQ(simd_out[4], 5.0F);// lane 1: dst[0] = v[5]
    EXPECT_FLOAT_EQ(simd_out[5], 4.0F);// dst[1] = v[4]
    EXPECT_FLOAT_EQ(simd_out[6], 7.0F);// dst[2] = v[7]
    EXPECT_FLOAT_EQ(simd_out[7], 6.0F);// dst[3] = v[6]
}
#else
TEST(AvxLearning, ShuffleWithinLane) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 7. Cross-Lane Permute — 跨越 128-bit 边界重排
// =========================================================================
// _mm256_permutevar8x32_ps (AVX2): 以 32-bit 粒度的任意排列。
#if defined(__AVX2__)
TEST(AvxLearning, CrossLanePermute) {
    alignas(32) const float data[8] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    alignas(32) float simd_out[8] = {};

    // 将 [0..7] 按索引 [7,6,5,4,3,2,1,0] 逆序
    alignas(32) const int indices[8] = {7, 6, 5, 4, 3, 2, 1, 0};
    __m256 v = _mm256_loadu_ps(data);
    __m256i vidx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
    __m256 vperm = _mm256_permutevar8x32_ps(v, vidx);
    _mm256_storeu_ps(simd_out, vperm);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], 7.0F - i) << "index " << i;
    }
}
#else
TEST(AvxLearning, CrossLanePermute) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 8. Compare + Blend — 掩码选择
// =========================================================================
// _mm256_cmp_ps + _mm256_blendv_ps: 比较后将负值替换为 0。
#if defined(__AVX2__)
TEST(AvxLearning, CompareBlend) {
    alignas(32) const float data[8] = {1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F, 7.0F, -8.0F};
    alignas(32) float simd_out[8] = {};
    alignas(32) float ref_out[8] = {};

    // SIMD: 将负值替换为 0
    __m256 v = _mm256_loadu_ps(data);
    __m256 vzero = _mm256_setzero_ps();
    // _CMP_LT_OS: less-than, ordered (quiet), signaling
    __m256 mask = _mm256_cmp_ps(v, vzero, _CMP_LT_OS); // v[i] < 0 时全 1
    __m256 vclamped = _mm256_blendv_ps(v, vzero, mask);// mask bit=1 时取 vzero
    _mm256_storeu_ps(simd_out, vclamped);

    // 标量 reference
    for (int i = 0; i < 8; ++i) {
        ref_out[i] = (data[i] < 0.0F) ? 0.0F : data[i];
    }

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], ref_out[i]) << "index " << i;
    }
}
#else
TEST(AvxLearning, CompareBlend) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 9. RMSNorm Building Block — sum_sq (逐行平方和)
// =========================================================================
// 演示如何用 SIMD 实现 RMSNorm 第一阶段: sum(x[i]²)。
// 处理 hidden_size = 8 的情形；实际 kernel 在循环中累加。
#if defined(__AVX2__)
TEST(AvxLearning, RmsNormSumSq) {
    alignas(32) const float row[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};

    // SIMD: row[i]² 向量化，再 horizontal sum
    __m256 v = _mm256_loadu_ps(row);
    __m256 vsq = _mm256_mul_ps(v, v);// 平方，8 路并行
    __m128 hi = _mm256_extractf128_ps(vsq, 1);
    __m128 lo = _mm256_castps256_ps128(vsq);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float simd_sum_sq = _mm_cvtss_f32(sum128);

    // 标量 reference
    float ref_sum_sq = 0.0F;
    for (int i = 0; i < 8; ++i) {
        ref_sum_sq += row[i] * row[i];
    }

    EXPECT_FLOAT_EQ(simd_sum_sq, ref_sum_sq);
}
#else
TEST(AvxLearning, RmsNormSumSq) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 10. RMSNorm Building Block — Scale (广播 inv_rms 后逐行缩放)
// =========================================================================
// 演示如何用 SIMD 实现 RMSNorm 第二阶段: output[i] = row[i] * inv_rms * weight[i]。
// 其中 inv_rms 是标量，weight 是向量。
#if defined(__AVX2__)
TEST(AvxLearning, RmsNormScale) {
    alignas(32) const float row[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    alignas(32) const float weight[8] = {1.0F, 0.5F, 1.5F, 2.0F, 1.0F, 0.5F, 1.5F, 2.0F};
    alignas(32) float simd_out[8] = {};
    alignas(32) float ref_out[8] = {};
    constexpr float inv_rms = 0.730297F;// 假设值，来自 row {1,2,3,4} 的 rms_norm

    // SIMD: inv_rms → broadcast, 然后 load weight → mul → store
    __m256 vrow = _mm256_loadu_ps(row);
    __m256 vweight = _mm256_loadu_ps(weight);
    __m256 vinv_rms = _mm256_set1_ps(inv_rms);
    __m256 vscaled = _mm256_mul_ps(vrow, vinv_rms);// row[i] * inv_rms
    __m256 vout = _mm256_mul_ps(vscaled, vweight); // * weight[i]
    _mm256_storeu_ps(simd_out, vout);

    // 标量 reference
    for (int i = 0; i < 8; ++i) {
        ref_out[i] = row[i] * inv_rms * weight[i];
    }

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], ref_out[i]) << "index " << i;
    }
}
#else
TEST(AvxLearning, RmsNormScale) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 11. Gather — 按索引收集非连续元素
// =========================================================================
// _mm256_i32gather_ps: 从基地址按 int32 索引收集 float。
// 适用于 embedding lookup、权重解包等场景。
#if defined(__AVX2__)
TEST(AvxLearning, Gather) {
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
    alignas(16) const int idx[8] = {3, 7, 1, 14, 0, 9, 5, 11};
    alignas(32) float simd_out[8] = {};
    alignas(32) float ref_out[8] = {};

    // SIMD gather
    __m256i vidx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx));
    __m256 vgathered = _mm256_i32gather_ps(table, vidx, /*scale=*/4);
    _mm256_storeu_ps(simd_out, vgathered);

    // 标量 reference
    for (int i = 0; i < 8; ++i) {
        ref_out[i] = table[idx[i]];
    }

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(simd_out[i], ref_out[i]) << "index " << i;
    }
}
#else
TEST(AvxLearning, Gather) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif

// =========================================================================
// 12. Masked Load — 处理尾部不足 8 个元素的情况
// =========================================================================
// AVX2 没有直接的 masked load (AVX-512 才有 _mm256_mask_loadu_ps)。
// 这里演示两种变通方案:
//   方案 A: 用标量处理尾部
//   方案 B: 用 _mm256_maskload_ps (AVX2, 需要 sign mask)
#if defined(__AVX2__)
TEST(AvxLearning, TailHandling) {
    // 假设 hidden_size = 10, 需要 8 + 2 两段处理
    alignas(32) const float data[10] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F,
                                        5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    float sum_simd[10] = {};// 初始化为 0
    float sum_ref[10] = {};

    // --- 方案 A: 8 路 SIMD + 标量尾部 ---
    __m256 vmain = _mm256_loadu_ps(data);// data[0..7]
    __m256 vsq = _mm256_mul_ps(vmain, vmain);
    _mm256_storeu_ps(sum_simd, vsq);// simd_out[0..7] = data²
    for (int i = 8; i < 10; ++i) {  // 标量尾部
        sum_simd[i] = data[i] * data[i];
    }

    // --- 方案 B: maskload (不依赖 AVX-512) ---
    // mask 高位为 1 = 读取; 高位为 0 = 填 0
    // 注意: maskload 用符号位，不是 bool
    alignas(16) const int32_t mask_data[4] = {-1, -1, 0, 0};// 只读前 2 个
    __m128i vmask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask_data));
    __m128 vtail = _mm_maskload_ps(data + 8, vmask);// 读 2 个，其余 2 个填 0
    alignas(16) float tail_buf[4] = {};
    _mm_storeu_ps(tail_buf, vtail);
    // tail_buf[0] = data[8]², tail_buf[1] = data[9]², tail_buf[2..3] = 0
    // 这里只是演示，实际 kernel 中用方案 A 更简单

    // 标量 reference
    for (int i = 0; i < 10; ++i) {
        sum_ref[i] = data[i] * data[i];
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(sum_simd[i], sum_ref[i]) << "index " << i;
    }
}
#else
TEST(AvxLearning, TailHandling) {
    GTEST_SKIP() << "AVX2 not supported by compiler (add -mavx2)";
}
#endif
