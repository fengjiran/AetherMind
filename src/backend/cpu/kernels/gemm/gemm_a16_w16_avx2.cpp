#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_a16_w16_internal.h"

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include "gemm_w16_pack_avx2.h"

#include <algorithm>
#include <cmath>
#endif

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
template<class Widen>
float DotProduct(const std::byte* input, const std::byte* weight, int64_t in_features) noexcept {
    __m256 accumulator = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + 8 <= in_features; k += 8) {
        accumulator = _mm256_fmadd_ps(Widen::LoadEight(input + k * 2),
                                      Widen::LoadEight(weight + k * 2),
                                      accumulator);
    }

    float sum = HorizontalSumAvx2(accumulator);
    for (; k < in_features; ++k) {
        const float product = Widen::LoadOne(input + k * 2) * Widen::LoadOne(weight + k * 2);
        sum = sum + product;
    }
    return sum;
}

template<class Widen>
float PackActivationRows(const std::byte* source, size_t row_bytes,
                         int64_t rows, int64_t padded_rows, int64_t kc, float* panel) noexcept {
    __m256 max_vector = _mm256_setzero_ps();
    float max_tail = 0.0F;
    for (int64_t row = 0; row < rows; ++row) {
        const auto* input = source + static_cast<size_t>(row) * row_bytes;
        auto* dest = panel + row * kGemmAvx2KC;
        int64_t k = 0;
        for (; k + 8 <= kc; k += 8) {
            const __m256 values = Widen::LoadEight(input + k * sizeof(uint16_t));
            _mm256_store_ps(dest + k, values);
            const __m256 magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0F), values);
            max_vector = _mm256_max_ps(magnitude, max_vector);
        }
        for (; k < kc; ++k) {
            dest[k] = Widen::LoadOne(input + k * sizeof(uint16_t));
            max_tail = std::max(max_tail, std::fabs(dest[k]));
        }
    }
    for (int64_t row = rows; row < padded_rows; ++row) {
        std::fill_n(panel + row * kGemmAvx2KC, kc, 0.0F);
    }
    alignas(32) float magnitudes[8];
    _mm256_store_ps(magnitudes, max_vector);
    for (const float value: magnitudes) {
        max_tail = std::max(max_tail, value);
    }
    return max_tail;
}

template<class Widen>
float PackWeightTail(const std::byte* source, size_t row_bytes,
                     int64_t cols, int64_t kc, float* panel) noexcept {
    float magnitude = 0.0F;
    for (int64_t k = 0; k < kc; ++k) {
        float* dest = panel + k * kGemmAvx2NR;
        for (int64_t col = 0; col < cols; ++col) {
            dest[col] = Widen::LoadOne(source + static_cast<size_t>(col) * row_bytes + k * sizeof(uint16_t));
            magnitude = std::max(magnitude, std::fabs(dest[col]));
        }
        std::fill(dest + cols, dest + kGemmAvx2NR, 0.0F);
    }
    return magnitude;
}

template<class Widen>
bool TryRunBlocked(const GemmA16W16Args& params) noexcept {
    const GemmArgs& args = params.gemm;
    // Carry the FP32 sums across K panels and round only while publishing the
    // completed output block. Scratch stays bounded independently of M/N/K.
    constexpr int64_t kOutputColumns = 128;
    static_assert(kOutputColumns % kGemmAvx2NR == 0);
    alignas(32) float panel_a[kGemmAvx2MB * kGemmAvx2KC];
    alignas(32) float panel_b[kGemmAvx2KC * kGemmAvx2NR];
    alignas(32) float sums[kGemmAvx2MB * kOutputColumns];
    const size_t input_row_bytes = static_cast<size_t>(args.lhs_m_stride) * sizeof(uint16_t);
    const size_t weight_row_bytes = static_cast<size_t>(args.rhs_n_stride) * sizeof(uint16_t);
    const size_t output_bytes = A16ElementBytes(params.output_dtype);

    for (int64_t mb = 0; mb < args.m; mb += kGemmAvx2MB) {
        const int64_t rows = std::min(args.m - mb, kGemmAvx2MB);
        const int64_t padded_rows = (rows + kGemmAvx2MR - 1) / kGemmAvx2MR * kGemmAvx2MR;
        for (int64_t nb = 0; nb < args.n; nb += kOutputColumns) {
            const int64_t cols = std::min(args.n - nb, kOutputColumns);
            const int64_t padded_cols = (cols + kGemmAvx2NR - 1) / kGemmAvx2NR * kGemmAvx2NR;
            std::fill_n(sums, padded_rows * padded_cols, 0.0F);
            for (int64_t kb = 0; kb < args.k; kb += kGemmAvx2KC) {
                const int64_t kc = std::min(args.k - kb, kGemmAvx2KC);
                const size_t k_offset = static_cast<size_t>(kb) * sizeof(uint16_t);
                const float max_a = PackActivationRows<Widen>(
                        args.lhs + static_cast<size_t>(mb) * input_row_bytes + k_offset,
                        input_row_bytes, rows, padded_rows, kc, panel_a);
                for (int64_t col = 0; col < cols; col += kGemmAvx2NR) {
                    const auto* weight = args.rhs + static_cast<size_t>(nb + col) * weight_row_bytes + k_offset;
                    const float max_b = cols - col >= kGemmAvx2NR
                                                ? PackGemmW16BPanelAvx2<Widen>(weight, weight_row_bytes, kc, panel_b)
                                                : PackWeightTail<Widen>(weight, weight_row_bytes, cols - col, kc, panel_b);
                    if (!GemmReductionMagnitudeIsSafe(max_a, max_b, args.k)) {
                        return false;
                    }
                    for (int64_t row = 0; row < padded_rows; row += kGemmAvx2MR) {
                        AccumulateF32Tile4x16Avx2(panel_a + row * kGemmAvx2KC, panel_b,
                                                  kc, padded_cols, sums + row * padded_cols + col);
                    }
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                auto* dest = args.output +
                             static_cast<size_t>((mb + row) * args.output_m_stride + nb) * output_bytes;
                for (int64_t col = 0; col < cols; ++col) {
                    const float value = sums[row * padded_cols + col];
                    if (!std::isfinite(value)) {
                        return false;
                    }
                    StoreA16Element(dest + col * output_bytes, value, params.output_dtype);
                }
            }
        }
    }
    return true;
}

template<class Widen>
Status RunAvx2(const GemmA16W16Args& params) noexcept {
    const GemmArgs& args = params.gemm;
    // The vector loop requires unit column strides; any other positive layout
    // uses the scalar reference so the prepared arguments never fail.
    if (args.lhs_k_stride != 1 || args.rhs_k_stride != 1 ||
        args.m == 0 || args.n == 0 || args.k == 0) {
        return RunGemmA16W16Reference(params);
    }

    if (args.m > 8 && args.k >= 8 && args.n >= kGemmAvx2NR &&
        args.output_n_stride == 1) {
        // A rejected block may follow completed blocks. The reference
        // overwrites the whole disjoint output in that case.
        return TryRunBlocked<Widen>(params) ? Status::Ok() : RunGemmA16W16Reference(params);
    }

    const size_t input_bytes = A16ElementBytes(params.operand_dtype);
    const size_t weight_bytes = A16ElementBytes(params.operand_dtype);
    const size_t output_bytes = A16ElementBytes(params.output_dtype);

    for (int64_t row = 0; row < args.m; ++row) {
        const std::byte* const input_row =
                args.lhs + static_cast<size_t>(row * args.lhs_m_stride) * input_bytes;
        for (int64_t col = 0; col < args.n; ++col) {
            const std::byte* const weight_row =
                    args.rhs + static_cast<size_t>(col * args.rhs_n_stride) * weight_bytes;
            const float sum = DotProduct<Widen>(input_row, weight_row, args.k);
            StoreA16Element(
                    args.output +
                            static_cast<size_t>(row * args.output_m_stride +
                                                col * args.output_n_stride) *
                                    output_bytes,
                    sum, params.output_dtype);
        }
    }
    return Status::Ok();
}

#endif

} // namespace

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
Status RunGemmA16W16Avx2Fma(const GemmA16W16Args& params) noexcept {
    if (params.operand_dtype == DataType::BFloat(16)) {
        return RunAvx2<Bf16GemmWeightOps>(params);
    }
    return RunAvx2<F16GemmWeightOps>(params);
}
#endif

} // namespace aethermind::cpu::detail
