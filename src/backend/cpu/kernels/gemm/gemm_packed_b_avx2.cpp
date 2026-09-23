#include "aethermind/backend/cpu/cpu_bpanel_packing.h"
#include "aethermind/base/macros.h"
#include "gemm_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace aethermind::cpu::detail {
namespace {

Status ValidatePackedBArgs(const PackedGemmF32Args& packed) noexcept {
    const GemmF32Args& args = packed.gemm;
    if (args.m < 0 || args.n < 0 || args.k < 0 || packed.logical_n < 0 ||
        packed.logical_k < 0 || args.k != packed.logical_k ||
        packed.weight_n_offset < 0 || packed.n_blocks < 0 ||
        packed.weight_n_offset > packed.logical_n ||
        args.n > packed.logical_n - packed.weight_n_offset) {
        return Status::InvalidArgument("Packed GEMM received invalid dimensions");
    }
    const int64_t expected_blocks =
            packed.logical_n / cpu::kCpuBPanelF32V1NR +
            (packed.logical_n % cpu::kCpuBPanelF32V1NR != 0);
    if (packed.n_blocks != expected_blocks) {
        return Status::InvalidArgument("Packed GEMM n_blocks does not match logical N");
    }
    AM_ASSIGN_OR_RETURN(const size_t required_bytes,
                        cpu::CpuBPanelF32V1PackedByteSize(
                                packed.logical_n, packed.logical_k));
    if (packed.packed_nbytes != required_bytes ||
        (required_bytes != 0 &&
         (packed.packed_b == nullptr ||
          reinterpret_cast<std::uintptr_t>(packed.packed_b) %
                          cpu::kCpuBPanelF32V1Alignment !=
                  0))) {
        return Status::InvalidArgument(
                "Packed GEMM artifact does not match its bpanel byte/alignment contract");
    }
    return Status::Ok();
}

constexpr int64_t kPackedBMR = 4;
constexpr int64_t kPackedBMB = 48;
constexpr int64_t kPackedBNR = cpu::kCpuBPanelF32V1NR;
constexpr int64_t kPackedBKC = cpu::kCpuBPanelF32V1KC;

size_t PackedBIndex(int64_t n, int64_t k, int64_t n_blocks) noexcept {
    const size_t panel = static_cast<size_t>(k / kPackedBKC);
    const size_t block = static_cast<size_t>(n / kPackedBNR);
    const size_t panel_row = static_cast<size_t>(k % kPackedBKC);
    const size_t column = static_cast<size_t>(n % kPackedBNR);
    return (((panel * static_cast<size_t>(n_blocks) + block) *
                     static_cast<size_t>(kPackedBKC) +
             panel_row) *
            static_cast<size_t>(kPackedBNR)) +
           column;
}

#if defined(__AVX2__) && defined(__FMA__)

void PackAPanel(const float* lhs,
                int64_t lhs_m_stride,
                int64_t kb,
                int64_t mb_base,
                int64_t mb_rows,
                int64_t kc_len,
                float* buf_a) noexcept {
    for (int64_t row = 0; row < mb_rows; ++row) {
        const float* const source = lhs + (mb_base + row) * lhs_m_stride + kb;
        float* const destination = buf_a + row * kPackedBKC;
        for (int64_t kk = 0; kk < kc_len; ++kk) {
            destination[kk] = source[kk];
        }
    }
}

void MicroKernel4x16(const float* buf_a,
                     const float* buf_b,
                     int64_t kc_len,
                     int64_t output_m_stride,
                     float* output_row0) noexcept {
    __m256 acc00 = _mm256_setzero_ps();
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps();
    __m256 acc21 = _mm256_setzero_ps();
    __m256 acc30 = _mm256_setzero_ps();
    __m256 acc31 = _mm256_setzero_ps();

    const float* const a0 = buf_a;
    const float* const a1 = a0 + kPackedBKC;
    const float* const a2 = a1 + kPackedBKC;
    const float* const a3 = a2 + kPackedBKC;
    for (int64_t kk = 0; kk < kc_len; ++kk) {
        const __m256 x0 = _mm256_set1_ps(a0[kk]);
        const __m256 x1 = _mm256_set1_ps(a1[kk]);
        const __m256 x2 = _mm256_set1_ps(a2[kk]);
        const __m256 x3 = _mm256_set1_ps(a3[kk]);
        const __m256 b0 = _mm256_loadu_ps(buf_b + kk * kPackedBNR);
        const __m256 b1 = _mm256_loadu_ps(buf_b + kk * kPackedBNR + 8);
        acc00 = _mm256_fmadd_ps(x0, b0, acc00);
        acc01 = _mm256_fmadd_ps(x0, b1, acc01);
        acc10 = _mm256_fmadd_ps(x1, b0, acc10);
        acc11 = _mm256_fmadd_ps(x1, b1, acc11);
        acc20 = _mm256_fmadd_ps(x2, b0, acc20);
        acc21 = _mm256_fmadd_ps(x2, b1, acc21);
        acc30 = _mm256_fmadd_ps(x3, b0, acc30);
        acc31 = _mm256_fmadd_ps(x3, b1, acc31);
    }

    float* const out0 = output_row0;
    float* const out1 = out0 + output_m_stride;
    float* const out2 = out1 + output_m_stride;
    float* const out3 = out2 + output_m_stride;
    _mm256_storeu_ps(out0, _mm256_add_ps(_mm256_loadu_ps(out0), acc00));
    _mm256_storeu_ps(out0 + 8, _mm256_add_ps(_mm256_loadu_ps(out0 + 8), acc01));
    _mm256_storeu_ps(out1, _mm256_add_ps(_mm256_loadu_ps(out1), acc10));
    _mm256_storeu_ps(out1 + 8, _mm256_add_ps(_mm256_loadu_ps(out1 + 8), acc11));
    _mm256_storeu_ps(out2, _mm256_add_ps(_mm256_loadu_ps(out2), acc20));
    _mm256_storeu_ps(out2 + 8, _mm256_add_ps(_mm256_loadu_ps(out2 + 8), acc21));
    _mm256_storeu_ps(out3, _mm256_add_ps(_mm256_loadu_ps(out3), acc30));
    _mm256_storeu_ps(out3 + 8, _mm256_add_ps(_mm256_loadu_ps(out3 + 8), acc31));
}

#endif

} // namespace

Status RunGemmF32PackedBScan(const PackedGemmF32Args& packed) noexcept {
    const GemmF32Args& args = packed.gemm;
    AM_RETURN_IF_ERROR(ValidatePackedBArgs(packed));
    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }
    if (args.output == nullptr) {
        return Status::InvalidArgument("Packed GEMM scan requires output data");
    }
    if (args.k == 0) {
        return RunGemmF32PackedBReference(packed);
    }
    if (args.lhs == nullptr) {
        return Status::InvalidArgument("Packed GEMM scan requires input data");
    }
#if !defined(__AVX2__) || !defined(__FMA__)
    return RunGemmF32PackedBReference(packed);
#else
    if (args.output_n_stride != 1) {
        return RunGemmF32PackedBReference(packed);
    }

    for (int64_t row = 0; row < args.m; ++row) {
        const float* const lhs_row = args.lhs + row * args.lhs_m_stride;
        float* const output_row = args.output + row * args.output_m_stride;
        int64_t col = 0;
        for (; col < args.n &&
               (packed.weight_n_offset + col) % kPackedBNR != 0;
             ++col) {
            double sum = 0.0;
            const int64_t weight_n = packed.weight_n_offset + col;
            for (int64_t k = 0; k < args.k; ++k) {
                sum += static_cast<double>(lhs_row[k * args.lhs_k_stride]) *
                       static_cast<double>(packed.packed_b[PackedBIndex(weight_n, k, packed.n_blocks)]);
            }
            output_row[col] = static_cast<float>(sum);
        }

        for (; col + kPackedBNR <= args.n; col += kPackedBNR) {
            const int64_t weight_n = packed.weight_n_offset + col;
            const size_t n_block = static_cast<size_t>(weight_n / kPackedBNR);
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int64_t kb = 0;
            size_t k_panel = 0;
            while (kb < args.k) {
                const int64_t kc_len = std::min(kPackedBKC, args.k - kb);
                const size_t panel_index =
                        (k_panel * static_cast<size_t>(packed.n_blocks) + n_block) *
                        static_cast<size_t>(kPackedBKC * kPackedBNR);
                const float* const b_panel = packed.packed_b + panel_index;
                for (int64_t kk = 0; kk < kc_len; ++kk) {
                    const __m256 x = _mm256_set1_ps(
                            lhs_row[(kb + kk) * args.lhs_k_stride]);
                    const float* const b_row = b_panel + kk * kPackedBNR;
                    acc0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b_row), acc0);
                    acc1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b_row + 8), acc1);
                }
                kb += kc_len;
                ++k_panel;
            }
            _mm256_storeu_ps(output_row + col, acc0);
            _mm256_storeu_ps(output_row + col + 8, acc1);
        }

        for (; col < args.n; ++col) {
            double sum = 0.0;
            const int64_t weight_n = packed.weight_n_offset + col;
            for (int64_t k = 0; k < args.k; ++k) {
                sum += static_cast<double>(lhs_row[k * args.lhs_k_stride]) *
                       static_cast<double>(packed.packed_b[PackedBIndex(weight_n, k, packed.n_blocks)]);
            }
            output_row[col] = static_cast<float>(sum);
        }
    }
    return Status::Ok();
#endif
}

Status RunGemmF32PackedBBlocked(const PackedGemmF32Args& packed) noexcept {
    const GemmF32Args& args = packed.gemm;
    AM_RETURN_IF_ERROR(ValidatePackedBArgs(packed));
    if (args.m == 0 || args.n == 0 || args.k == 0) {
        return RunGemmF32PackedBScan(packed);
    }
#if !defined(__AVX2__) || !defined(__FMA__)
    return RunGemmF32PackedBReference(packed);
#else
    if (args.m < 9 || args.lhs_k_stride != 1 || args.output_n_stride != 1 ||
        packed.weight_n_offset % kPackedBNR != 0 || args.lhs == nullptr ||
        args.output == nullptr) {
        return RunGemmF32PackedBScan(packed);
    }
    const int64_t m_full = args.m - args.m % kPackedBMR;
    const int64_t n_full = args.n - args.n % kPackedBNR;
    if (m_full == 0 || n_full == 0) {
        return RunGemmF32PackedBScan(packed);
    }

    for (int64_t row = 0; row < m_full; ++row) {
        float* const output_row = args.output + row * args.output_m_stride;
        std::fill_n(output_row, n_full, 0.0F);
    }

    alignas(32) float buf_a[static_cast<size_t>(kPackedBMB * kPackedBKC)];
    int64_t kb = 0;
    size_t k_panel = 0;
    while (kb < args.k) {
        const int64_t kc_len = std::min(kPackedBKC, args.k - kb);
        for (int64_t mb = 0; mb < m_full; mb += kPackedBMB) {
            const int64_t mb_rows = std::min(kPackedBMB, m_full - mb);
            PackAPanel(args.lhs, args.lhs_m_stride, kb, mb, mb_rows, kc_len,
                       buf_a);
            for (int64_t nb = 0; nb < n_full; nb += kPackedBNR) {
                const int64_t weight_n = packed.weight_n_offset + nb;
                const size_t n_block = static_cast<size_t>(weight_n / kPackedBNR);
                const size_t panel_index =
                        (k_panel * static_cast<size_t>(packed.n_blocks) + n_block) *
                        static_cast<size_t>(kPackedBKC * kPackedBNR);
                const float* const b_panel = packed.packed_b + panel_index;
                for (int64_t row = 0; row < mb_rows; row += kPackedBMR) {
                    MicroKernel4x16(
                            buf_a + row * kPackedBKC, b_panel, kc_len,
                            args.output_m_stride,
                            args.output + (mb + row) * args.output_m_stride + nb);
                }
            }
        }
        kb += kc_len;
        ++k_panel;
    }

    if (n_full < args.n) {
        PackedGemmF32Args right_edge = packed;
        right_edge.gemm.m = m_full;
        right_edge.gemm.n = args.n - n_full;
        right_edge.gemm.output += n_full;
        right_edge.weight_n_offset += n_full;
        AM_RETURN_IF_ERROR(RunGemmF32PackedBScan(right_edge));
    }
    if (m_full < args.m) {
        PackedGemmF32Args bottom_edge = packed;
        bottom_edge.gemm.lhs += m_full * args.lhs_m_stride;
        bottom_edge.gemm.output += m_full * args.output_m_stride;
        bottom_edge.gemm.m = args.m - m_full;
        AM_RETURN_IF_ERROR(RunGemmF32PackedBScan(bottom_edge));
    }
    return Status::Ok();
#endif
}

Status RunGemmF32PackedB(const PackedGemmF32Args& args) noexcept {
    return args.gemm.m >= 9 ? RunGemmF32PackedBBlocked(args)
                            : RunGemmF32PackedBScan(args);
}

} // namespace aethermind::cpu::detail
