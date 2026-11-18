#include "gemm_internal.h"

#include "aethermind/base/macros.h"

#include <algorithm>
#include <array>

namespace aethermind::cpu::detail {
namespace {

constexpr int64_t kScalarKContiguousOutputBlock = 4;
// This is an N traversal tile, not Rows * NC independent scalar registers.
// Accumulators may reside in L1 while each K step consumes adjacent B values.
constexpr int64_t kScalarNContiguousBlock = 128;
constexpr int64_t kScalarNContiguousKBlock = 128;
// This is a conservative driver split for the first scalar candidate, not a
// measured dispatch threshold.
constexpr int64_t kScalarSmallMMax = 8;

template<int Rows>
void RunNContiguousSmallRowBlock(const GemmArgs& args, int64_t first_row) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const float*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);

    int64_t col = 0;
    for (; col + 4 <= args.n; col += 4) {
        std::array<std::array<float, 4>, Rows> sums{};
        int64_t inner = 0;
        for (; inner + 1 < args.k; inner += 2) {
            const float* const weight0 = rhs_data + inner * args.rhs_k_stride + col;
            const float* const weight1 = weight0 + args.rhs_k_stride;
            for (int r = 0; r < Rows; ++r) {
                const float* const lhs = lhs_data + (first_row + r) * args.lhs_m_stride;
                const float lhs0 = lhs[inner];
                const float lhs1 = lhs[inner + 1];
                for (int c = 0; c < 4; ++c) {
                    sums[r][c] += lhs0 * weight0[c];
                    sums[r][c] += lhs1 * weight1[c];
                }
            }
        }

        if (inner < args.k) {
            const float* const weight = rhs_data + inner * args.rhs_k_stride + col;
            for (int r = 0; r < Rows; ++r) {
                const float lhs = lhs_data[(first_row + r) * args.lhs_m_stride + inner];
                for (int c = 0; c < 4; ++c) {
                    sums[r][c] += lhs * weight[c];
                }
            }
        }

        for (int r = 0; r < Rows; ++r) {
            std::copy_n(sums[r].data(), 4,
                        output_data + (first_row + r) * args.output_m_stride + col);
        }
    }

    for (int r = 0; r < Rows; ++r) {
        const float* const lhs = lhs_data + (first_row + r) * args.lhs_m_stride;
        for (int64_t c = col; c < args.n; ++c) {
            float sum = 0.0f;
            for (int64_t inner = 0; inner < args.k; ++inner) {
                sum += lhs[inner] * rhs_data[inner * args.rhs_k_stride + c];
            }
            output_data[(first_row + r) * args.output_m_stride + c] = sum;
        }
    }
}

template<int Rows>
void RunNContiguousRowBlock(const GemmArgs& args, int64_t first_row) noexcept {
    if constexpr (Rows > 1) {
        // Bound B's physical span to 32 KiB. Wide accumulator setup is not
        // amortized on these tiny multi-row problems; large strides still use
        // the cache-aware path even when logical N is small.
        if (args.n <= 64 && args.k <= 128 && args.rhs_k_stride <= 64) {
            RunNContiguousSmallRowBlock<Rows>(args, first_row);
            return;
        }
    }

    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const float*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);

    // Visit adjacent N tiles within one K panel, so the same small group of
    // B pages is reused before advancing K. Merely tiling K inside a full-K
    // output block would retain the original long cache-line reuse distance.
    // A single N tile has no neighboring tile to reuse; carry its sums through
    // all K without repeatedly storing and loading partial output.
    const int64_t k_block = args.n <= kScalarNContiguousBlock ? args.k : kScalarNContiguousKBlock;
    int64_t kb = 0;
    while (kb < args.k) {
        const int64_t k_end = kb + std::min(k_block, args.k - kb);
        int64_t col = 0;
        while (col < args.n) {
            const int64_t width = std::min(kScalarNContiguousBlock, args.n - col);
            std::array<std::array<float, kScalarNContiguousBlock>, Rows> sums{};
            if (kb != 0) {
                for (int r = 0; r < Rows; ++r) {
                    const float* const output = output_data + (first_row + r) * args.output_m_stride + col;
                    std::copy_n(output, width, sums[r].data());
                }
            }

            int64_t inner = kb;
            for (; inner + 1 < k_end; inner += 2) {
                const float* const weight0 = rhs_data + inner * args.rhs_k_stride + col;
                const float* const weight1 = weight0 + args.rhs_k_stride;
                for (int r = 0; r < Rows; ++r) {
                    const float* const lhs = lhs_data + (first_row + r) * args.lhs_m_stride;
                    const float lhs0 = lhs[inner];
                    const float lhs1 = lhs[inner + 1];
                    for (int64_t c = 0; c < width; ++c) {
                        // Keep each output's ascending-K FP32 accumulation order.
                        sums[r][c] += lhs0 * weight0[c];
                        sums[r][c] += lhs1 * weight1[c];
                    }
                }
            }

            if (inner < k_end) {
                const float* const weight = rhs_data + inner * args.rhs_k_stride + col;
                for (int r = 0; r < Rows; ++r) {
                    const float lhs = lhs_data[(first_row + r) * args.lhs_m_stride + inner];
                    for (int64_t c = 0; c < width; ++c) {
                        sums[r][c] += lhs * weight[c];
                    }
                }
            }

            for (int r = 0; r < Rows; ++r) {
                float* const output = output_data + (first_row + r) * args.output_m_stride + col;
                std::copy_n(sums[r].data(), width, output);
            }
            col += width;
        }
        kb = k_end;
    }
}

template<int Rows>
void RunKContiguousOutputBlock(const GemmArgs& args, int64_t first_row,
                               int64_t first_col) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const float*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);

    std::array<const float*, Rows> lhs_rows{};
    std::array<float*, Rows> output_rows{};
    std::array<std::array<float, kScalarKContiguousOutputBlock>, Rows> sums{};
    for (int i = 0; i < Rows; ++i) {
        lhs_rows[i] = lhs_data + (first_row + i) * args.lhs_m_stride;
        output_rows[i] = output_data + (first_row + i) * args.output_m_stride + first_col;
    }

    int64_t inner = 0;
    std::array weights{
            rhs_data + first_col * args.rhs_n_stride,
            rhs_data + (first_col + 1) * args.rhs_n_stride,
            rhs_data + (first_col + 2) * args.rhs_n_stride,
            rhs_data + (first_col + 3) * args.rhs_n_stride,
    };

    for (; inner + 1 < args.k; inner += 2) {
        for (int r = 0; r < Rows; ++r) {
            const float lhs0 = lhs_rows[r][inner];
            const float lhs1 = lhs_rows[r][inner + 1];
            sums[r][0] += lhs0 * weights[0][0];
            sums[r][0] += lhs1 * weights[0][1];
            sums[r][1] += lhs0 * weights[1][0];
            sums[r][1] += lhs1 * weights[1][1];
            sums[r][2] += lhs0 * weights[2][0];
            sums[r][2] += lhs1 * weights[2][1];
            sums[r][3] += lhs0 * weights[3][0];
            sums[r][3] += lhs1 * weights[3][1];
        }

        for (auto& weight: weights) {
            weight += 2;
        }
    }

    if (inner < args.k) {
        for (int r = 0; r < Rows; ++r) {
            const float lhs0 = lhs_rows[r][inner];
            sums[r][0] += lhs0 * weights[0][0];
            sums[r][1] += lhs0 * weights[1][0];
            sums[r][2] += lhs0 * weights[2][0];
            sums[r][3] += lhs0 * weights[3][0];
        }
    }

    for (int r = 0; r < Rows; ++r) {
        output_rows[r][0] = sums[r][0];
        output_rows[r][1] = sums[r][1];
        output_rows[r][2] = sums[r][2];
        output_rows[r][3] = sums[r][3];
    }
}

template<int Rows>
void RunKContiguousOutputTail(const GemmArgs& args, int64_t first_row,
                              int64_t first_col) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const float*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);

    for (int r = 0; r < Rows; ++r) {
        const float* const lhs = lhs_data + (first_row + r) * args.lhs_m_stride;
        float* const output = output_data + (first_row + r) * args.output_m_stride;
        for (int64_t c = first_col; c < args.n; ++c) {
            float sum = 0.0f;
            const float* weight = rhs_data + c * args.rhs_n_stride;
            for (int64_t inner = 0; inner < args.k; ++inner) {
                sum += lhs[inner] * weight[inner];
            }
            output[c] = sum;
        }
    }
}

// Keep row zero visible when called from the M=1 dispatch. GCC generates a
// different inner loop when it shares the out-of-line path for arbitrary rows.
template<int Rows>
AM_ALWAYS_INLINE void RunKContiguousRowBlock(const GemmArgs& args, int64_t first_row) noexcept {
    int64_t col = 0;
    for (; col + kScalarKContiguousOutputBlock <= args.n; col += kScalarKContiguousOutputBlock) {
        RunKContiguousOutputBlock<Rows>(args, first_row, col);
    }

    if (col < args.n) {
        RunKContiguousOutputTail<Rows>(args, first_row, col);
    }
}

template<int Rows, bool RhsKContiguous>
void RunRowBlock(const GemmArgs& args, int64_t first_row) noexcept {
    if constexpr (RhsKContiguous) {
        RunKContiguousRowBlock<Rows>(args, first_row);
    } else {
        RunNContiguousRowBlock<Rows>(args, first_row);
    }
}

template<bool RhsKContiguous>
void RunSmallM(const GemmArgs& args) noexcept {
    int64_t r = 0;
    for (; r + 2 <= args.m; r += 2) {
        RunRowBlock<2, RhsKContiguous>(args, r);
    }

    if (r < args.m) {
        RunRowBlock<1, RhsKContiguous>(args, r);
    }
}

template<bool RhsKContiguous>
void RunGenericM(const GemmArgs& args) noexcept {
    int64_t r = 0;
    for (; r + 4 <= args.m; r += 4) {
        RunRowBlock<4, RhsKContiguous>(args, r);
    }

    if (r + 2 <= args.m) {
        RunRowBlock<2, RhsKContiguous>(args, r);
        r += 2;
    }

    if (r < args.m) {
        RunRowBlock<1, RhsKContiguous>(args, r);
    }
}

} // namespace

Status RunGemmF32Scalar(const GemmArgs& args) noexcept {
    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }

    // Preserve the reference's exact zero-inner semantics and retain it for
    // every layout that this candidate does not explicitly optimize.
    if (args.k == 0 || args.lhs_k_stride != 1 || args.output_n_stride != 1) {
        return RunGemmF32Reference(args);
    }

    if (args.rhs_k_stride == 1) {
        if (args.m == 1) {
            RunKContiguousRowBlock<1>(args, 0);
        } else if (args.m <= kScalarSmallMMax) {
            RunSmallM<true>(args);
        } else {
            RunGenericM<true>(args);
        }
        return Status::Ok();
    }

    if (args.rhs_n_stride == 1) {
        if (args.m == 1) {
            RunNContiguousRowBlock<1>(args, 0);
        } else if (args.m <= kScalarSmallMMax) {
            RunSmallM<false>(args);
        } else {
            RunGenericM<false>(args);
        }
        return Status::Ok();
    }
    return RunGemmF32Reference(args);
}

} // namespace aethermind::cpu::detail
