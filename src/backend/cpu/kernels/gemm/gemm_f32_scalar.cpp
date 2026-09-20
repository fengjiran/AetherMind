#include "gemm_internal.h"

#include <array>

namespace aethermind::cpu::detail {
namespace {

constexpr int64_t kScalarOutputBlock = 4;
// This is a conservative driver split for the first scalar candidate, not a
// measured dispatch threshold.
constexpr int64_t kScalarSmallMMax = 8;

void RunM1KContiguous(const GemmF32Args& args) noexcept {
    const float* const lhs = args.lhs;
    int64_t col = 0;
    for (; col + kScalarOutputBlock <= args.n; col += kScalarOutputBlock) {
        const float* weight0 = args.rhs + col * args.rhs_n_stride;
        const float* weight1 = weight0 + args.rhs_n_stride;
        const float* weight2 = weight1 + args.rhs_n_stride;
        const float* weight3 = weight2 + args.rhs_n_stride;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        int64_t inner = 0;
        for (; inner + 1 < args.k; inner += 2) {
            const float lhs0 = lhs[inner];
            const float lhs1 = lhs[inner + 1];
            sum0 += lhs0 * weight0[0];
            sum0 += lhs1 * weight0[1];
            sum1 += lhs0 * weight1[0];
            sum1 += lhs1 * weight1[1];
            sum2 += lhs0 * weight2[0];
            sum2 += lhs1 * weight2[1];
            sum3 += lhs0 * weight3[0];
            sum3 += lhs1 * weight3[1];
            weight0 += 2;
            weight1 += 2;
            weight2 += 2;
            weight3 += 2;
        }

        if (inner < args.k) {
            const float lhs0 = lhs[inner];
            sum0 += lhs0 * weight0[0];
            sum1 += lhs0 * weight1[0];
            sum2 += lhs0 * weight2[0];
            sum3 += lhs0 * weight3[0];
        }

        args.output[col] = sum0;
        args.output[col + 1] = sum1;
        args.output[col + 2] = sum2;
        args.output[col + 3] = sum3;
    }

    for (; col < args.n; ++col) {
        const float* input = lhs;
        const float* weight = args.rhs + col * args.rhs_n_stride;
        float sum = 0.0f;
        for (int64_t inner = 0; inner < args.k; ++inner) {
            sum += *input++ * *weight++;
        }
        args.output[col] = sum;
    }
}

void RunM1NContiguous(const GemmF32Args& args) noexcept {
    const float* const lhs = args.lhs;
    int64_t col = 0;
    for (; col + kScalarOutputBlock <= args.n; col += kScalarOutputBlock) {
        const float* weight = args.rhs + col;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        int64_t inner = 0;
        for (; inner + 1 < args.k; inner += 2) {
            const float lhs0 = lhs[inner];
            const float lhs1 = lhs[inner + 1];
            sum0 += lhs0 * weight[0];
            sum1 += lhs0 * weight[1];
            sum2 += lhs0 * weight[2];
            sum3 += lhs0 * weight[3];
            weight += args.rhs_k_stride;
            sum0 += lhs1 * weight[0];
            sum1 += lhs1 * weight[1];
            sum2 += lhs1 * weight[2];
            sum3 += lhs1 * weight[3];
            weight += args.rhs_k_stride;
        }

        if (inner < args.k) {
            const float lhs0 = lhs[inner];
            sum0 += lhs0 * weight[0];
            sum1 += lhs0 * weight[1];
            sum2 += lhs0 * weight[2];
            sum3 += lhs0 * weight[3];
        }

        args.output[col] = sum0;
        args.output[col + 1] = sum1;
        args.output[col + 2] = sum2;
        args.output[col + 3] = sum3;
    }

    for (; col < args.n; ++col) {
        const float* input = lhs;
        const float* weight = args.rhs + col;
        float sum = 0.0f;
        for (int64_t inner = 0; inner < args.k; ++inner) {
            sum += *input++ * *weight;
            weight += args.rhs_k_stride;
        }
        args.output[col] = sum;
    }
}

template<int Rows, bool RhsKContiguous>
void RunFullOutputBlock(const GemmF32Args& args, int64_t first_row, int64_t first_col) noexcept {
    std::array<const float*, Rows> lhs_rows{};
    std::array<float*, Rows> output_rows{};
    std::array<std::array<float, kScalarOutputBlock>, Rows> sums{};
    for (int i = 0; i < Rows; ++i) {
        lhs_rows[i] = args.lhs + (first_row + i) * args.lhs_m_stride;
        output_rows[i] = args.output + (first_row + i) * args.output_m_stride + first_col;
    }

    int64_t inner = 0;
    if constexpr (RhsKContiguous) {
        std::array<const float*, kScalarOutputBlock> weights{
                args.rhs + first_col * args.rhs_n_stride,
                args.rhs + (first_col + 1) * args.rhs_n_stride,
                args.rhs + (first_col + 2) * args.rhs_n_stride,
                args.rhs + (first_col + 3) * args.rhs_n_stride,
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
    } else {
        const float* weight = args.rhs + first_col;
        for (; inner + 1 < args.k; inner += 2) {
            for (int r = 0; r < Rows; ++r) {
                const float lhs0 = lhs_rows[r][inner];
                sums[r][0] += lhs0 * weight[0];
                sums[r][1] += lhs0 * weight[1];
                sums[r][2] += lhs0 * weight[2];
                sums[r][3] += lhs0 * weight[3];
            }

            weight += args.rhs_k_stride;
            for (int r = 0; r < Rows; ++r) {
                const float lhs1 = lhs_rows[r][inner + 1];
                sums[r][0] += lhs1 * weight[0];
                sums[r][1] += lhs1 * weight[1];
                sums[r][2] += lhs1 * weight[2];
                sums[r][3] += lhs1 * weight[3];
            }
            weight += args.rhs_k_stride;
        }

        if (inner < args.k) {
            for (int r = 0; r < Rows; ++r) {
                const float lhs0 = lhs_rows[r][inner];
                sums[r][0] += lhs0 * weight[0];
                sums[r][1] += lhs0 * weight[1];
                sums[r][2] += lhs0 * weight[2];
                sums[r][3] += lhs0 * weight[3];
            }
        }
    }

    for (int r = 0; r < Rows; ++r) {
        output_rows[r][0] = sums[r][0];
        output_rows[r][1] = sums[r][1];
        output_rows[r][2] = sums[r][2];
        output_rows[r][3] = sums[r][3];
    }
}

template<int Rows, bool RhsKContiguous>
void RunOutputTail(const GemmF32Args& args, int64_t first_row, int64_t first_col) noexcept {
    for (int r = 0; r < Rows; ++r) {
        const float* const lhs = args.lhs + (first_row + r) * args.lhs_m_stride;
        float* const output = args.output + (first_row + r) * args.output_m_stride;
        for (int64_t c = first_col; c < args.n; ++c) {
            float sum = 0.0f;
            if constexpr (RhsKContiguous) {
                const float* weight = args.rhs + c * args.rhs_n_stride;
                for (int64_t inner = 0; inner < args.k; ++inner) {
                    sum += lhs[inner] * weight[inner];
                }
            } else {
                const float* weight = args.rhs + c;
                for (int64_t inner = 0; inner < args.k; ++inner) {
                    sum += lhs[inner] * *weight;
                    weight += args.rhs_k_stride;
                }
            }
            output[c] = sum;
        }
    }
}

template<int Rows, bool RhsKContiguous>
void RunRowBlock(const GemmF32Args& args, int64_t first_row) noexcept {
    int64_t col = 0;
    for (; col + kScalarOutputBlock <= args.n; col += kScalarOutputBlock) {
        RunFullOutputBlock<Rows, RhsKContiguous>(args, first_row, col);
    }

    if (col < args.n) {
        RunOutputTail<Rows, RhsKContiguous>(args, first_row, col);
    }
}

template<bool RhsKContiguous>
void RunSmallM(const GemmF32Args& args) noexcept {
    int64_t r = 0;
    for (; r + 2 <= args.m; r += 2) {
        RunRowBlock<2, RhsKContiguous>(args, r);
    }

    if (r < args.m) {
        RunRowBlock<1, RhsKContiguous>(args, r);
    }
}

template<bool RhsKContiguous>
void RunGenericM(const GemmF32Args& args) noexcept {
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

Status RunGemmF32ScalarOptimized(const GemmF32Args& args) noexcept {
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
            RunM1KContiguous(args);
        } else if (args.m <= kScalarSmallMMax) {
            RunSmallM<true>(args);
        } else {
            RunGenericM<true>(args);
        }
        return Status::Ok();
    }

    if (args.rhs_n_stride == 1) {
        if (args.m == 1) {
            RunM1NContiguous(args);
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
