#include "gemm_internal.h"

namespace aethermind::cpu::detail {
namespace {

constexpr int64_t kScalarOutputBlock = 4;

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
        float sum0 = 0.0F;
        float sum1 = 0.0F;
        float sum2 = 0.0F;
        float sum3 = 0.0F;

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
        float sum = 0.0F;
        for (int64_t inner = 0; inner < args.k; ++inner) {
            sum += *input++ * *weight;
            weight += args.rhs_k_stride;
        }
        args.output[col] = sum;
    }
}

} // namespace

Status RunGemmF32ScalarOptimized(const GemmF32Args& args) noexcept {
    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }

    // Preserve the reference's exact zero-inner semantics and retain it for
    // every layout that this first candidate does not explicitly optimize.
    if (args.k == 0 || args.m != 1 || args.lhs_k_stride != 1 ||
        args.output_n_stride != 1) {
        return RunGemmF32Reference(args);
    }

    if (args.rhs_k_stride == 1) {
        RunM1KContiguous(args);
        return Status::Ok();
    }

    if (args.rhs_n_stride == 1) {
        RunM1NContiguous(args);
        return Status::Ok();
    }
    return RunGemmF32Reference(args);
}

} // namespace aethermind::cpu::detail
