#include "dot_product_internal.h"

namespace aethermind::cpu {

float DotProductF32Scalar(const float* a, const float* b, std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void DotProductF32MultiTargetScalar(const float* lhs, const float* rhs, std::size_t n,
                                    std::size_t num_targets, std::size_t rhs_stride,
                                    float* out) noexcept {
    for (std::size_t t = 0; t < num_targets; ++t) {
        out[t] = DotProductF32Scalar(lhs, rhs + t * rhs_stride, n);
    }
}

} // namespace aethermind::cpu