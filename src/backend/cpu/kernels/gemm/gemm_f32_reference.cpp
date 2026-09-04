#include "gemm_internal.h"

namespace aethermind::cpu::detail {

Status RunGemmF32Reference(const GemmF32Args& args) noexcept {
    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }

    for (int64_t row = 0; row < args.m; ++row) {
        for (int64_t col = 0; col < args.n; ++col) {
            double sum = 0.0;
            for (int64_t inner = 0; inner < args.k; ++inner) {
                sum += static_cast<double>(args.lhs[row * args.lhs_m_stride + inner * args.lhs_k_stride]) *
                       static_cast<double>(args.rhs[inner * args.rhs_k_stride + col * args.rhs_n_stride]);
            }
            args.output[row * args.output_m_stride + col * args.output_n_stride] =
                    static_cast<float>(sum);
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
