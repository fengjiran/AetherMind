#include "gemm_internal.h"

namespace aethermind::cpu::detail {

Status RunGemmA32W32Reference(const GemmArgs& args) noexcept {
    const auto* const lhs = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs = reinterpret_cast<const float*>(args.rhs);
    auto* const output = reinterpret_cast<float*>(args.output);

    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }

    for (int64_t r = 0; r < args.m; ++r) {
        for (int64_t c = 0; c < args.n; ++c) {
            double sum = 0.0;
            for (int64_t inner = 0; inner < args.k; ++inner) {
                sum += static_cast<double>(lhs[r * args.lhs_m_stride + inner * args.lhs_k_stride]) *
                       static_cast<double>(rhs[inner * args.rhs_k_stride + c * args.rhs_n_stride]);
            }
            output[r * args.output_m_stride + c * args.output_n_stride] = static_cast<float>(sum);
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
