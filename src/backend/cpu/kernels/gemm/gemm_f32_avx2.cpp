#include "../common/dot_product_internal.h"
#include "gemm_internal.h"

#include <cstddef>

namespace aethermind::cpu::detail {

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGemmF32Avx2Fma(const GemmF32Args& args) noexcept {
    if (args.m == 1 && args.k > 0 && args.n >= 0 && args.lhs_k_stride == 1 &&
        args.rhs_k_stride == 1 && args.output_n_stride == 1) {
        // The M=1 fast path is the shared multi-target dot product: one lhs
        // row against every N-contiguous rhs column. k and the strides are
        // validated above, so the signed-to-unsigned casts are safe.
        DotProductF32MultiTargetAvx2(args.lhs, args.rhs,
                                     static_cast<std::size_t>(args.k),
                                     static_cast<std::size_t>(args.n),
                                     static_cast<std::size_t>(args.rhs_n_stride),
                                     args.output);
        return Status::Ok();
    }
    return RunGemmF32Scalar(args);
}
#endif

} // namespace aethermind::cpu::detail