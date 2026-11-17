#include "linear_internal.h"

namespace aethermind::cpu::detail {
namespace {

template<class Args>
GemmArgs MakeLinearGemmArgs(const Args& args) noexcept {
    return GemmArgs{
            .lhs = reinterpret_cast<const std::byte*>(args.input),
            .rhs = reinterpret_cast<const std::byte*>(args.weight),
            .output = reinterpret_cast<std::byte*>(args.output),
            .m = args.row_count,
            .n = args.out_features,
            .k = args.in_features,
            .lhs_m_stride = args.input_row_stride,
            .lhs_k_stride = args.input_col_stride,
            .rhs_k_stride = args.weight_col_stride,
            .rhs_n_stride = args.weight_row_stride,
            .output_m_stride = args.output_row_stride,
            .output_n_stride = args.output_col_stride,
    };
}

} // namespace

Status RunLinearF32Scalar(const LinearA32W32KernelArgs& args) noexcept {
    return RunGemmF32Scalar(MakeLinearGemmArgs(args));
}

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunLinearF32Avx2Fma(const LinearA32W32KernelArgs& args) noexcept {
    return RunGemmF32Avx2Fma(MakeLinearGemmArgs(args));
}

Status RunLinearF32Bf16Avx2Fma(const LinearA32W16KernelArgs& args) noexcept {
    if (!TryRunGemmA32Bf16Avx2Fma(MakeLinearGemmArgs(args))) {
        return RunLinearF32W16Reference(args);
    }
    return Status::Ok();
}
#endif

#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
Status RunLinearF32F16Avx2Fma(const LinearA32W16KernelArgs& args) noexcept {
    if (!TryRunGemmA32F16Avx2Fma(MakeLinearGemmArgs(args))) {
        return RunLinearF32W16Reference(args);
    }
    return Status::Ok();
}
#endif

} // namespace aethermind::cpu::detail
