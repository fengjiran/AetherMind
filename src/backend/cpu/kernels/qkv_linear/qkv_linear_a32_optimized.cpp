#include "backend/cpu/kernels/linear/linear_internal.h"
#include "qkv_linear_internal.h"
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

template<bool Reduced, auto Runner, class Args, class Weight>
Status RunProjection(const Args& args, const Weight* weight, float* output,
                     int64_t n, int64_t output_row_stride, int64_t output_col_stride) noexcept {
    using LinearArgs = std::conditional_t<Reduced, LinearA32W16KernelArgs, LinearA32W32KernelArgs>;
    LinearArgs projection{
            .input = args.input,
            .weight = weight,
            .output = output,
            .row_count = args.row_count,
            .in_features = args.in_features,
            .out_features = n,
            .input_row_stride = args.input_row_stride,
            .input_col_stride = args.input_col_stride,
            .weight_row_stride = args.in_features,
            .weight_col_stride = 1,
            .output_row_stride = output_row_stride,
            .output_col_stride = output_col_stride,
    };
    if constexpr (Reduced) {
        projection.weight_dtype = args.weight_dtype;
    }
    return Runner(projection);
}

template<bool Reduced, auto Runner, class Args>
Status RunProjections(const Args& args) noexcept {
    if (args.row_count == 0) {
        return Status::Ok();
    }
    AM_RETURN_IF_ERROR((RunProjection<Reduced, Runner>(args, args.q_weight, args.query, args.q_out_features, args.query_row_stride, args.query_col_stride)));
    AM_RETURN_IF_ERROR((RunProjection<Reduced, Runner>(args, args.k_weight, args.key, args.k_out_features, args.key_row_stride, args.key_col_stride)));
    return RunProjection<Reduced, Runner>(args, args.v_weight, args.value, args.v_out_features, args.value_row_stride, args.value_col_stride);
}

} // namespace

Status RunQkvLinearF32Scalar(const QkvLinearA32W32KernelArgs& args) noexcept {
    return RunProjections<false, &RunLinearF32Scalar>(args);
}

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunQkvLinearF32Avx2Fma(const QkvLinearA32W32KernelArgs& args) noexcept {
    return RunProjections<false, &RunLinearF32Avx2Fma>(args);
}
#endif

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunQkvLinearF32Bf16Avx2Fma(const QkvLinearA32W16KernelArgs& args) noexcept {
    return RunProjections<true, &RunLinearF32Bf16Avx2Fma>(args);
}
#endif

#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
Status RunQkvLinearF32F16Avx2Fma(const QkvLinearA32W16KernelArgs& args) noexcept {
    return RunProjections<true, &RunLinearF32F16Avx2Fma>(args);
}
#endif

} // namespace aethermind::cpu::detail
