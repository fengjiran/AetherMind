#include "backend/cpu/kernels/gemm/gemm_internal.h"
#include "qkv_linear_internal.h"

namespace aethermind::cpu::detail {
namespace {

Status RunProjection(const QkvLinearF32KernelArgs& args,
                     const float* weight,
                     float* output,
                     int64_t out_features,
                     int64_t output_row_stride,
                     int64_t output_col_stride) noexcept {
    return RunGemmF32Reference(GemmF32Args{
            .lhs = args.in_features == 0 ? nullptr : args.input,
            .rhs = args.in_features == 0 ? nullptr : weight,
            .output = output,
            .m = args.row_count,
            .n = out_features,
            .k = args.in_features,
            .lhs_m_stride = args.input_row_stride,
            .lhs_k_stride = args.input_col_stride,
            .rhs_k_stride = 1,
            .rhs_n_stride = args.in_features,
            .output_m_stride = output_row_stride,
            .output_n_stride = output_col_stride,
    });
}

} // namespace

Status RunQkvLinearF32Reference(const QkvLinearF32KernelArgs& args) noexcept {
    if (args.row_count == 0) {
        return Status::Ok();
    }

    AM_RETURN_IF_ERROR(RunProjection(args, args.q_weight, args.query,
                                     args.q_out_features, args.query_row_stride,
                                     args.query_col_stride));
    AM_RETURN_IF_ERROR(RunProjection(args, args.k_weight, args.key,
                                     args.k_out_features, args.key_row_stride,
                                     args.key_col_stride));
    return RunProjection(args, args.v_weight, args.value,
                         args.v_out_features, args.value_row_stride,
                         args.value_col_stride);
}

} // namespace aethermind::cpu::detail
