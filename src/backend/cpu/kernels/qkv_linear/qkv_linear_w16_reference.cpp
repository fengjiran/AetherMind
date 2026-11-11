#include "backend/cpu/kernels/linear/linear_internal.h"
#include "qkv_linear_internal.h"

namespace aethermind::cpu::detail {
namespace {

Status RunProjection(const QkvLinearF32W16KernelArgs& args,
                     const std::byte* weight,
                     float* output,
                     int64_t out_features,
                     int64_t output_row_stride,
                     int64_t output_col_stride) noexcept {
    return RunLinearF32W16Reference(LinearF32W16KernelArgs{
            .input = args.input,
            .weight = weight,
            .output = output,
            .weight_dtype = args.weight_dtype,
            .row_count = args.row_count,
            .in_features = args.in_features,
            .out_features = out_features,
            .input_row_stride = args.input_row_stride,
            .input_col_stride = args.input_col_stride,
            .weight_row_stride = args.in_features,
            .weight_col_stride = 1,
            .output_row_stride = output_row_stride,
            .output_col_stride = output_col_stride,
    });
}

} // namespace

Status RunQkvLinearF32W16Reference(const QkvLinearF32W16KernelArgs& args) noexcept {
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