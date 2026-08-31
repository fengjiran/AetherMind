#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/operators/op_params.h"
#include "rmsnorm_internal.h"
#include "utils/overflow_check.h"

#include <cmath>
#include <cstring>

namespace aethermind::cpu::detail {
namespace {

const RmsNormKernelParams* GetParams(const void* kernel_params) noexcept {
    return static_cast<const RmsNormKernelParams*>(kernel_params);
}

bool HasUnitColumnStrides(const RmsNormFp32KernelArgs& args) noexcept {
    return args.input_col_stride == 1 && args.weight_stride == 1 && args.output_col_stride == 1;
}

StatusOr<int64_t> ComputeRowCount(const TensorView& input) noexcept {
    int64_t row_count = 1;
    for (int32_t i = 0; i < input.rank() - 1; ++i) {
        const int64_t extent = input.dim(i);
        if (extent == 0) {
            return int64_t{0};
        }

        int64_t next_row_count = 0;
        if (CheckOverflowMul(row_count, extent, &next_row_count)) {
            return Status::InvalidArgument("RmsNormKernelEntry row count overflow");
        }
        row_count = next_row_count;
    }
    return row_count;
}

Status ValidatePositiveStrides(const TensorView& tensor, const char* message) noexcept {
    for (int32_t dim = 0; dim < tensor.rank(); ++dim) {
        if (tensor.stride(dim) <= 0) {
            return Status::InvalidArgument(message);
        }
    }
    return Status::Ok();
}

Status ValidatePositiveStrides(const MutableTensorView& tensor, const char* message) noexcept {
    for (int32_t i = 0; i < tensor.rank(); ++i) {
        if (tensor.stride(i) <= 0) {
            return Status::InvalidArgument(message);
        }
    }
    return Status::Ok();
}

Status ValidateMaxOffset(int64_t row_count,
                         int64_t hidden_size,
                         int64_t row_stride,
                         int64_t column_stride,
                         const char* message) noexcept {
    int64_t row_offset = 0;
    if (CheckOverflowMul(row_count - 1, row_stride, &row_offset)) {
        return Status::InvalidArgument(message);
    }

    int64_t column_offset = 0;
    if (CheckOverflowMul(hidden_size - 1, column_stride, &column_offset)) {
        return Status::InvalidArgument(message);
    }

    int64_t max_offset = 0;
    if (CheckOverflowAdd(row_offset, column_offset, &max_offset)) {
        return Status::InvalidArgument(message);
    }
    return Status::Ok();
}

template<typename TensorLike>
Status ValidateCollapsibleLeadingDimensions(const TensorLike& tensor) noexcept {
    for (int32_t i = 0; i < tensor.rank() - 2; ++i) {
        int64_t expected_stride = 0;
        if (CheckOverflowMul(tensor.dim(i + 1), tensor.stride(i + 1), &expected_stride)) {
            return Status::InvalidArgument(
                    "RmsNormKernelEntry leading-dimension stride product overflow");
        }

        if (tensor.stride(i) != expected_stride) {
            return Status::Unimplemented(
                    "CPU RmsNorm requires collapsible leading dimensions for rank > 2");
        }
    }
    return Status::Ok();
}

Status ValidateAndBuildRmsNormFp32Args(const KernelContext& ctx,
                                       RmsNormFp32KernelArgs& args) noexcept {
    float eps = 0.0f;
    if (ctx.attrs.size() != sizeof(float)) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires epsilon in KernelContext.attrs");
    }

    std::memcpy(&eps, ctx.attrs.data(), sizeof(float));
    if (!std::isfinite(eps) || eps <= 0.0F) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires finite positive epsilon");
    }

    const RmsNormKernelParams* params = GetParams(ctx.kernel_params);
    if (params == nullptr) {
        return Status::InvalidArgument("RmsNormKernelEntry requires RmsNormKernelParams "
                                       "in KernelContext.kernel_params");
    }

    const TensorView& input = params->input_tensor;
    const TensorView& weight = params->weight_tensor;
    const MutableTensorView& output = params->output_tensor;
    if (!input.is_valid()) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires a valid input TensorView");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires a valid weight TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires a valid output MutableTensorView");
    }

    if (auto expect_dtype = DataType::Float32();
        input.dtype() != expect_dtype || weight.dtype() != expect_dtype || output.dtype() != expect_dtype) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires float32 input, weight, and output TensorViews");
    }

    const int32_t rank = input.rank();
    if (rank < 1) {
        return Status::InvalidArgument("RmsNormKernelEntry requires input rank >= 1");
    }

    if (weight.rank() != 1) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires rank-1 weight TensorView");
    }

    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires output rank to match input rank");
    }

    for (int32_t i = 0; i < rank; ++i) {
        if (output.dim(i) != input.dim(i)) {
            return Status::InvalidArgument(
                    "RmsNormKernelEntry requires output shape to match input shape");
        }
    }

    const int64_t hidden_size = input.dim(rank - 1);
    if (hidden_size <= 0) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires positive hidden_size");
    }

    if (weight.dim(0) != hidden_size) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires weight length to match hidden_size");
    }

    const StatusOr<int64_t> row_count = ComputeRowCount(input);
    if (!row_count.ok()) {
        return row_count.status();
    }

    if (row_count.value() == 0) {
        args = RmsNormFp32KernelArgs{
                .row_count = 0,
                .hidden_size = hidden_size,
                .eps = eps,
        };
        return Status::Ok();
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires non-null data pointers for non-empty tensors");
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            input, "RmsNormKernelEntry requires positive input strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            weight, "RmsNormKernelEntry requires positive weight strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            output, "RmsNormKernelEntry requires positive output strides"));

    if (rank > 2) {
        AM_RETURN_IF_ERROR(ValidateCollapsibleLeadingDimensions(input));
        AM_RETURN_IF_ERROR(ValidateCollapsibleLeadingDimensions(output));
    }

    const int64_t input_row_stride = rank == 1 ? 0 : input.stride(rank - 2);
    const int64_t output_row_stride = rank == 1 ? 0 : output.stride(rank - 2);
    AM_RETURN_IF_ERROR(ValidateMaxOffset(
            row_count.value(),
            hidden_size,
            input_row_stride,
            input.stride(rank - 1),
            "RmsNormKernelEntry input offset overflow"));
    AM_RETURN_IF_ERROR(ValidateMaxOffset(
            1,
            hidden_size,
            0,
            weight.stride(0),
            "RmsNormKernelEntry weight offset overflow"));
    AM_RETURN_IF_ERROR(ValidateMaxOffset(
            row_count.value(),
            hidden_size,
            output_row_stride,
            output.stride(rank - 1),
            "RmsNormKernelEntry output offset overflow"));

    args = RmsNormFp32KernelArgs{
            .input = input.data<float>(),
            .weight = weight.data<float>(),
            .output = output.data<float>(),
            .row_count = row_count.value(),
            .hidden_size = hidden_size,
            .input_row_stride = input_row_stride,
            .input_col_stride = input.stride(rank - 1),
            .weight_stride = weight.stride(0),
            .output_row_stride = output_row_stride,
            .output_col_stride = output.stride(rank - 1),
            .eps = eps,
    };
    return Status::Ok();
}

Status BuildRmsNormKernelParams(std::span<const TensorView> inputs,
                                std::span<const MutableTensorView> outputs,
                                void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("RmsNorm requires 2 inputs and 1 output");
    }

    ::new (params_buffer) RmsNormKernelParams{
            .input_tensor = inputs[0],
            .weight_tensor = inputs[1],
            .output_tensor = outputs[0],
    };
    return Status::Ok();
}

Status BuildRmsNormMetadata(const OpParams& params,
                            std::vector<std::byte>& attrs) {
    const auto* rmsnorm_params = std::get_if<RmsNormParams>(&params);
    if (rmsnorm_params == nullptr) {
        return Status::InvalidArgument("RmsNorm kernel requires RmsNormParams");
    }

    if (!std::isfinite(rmsnorm_params->eps) || rmsnorm_params->eps <= 0.0F) {
        return Status::InvalidArgument("RmsNorm kernel requires finite positive epsilon");
    }

    const auto eps_bytes = std::as_bytes(std::span{&rmsnorm_params->eps, size_t{1}});
    attrs.assign(eps_bytes.begin(), eps_bytes.end());
    return Status::Ok();
}

Status RmsNormKernelEntryFp32Scalar(const KernelContext& ctx) noexcept {
    RmsNormFp32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildRmsNormFp32Args(ctx, args));
    if (args.row_count == 0) {
        return Status::Ok();
    }
    return RunRmsNormFp32Scalar(args);
}

#if defined(AETHERMIND_HAS_RMSNORM_AVX2_FMA_KERNEL)
Status RmsNormKernelEntryFp32Avx2Fma(const KernelContext& ctx) noexcept {
    RmsNormFp32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildRmsNormFp32Args(ctx, args));
    if (args.row_count == 0) {
        return Status::Ok();
    }

    if (!HasUnitColumnStrides(args)) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry AVX2 requires unit column strides");
    }
    return RunRmsNormFp32Avx2Fma(args);
}
#endif

}// namespace

AM_REGISTER_KERNEL(RmsNormFp32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kRmsNorm,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &RmsNormKernelEntryFp32Scalar,
                           .name = "cpu::rmsnorm_f32_scalar",
                           .priority = 10,
                           .params_builder = &BuildRmsNormKernelParams,
                           .params_size = sizeof(RmsNormKernelParams),
                           .metadata_builder = &BuildRmsNormMetadata,
                   });

#if defined(AETHERMIND_HAS_RMSNORM_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(RmsNormFp32Avx2Fma,
                   KernelDescriptor{
                           .op_type = OpType::kRmsNorm,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .cpu_requirements = {
                                   .all_of = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                           },
                           .kernel_func = &RmsNormKernelEntryFp32Avx2Fma,
                           .name = "cpu::rmsnorm_f32_avx2_fma",
                           .priority = 20,
                           .params_builder = &BuildRmsNormKernelParams,
                           .params_size = sizeof(RmsNormKernelParams),
                           .metadata_builder = &BuildRmsNormMetadata,
                   });
#endif

}// namespace aethermind::cpu::detail
