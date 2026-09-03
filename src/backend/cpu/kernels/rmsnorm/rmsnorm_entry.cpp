#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "rmsnorm_internal.h"
#include "utils/overflow_check.h"

#include <cmath>
#include <cstring>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

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

bool HasIdenticalMapping(const TensorView& input,
                         const MutableTensorView& output) noexcept {
    if (input.data() != output.data() || input.dtype() != output.dtype() ||
        input.rank() != output.rank()) {
        return false;
    }

    for (int32_t i = 0; i < input.rank(); ++i) {
        if (input.dim(i) != output.dim(i) || input.stride(i) != output.stride(i)) {
            return false;
        }
    }

    return true;
}

StatusOr<int64_t> ComputeRowSpan(int64_t hidden_size,
                                 int64_t column_stride) noexcept {
    int64_t last_column_offset = 0;
    if (CheckOverflowMul(hidden_size - 1, column_stride, &last_column_offset)) {
        return Status::InvalidArgument(
                "RmsNorm output row span overflow");
    }

    int64_t row_span = 0;
    if (CheckOverflowAdd(last_column_offset, int64_t{1}, &row_span)) {
        return Status::InvalidArgument(
                "RmsNorm output row span overflow");
    }

    return row_span;
}

Status ValidateNonOverlappingOutputRows(int64_t row_count,
                                        int64_t hidden_size,
                                        int64_t row_stride,
                                        int64_t column_stride) noexcept {
    if (row_count <= 1) {
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_span, ComputeRowSpan(hidden_size, column_stride));

    if (row_stride < row_span) {
        return Status::InvalidArgument("CPU RmsNorm output rows must not overlap");
    }

    return Status::Ok();
}

Status ValidateAndBuildCommonRmsNormFp32Args(
        const KernelParamsBuildContext& context,
        RmsNormFp32KernelArgs& args) noexcept {
    float eps = 0.0f;
    if (context.attrs.size() != sizeof(float)) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires epsilon in KernelContext.attrs");
    }

    std::memcpy(&eps, context.attrs.data(), sizeof(float));
    if (!std::isfinite(eps) || eps <= 0.0F) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires finite positive epsilon");
    }

    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("RmsNorm requires 2 inputs and 1 output");
    }

    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
    const MutableTensorView& output = outputs[0];
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

    AM_RETURN_IF_ERROR(ValidateNonOverlappingOutputRows(
            row_count.value(),
            hidden_size,
            output_row_stride,
            output.stride(rank - 1)));

    if (const bool shares_base_pointer = input.data() == output.data();
        shares_base_pointer && !HasIdenticalMapping(input, output)) {
        return Status::InvalidArgument(
                "CPU RmsNorm in-place execution requires identical "
                "input and output shape/strides");
    }

    if (weight.data() == output.data()) {
        return Status::InvalidArgument(
                "CPU RmsNorm output must not alias weight");
    }

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

Status BuildRmsNormFp32ScalarArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    RmsNormFp32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildCommonRmsNormFp32Args(context, args));
    ::new (params_buffer) RmsNormFp32KernelArgs(args);
    return Status::Ok();
}

Status BuildRmsNormFp32Avx2FmaArgs(const KernelParamsBuildContext& context,
                                   void* params_buffer) noexcept {
    RmsNormFp32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildCommonRmsNormFp32Args(context, args));

    if (args.row_count != 0 && !HasUnitColumnStrides(args)) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry AVX2 requires unit column strides");
    }

    ::new (params_buffer) RmsNormFp32KernelArgs(args);
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
    const auto* args = static_cast<const RmsNormFp32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunRmsNormFp32Scalar(*args);
}

#if defined(AETHERMIND_HAS_RMSNORM_AVX2_FMA_KERNEL)
Status RmsNormKernelEntryFp32Avx2Fma(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const RmsNormFp32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunRmsNormFp32Avx2Fma(*args);
}
#endif

} // namespace

static_assert(std::is_trivially_destructible_v<RmsNormFp32KernelArgs>);
static_assert(alignof(RmsNormFp32KernelArgs) <= alignof(std::max_align_t));

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
                           .priority = 10,
                           .params_size = sizeof(RmsNormFp32KernelArgs),
                           .params_builder = &BuildRmsNormFp32ScalarArgs,
                           .metadata_builder = &BuildRmsNormMetadata,
                           .name = "cpu::rmsnorm_f32_scalar",
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
                           .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                           .kernel_func = &RmsNormKernelEntryFp32Avx2Fma,
                           .priority = 20,
                           .params_size = sizeof(RmsNormFp32KernelArgs),
                           .params_builder = &BuildRmsNormFp32Avx2FmaArgs,
                           .metadata_builder = &BuildRmsNormMetadata,
                           .name = "cpu::rmsnorm_f32_avx2_fma",
                   });
#endif

} // namespace aethermind::cpu::detail
